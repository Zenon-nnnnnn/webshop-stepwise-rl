"""Shared model loading and history handling for SFT and rollout."""
import re
from .common import goal_text

SYSTEM = ("You are a web shopping agent in a simulated benchmark. Follow the shopping instruction. "
          "Choose an available action. Reply with a short Thought: line followed by exactly one line "
          "Action: search[keywords] or Action: click[value]. Use a clickable value exactly as provided.")


def load(config, training=False, *, dtype=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    if not torch.cuda.is_available():
        raise RuntimeError('Training/rollout configuration requires a CUDA GPU; run CPU tools separately.')
    name = config['model']
    policy_dtype = config.get('policy_dtype', 'auto')
    if policy_dtype not in ('auto', 'float32'):
        raise ValueError('policy_dtype must be auto or float32')
    if policy_dtype == 'float32':
        if config.get('load_in_4bit'):
            raise ValueError('float32 policy requires nonquantized weights')
        if dtype is not None and dtype != torch.float32:
            raise ValueError('Explicit dtype conflicts with policy_dtype=float32')
        dtype = torch.float32
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    revision_kwargs = {'revision': config['revision']} if config.get('revision') else {}
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=False, **revision_kwargs)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(torch_dtype=dtype, device_map={'': 0}, trust_remote_code=False,
                  attn_implementation='sdpa')
    kwargs.update(revision_kwargs)
    if config.get('load_in_4bit', False):
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model_source(config, model, tok)  # Reject a resolved commit that contradicts an exact pin.
    if training and config.get('load_in_4bit', False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if not training and config.get('adapter'):
        model = PeftModel.from_pretrained(model, config['adapter'])
    model.config.pad_token_id = tok.pad_token_id
    return model, tok, dtype


def model_source(config, model, tokenizer):
    """Record requested and resolved versions separately; never invent a resolved commit."""
    requested = config.get('revision') or None
    model_commit = getattr(model.config, '_commit_hash', None)
    tokenizer_commit = getattr(tokenizer, 'init_kwargs', {}).get('_commit_hash')
    # A local directory may not expose a Hub commit. Preserve null rather than claiming
    # that the requested revision proves the identity of local files.
    resolved = [value for value in (model_commit, tokenizer_commit) if value]
    if len(set(resolved)) > 1:
        raise ValueError('Model and tokenizer resolved to different Hub commits')
    if requested and re.fullmatch(r'[0-9a-fA-F]{40}', requested):
        if any(value.lower() != requested.lower() for value in resolved):
            raise ValueError('Resolved model/tokenizer commit differs from the requested revision')
    return {'model': config['model'], 'requested_revision': requested,
            'resolved_model_revision': model_commit,
            'resolved_tokenizer_revision': tokenizer_commit,
            'note': 'Null resolved revisions require separate local-file hash evidence; a requested revision is not proof of local file identity.'}


def fit_history(tok, messages, max_tokens, generation=False):
    """Retain system task and latest complete exchanges; never cut raw token strings."""
    messages = [dict(m) for m in messages]
    while True:
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation)
        if len(ids) <= max_tokens:
            return messages, ids
        # Canonical format: system (including task), user/assistant pairs, [user].
        if len(messages) <= (2 if generation else 3):
            raise ValueError('Instruction/current exchange exceeds context budget; increase context or inspect observation.')
        del messages[1:3]


def training_messages(row):
    turns = row['conversations']
    # Source starts with generic instruction + assistant OK + actual shopping instruction.
    if len(turns) >= 4 and turns[1]['value'].strip().upper() == 'OK':
        turns = turns[2:]
    return [{'role': 'system', 'content': SYSTEM + '\nShopping instruction: ' + goal_text(row)}] + [
        {'role': 'user' if t['from'] == 'human' else 'assistant', 'content': t['value']} for t in turns]


def precision_record(model, loaded_dtype):
    """Report the observed parameter mix, including PEFT's possible FP32 adapters."""
    parameters, trainable = {}, {}
    for _, parameter in model.named_parameters():
        key = str(parameter.dtype)
        parameters[key] = parameters.get(key, 0) + parameter.numel()
        if parameter.requires_grad:
            trainable[key] = trainable.get(key, 0) + parameter.numel()
    return {'selected_load_dtype': str(loaded_dtype),
            'parameter_numel_by_dtype': parameters,
            'trainable_numel_by_dtype': trainable}
