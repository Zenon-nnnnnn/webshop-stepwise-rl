"""Shared model loading and history handling for SFT and rollout."""
from .common import goal_text

SYSTEM = ("You are a web shopping agent in a simulated benchmark. Follow the shopping instruction. "
          "Choose an available action. Reply with a short Thought: line followed by exactly one line "
          "Action: search[keywords] or Action: click[value]. Use a clickable value exactly as provided.")


def load(config, training=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    if not torch.cuda.is_available():
        raise RuntimeError('Training/rollout configuration requires a CUDA GPU; run CPU tools separately.')
    name = config['model']
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(torch_dtype=dtype, device_map={'': 0}, trust_remote_code=False,
                  attn_implementation='sdpa')
    if config.get('load_in_4bit', False):
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if training and config.get('load_in_4bit', False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if not training and config.get('adapter'):
        model = PeftModel.from_pretrained(model, config['adapter'])
    model.config.pad_token_id = tok.pad_token_id
    return model, tok, dtype


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
