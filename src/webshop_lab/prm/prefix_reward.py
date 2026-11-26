"""Arm 4 ASIN context and current-prefix value scoring."""
import math
import re

from .tokenize import TrajectoryTooLong, tokenize_trajectory


ASIN_CLICK_RE = re.compile(r"click\[(b[0-9a-z]{9})\]", flags=re.IGNORECASE)


def extract_preclick_asin_snippet(messages: list[dict], msg_idx: int) -> dict | None:
    """Inspect `messages[msg_idx]` and read product title + price strictly from `messages[msg_idx - 1]`."""
    if msg_idx <= 0 or messages[msg_idx].get("role") != "assistant":
        return None
    if messages[msg_idx - 1].get("role") != "user":
        return None
    m_click = ASIN_CLICK_RE.search(messages[msg_idx].get("content", ""))
    if not m_click:
        return None
    asin_raw = m_click.group(1)
    asin_upper = asin_raw.upper()
    prev_page = messages[msg_idx - 1].get("content", "")
    pat = re.compile(
        r"\[SEP\]\s*" + re.escape(asin_upper) + r"\s*\[SEP\]\s*(.*?)\s*\[SEP\]\s*(\$[^\[]+?)(?:\s*\[SEP\]|$)",
        flags=re.DOTALL,
    )
    hit = pat.search(prev_page)
    if not hit:
        return None
    title = " ".join(hit.group(1).split())[:180]
    price = " ".join(hit.group(2).split())[:32]
    augmented_content = f"{messages[msg_idx]['content']} [Item {asin_upper}: {title} | {price}]"
    return {
        "assistant_message_index": msg_idx,
        "source_search_message_index": msg_idx - 1,
        "clicked_asin_raw": asin_raw,
        "matched_asin_upper": asin_upper,
        "source_char_span": [hit.start(), hit.end()],
        "source_verbatim_record": prev_page[hit.start():hit.end()],
        "extracted_title": title,
        "extracted_price": price,
        "original_assistant_content": messages[msg_idx]["content"],
        "augmented_assistant_content": augmented_content,
    }


def encode_trajectory_for_pv(tokenizer, messages: list[dict], max_length: int, attach_asin_context: bool):
    """Encode raw trajectory messages, optionally attaching deterministic pre-click ASIN context."""
    base_enc = tokenize_trajectory(tokenizer, messages, max_length)
    provenance = []
    chosen_enc = base_enc

    if attach_asin_context:
        aug_messages = [dict(m) for m in messages]
        assistant_turn_counter = 0
        extracted_records = []
        for msg_idx, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                info = extract_preclick_asin_snippet(messages, msg_idx)
                if info is not None:
                    info["turn_index"] = assistant_turn_counter
                    aug_messages[msg_idx]["content"] = info["augmented_assistant_content"]
                    extracted_records.append(info)
                assistant_turn_counter += 1

        if extracted_records:
            try:
                cand_enc = tokenize_trajectory(tokenizer, aug_messages, max_length)
                if len(cand_enc["turn_end_indices"]) == len(base_enc["turn_end_indices"]):
                    chosen_enc = cand_enc
                    for rec in extracted_records:
                        t_idx = rec["turn_index"]
                        tok_pos = chosen_enc["turn_end_indices"][t_idx]
                        tok_id = chosen_enc["input_ids"][tok_pos]
                        rec["fallback_to_raw_due_to_max_length"] = False
                        rec["raw_turn_end_token_index"] = base_enc["turn_end_indices"][t_idx]
                        rec["scored_turn_end_token_index"] = tok_pos
                        rec["scored_turn_end_token"] = tokenizer.decode([tok_id])
                        provenance.append(rec)
            except TrajectoryTooLong:
                chosen_enc = base_enc
                for rec in extracted_records:
                    t_idx = rec["turn_index"]
                    tok_pos = base_enc["turn_end_indices"][t_idx]
                    tok_id = base_enc["input_ids"][tok_pos]
                    rec["fallback_to_raw_due_to_max_length"] = True
                    rec["raw_turn_end_token_index"] = tok_pos
                    rec["scored_turn_end_token_index"] = tok_pos
                    rec["scored_turn_end_token"] = tokenizer.decode([tok_id])
                    provenance.append(rec)

    return chosen_enc, provenance



def encode_current_prefix(tokenizer, messages, max_length):
    """Choose ASIN augmentation using only the history ending at this action."""
    encoded, provenance = encode_trajectory_for_pv(tokenizer, messages, max_length, True)
    return encoded, any(record["fallback_to_raw_due_to_max_length"] for record in provenance)


class OnlinePrefixReward:
    """Frozen Arm 4 inference with one immutable value per observed action."""
    def __init__(self, config, checkpoint):
        import torch
        from .config import load_config
        from .runtime import build_prm

        self.config = load_config(config)
        self.model, self.tokenizer, _, _ = build_prm(self.config, checkpoint, "cuda", False)
        self.device = next(self.model.parameters()).device
        self.inference_mode = torch.no_grad

    def __call__(self, messages):
        from .runtime import tensor_batch

        encoded, fallback = encode_current_prefix(self.tokenizer, messages, self.config.max_length)
        batch = tensor_batch([encoded], self.tokenizer.pad_token_id, self.device, False)
        with self.inference_mode():
            value = float(self.model(**batch)["step_scores"][0, -1])
        if not math.isfinite(value):
            raise ValueError("Arm 4 returned a non-finite prefix value")
        return value, fallback
