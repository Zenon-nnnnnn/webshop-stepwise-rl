import unittest
from unittest.mock import Mock, patch

from webshop_lab.prm import prefix_reward
from webshop_lab.prm.tokenize import TrajectoryTooLong


class OnlinePrefixTests(unittest.TestCase):
    def test_later_overflow_does_not_remove_earlier_asin_context(self):
        def encode(_tokenizer, messages, limit):
            content = " ".join(message["content"] for message in messages)
            if len(content) > limit:
                raise TrajectoryTooLong()
            ends = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
            return {"content": content, "turn_end_indices": ends, "input_ids": [0] * len(messages)}

        click = {"role": "assistant", "content": "Action: click[b012345678]"}
        initial = [
            {"role": "system", "content": "Shop"},
            {"role": "user", "content": "[SEP] B012345678 [SEP] Running shoe [SEP] $35"},
            click,
        ]
        later = initial + [
            {"role": "user", "content": "x" * 45},
            {"role": "assistant", "content": "Action: click[Buy Now]"},
        ]
        tokenizer = Mock()
        with patch.object(prefix_reward, "tokenize_trajectory", encode):
            early, early_fallback = prefix_reward.encode_current_prefix(tokenizer, initial, 149)
            late, late_fallback = prefix_reward.encode_current_prefix(tokenizer, later, 149)
            repeated, repeated_fallback = prefix_reward.encode_current_prefix(tokenizer, later[:3], 149)
        assert early == repeated
        assert not early_fallback and not repeated_fallback
        assert late_fallback
        assert "[Item B012345678: Running shoe | $35]" in early["content"]
        assert "[Item " not in late["content"]
        assert click["content"] == "Action: click[b012345678]"
