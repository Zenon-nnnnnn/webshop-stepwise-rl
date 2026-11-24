import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from webshop_lab.rewards import run


class RewardExportTests(unittest.TestCase):
    def test_page_match_does_not_pay_for_an_unfinished_episode(self):
        steps = [{"observation": "page", "response": "action", "grounding": 1,
                  "progress": 0.125}] * 2
        rows = [{"task_id": task, "instruction": "Find an item", "steps": steps,
                 "done": done, "final_reward": score,
                 "termination": "purchase" if done else "max_steps"}
                for task, done, score in [(1, False, 0.875), (2, True, 0.75)]]
        for mode, expected in {
            "terminal": [[0.0, 0.0], [0.0, 0.75]],
            "uniform": [[0.0, 0.0], [0.375, 0.375]],
            "spa": [[0.625, 0.625], [0.625, 0.625]],
        }.items():
            with self.subTest(mode=mode), TemporaryDirectory() as folder:
                source = Path(folder) / "episodes.jsonl"
                output = Path(folder) / "rewards.json"
                source.write_text("".join(json.dumps(row) + "\n" for row in rows))
                run(SimpleNamespace(input=source, output=output, mode=mode, alpha=1.0, beta=0.5))
                exported = json.loads(output.read_text())
                self.assertEqual([row["rewards"] for row in exported], expected)
                self.assertEqual([row["terminal_reward"] for row in exported], [0.0, 0.75])
                self.assertEqual([row["final_reward"] for row in exported], [0.875, 0.75])
