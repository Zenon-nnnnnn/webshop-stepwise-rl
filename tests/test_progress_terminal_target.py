import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from webshop_lab.progress import dot, total_features, train


def episode(task, done, score):
    return {"task_id": task, "instruction": f"Find item {task}",
            "done": done, "final_reward": score,
            "steps": [{"observation": f"page {task}", "response": "action"}]}


class ProgressTerminalTargetTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        self.validation = episode(3, False, 0.875)
        self.args = SimpleNamespace(train=root / "train.jsonl", val=root / "val.jsonl",
                                    output=root / "progress.json", dimensions=8, epochs=1, seed=42)
        self.write_split(self.args.val, [self.validation])

    @staticmethod
    def write_split(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_training_and_validation_use_purchase_terminal_scores(self):
        self.write_split(self.args.train, [episode(1, True, 0.75), episode(2, False, 0.75)])
        train(self.args)
        model = json.loads(self.args.output.read_text())
        self.assertEqual(model["target"], "terminal_reward")
        prediction = dot(model["weights"], total_features(self.validation, self.args.dimensions))
        self.assertNotEqual(prediction, 0.0)
        self.assertAlmostEqual(model["validation_mse"], prediction ** 2)

    def test_page_scores_do_not_create_training_target_variation(self):
        self.write_split(self.args.train, [episode(1, False, 0.25), episode(2, False, 0.75)])
        with self.assertRaisesRegex(ValueError, "rewards have no variation"):
            train(self.args)
