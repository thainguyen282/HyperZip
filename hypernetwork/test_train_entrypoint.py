"""Configuration and dispatch checks that do not load an 8B model."""
import unittest
from unittest.mock import patch

from .train_config import load_nemotron_config


class TrainEntrypointTests(unittest.TestCase):
    def test_architecture_and_dataset_shortcuts(self):
        cfg = load_nemotron_config([
            "--data", "data/hypernetwork_fineweb_1367451",
            "--output", "checkpoints/entrypoint-test",
            "--steps", "3", "--width", "96", "--depth", "3",
            "--rank", "8", "--alpha", "8", "--global-batch-size", "4",
            "--wandb.enable", "false",
        ])
        for name in ("model", "dataset", "validation_dataset"):
            self.assertEqual(cfg.get(f"{name}.data_dir"), "data/hypernetwork_fineweb_1367451")
        self.assertEqual(cfg.get("checkpoint.checkpoint_dir"), "checkpoints/entrypoint-test")
        self.assertEqual(cfg.get("step_scheduler.max_steps"), 3)
        self.assertEqual(cfg.get("step_scheduler.global_batch_size"), 4)
        self.assertEqual([cfg.get(f"model.{name}") for name in ("width", "depth", "rank", "alpha")],
                         [96, 3, 8, 8])
        self.assertIsNone(cfg.get("wandb"))

    def test_main_calls_native_recipe(self):
        from . import train
        with patch("hypernetwork.nemotron.HypernetworkDiffusionLMSFTRecipe") as recipe:
            train.main(["--nemotron", "--steps", "2"])
        self.assertEqual(recipe.call_args.args[0].get("step_scheduler.max_steps"), 2)
        recipe.return_value.setup.assert_called_once_with()
        recipe.return_value.run_train_validation_loop.assert_called_once_with()

    def test_unsupported_architecture_is_rejected(self):
        with self.assertRaises(SystemExit):
            load_nemotron_config(["--architecture", "perceiver"])


if __name__ == "__main__":
    unittest.main()
