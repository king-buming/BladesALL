"""Public entrypoints must run from a fresh interpreter."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class CommandLineTests(unittest.TestCase):
    def test_utf8_yaml_loads_with_chinese_comments(self):
        from blades.train import SupportedFileType, load_experiments_from_file

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "experiment.yaml"
            config_path.write_text(
                "# 文本分类配置与模型参数\n"
                "example:\n  run: FEDAVG\n  config: {}\n",
                encoding="utf-8",
            )
            experiments = load_experiments_from_file(
                str(config_path), SupportedFileType.yaml
            )
        self.assertEqual(experiments["example"]["run"], "FEDAVG")

    def test_training_finishes_without_a_checkpoint(self):
        from ray.tune.experiment import Trial
        from blades.train import run_blades_experiments

        trial = Trial("FEDAVG", stub=True)
        # Only replace the external Ray cluster and expensive experiment execution.
        with patch("blades.train.ray.init"), patch("blades.train.ray.shutdown"), patch(
            "blades.train.run_experiments", return_value=[trial]
        ):
            run_blades_experiments(
                experiments={}, v=False, vv=False, trace=False,
                ray_num_nodes=None, ray_num_cpus=1, ray_num_gpus=0,
                ray_object_store_memory=None, ray_ui=False, ray_address=None,
                local_mode=False, resume=False, scheduler="FIFO",
                scheduler_config="{}", algo="FEDAVG", wandb_key=None,
            )

    def test_python_config_resolves_configured_trainer(self):
        import torch  # Initialize native libraries before Ray on Windows.
        from blades.algorithms.fedavg import Fedavg
        from blades.train import SupportedFileType, load_experiments_from_file

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "experiment.py"
            config_path.write_text(
                "from blades.algorithms.fedavg import FedavgConfig\n"
                "config = FedavgConfig()\nstop = {'training_iteration': 2}\n",
                encoding="utf-8",
            )
            experiment = load_experiments_from_file(
                str(config_path), SupportedFileType.python
            )["default"]
        self.assertIs(experiment["run"], Fedavg)
        self.assertEqual(experiment["stop"], {"training_iteration": 2})

    def test_module_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "blades", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("file", result.stdout)


if __name__ == "__main__":
    unittest.main()
