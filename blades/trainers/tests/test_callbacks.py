import unittest

import ray
import torch

from blades.trainers import TrainerCallback
from blades.trainers.fedavg import FedavgTrainerConfig
from blades.datasets import DatasetCatalog, ToyFLDataset


class InitCallbacks(TrainerCallback):
    def on_trainer_init(self, *, trainer, **kwargs):
        self._on_init_was_called = True


class TestCallbacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init()
        DatasetCatalog.register_custom_dataset("simple", ToyFLDataset)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_on_trainer_init(self):
        model = torch.nn.Linear(2, 2)
        algo = (
            FedavgTrainerConfig()
            .resources(num_remote_workers=2, num_gpus_per_worker=0)
            .data(
                num_clients=1,
                dataset_config={"custom_dataset": "simple"},
            )
            .training(global_model=model)
            .callbacks(InitCallbacks)
            .build()
        )

        self.assertTrue(algo.callbacks[0]._on_init_was_called)
