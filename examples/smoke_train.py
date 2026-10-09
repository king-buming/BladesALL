"""Two CPU rounds with local data, a sign-flip attack and a median defense.

Run from the project root: python -m examples.smoke_train
No dataset download, GPU, remote workers or external logging service is needed.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from ray.tune.logger import NoopLogger

from blades.algorithms.fedavg import FedavgConfig
from blades.datasets import DatasetCatalog, FLDataset
from blades.datasets.partitioners import IIDPartitioner


class ToySamples(Dataset):
    def __init__(self):
        self.features = torch.tensor(
            [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]
        ).repeat(6, 1)
        self.targets = torch.tensor([1, 0, 0, 1]).repeat(6)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


class SmokeDataset(FLDataset):
    num_clients = 3

    def __init__(self):
        samples = ToySamples()
        clients = IIDPartitioner(num_clients=self.num_clients).generate_client_datasets(
            samples, samples, train_batch_size=4, test_batch_size=4
        )
        super().__init__(clients)


def run_smoke(rounds=2):
    torch.manual_seed(1234)
    torch.set_num_threads(1)
    DatasetCatalog.register_custom_dataset("bladesall_smoke", SmokeDataset)
    config = (
        FedavgConfig()
        .resources(num_remote_workers=0, num_gpus_per_worker=0, num_gpus_for_driver=0)
        .data(dataset_config={"custom_dataset": "bladesall_smoke"})
        .training(
            global_model=torch.nn.Linear(2, 2),
            server_config={
                "aggregator": {"type": "Median"},
                "optimizer": {"type": "SGD", "lr": 0.1},
            },
        )
        .client(client_config={"lr": 0.1, "num_batch_per_round": 1})
        .adversary(
            num_malicious_clients=1,
            adversary_config={"type": "blades.adversaries.SignFlipAdversary"},
        )
        .evaluation(evaluation_interval=1)
    )
    logdir = Path(__file__).resolve().parents[1] / "outputs" / "smoke"
    logdir.mkdir(parents=True, exist_ok=True)
    trainer = config.build(logger_creator=lambda cfg: NoopLogger(cfg, str(logdir)))
    initial_weights = trainer.server.get_global_model().weight.detach().clone()
    results = []
    try:
        for _ in range(rounds):
            result = trainer.train()
            results.append({
                "round": result["training_iteration"],
                "accuracy": float(result["acc_top_1"]),
                "loss": float(result["ce_loss"]),
            })
        assert not torch.equal(initial_weights, trainer.server.get_global_model().weight)
        return results
    finally:
        trainer.stop()


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2))
