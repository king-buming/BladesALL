"""Two CPU rounds with Spectral SPEAR, real DnC, and generated local samples.

Run from the project root: python -m examples.smoke_spectral_spear
No dataset download, GPU, remote workers, or external logging is needed.
This is an integration check, not an attack-effectiveness benchmark.
"""

import json
from pathlib import Path

import torch
from ray.tune.logger import NoopLogger
from torch.utils.data import Dataset

from blades.adversaries.spear_spectral_adversary import SPEARSpectralAdversary
from blades.algorithms.fedavg import FedavgConfig
from blades.datasets import DatasetCatalog, FLDataset
from blades.datasets.partitioners import IIDPartitioner


class SpectralToySamples(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(1234)
        self.features = torch.randn(60, 2, generator=generator)
        self.targets = (self.features[:, 0] + self.features[:, 1] > 0).long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


class SpectralSmokeDataset(FLDataset):
    num_clients = 5

    def __init__(self):
        samples = SpectralToySamples()
        clients = IIDPartitioner(num_clients=self.num_clients).generate_client_datasets(
            samples, samples, train_batch_size=4, test_batch_size=4
        )
        super().__init__(clients)


def run_smoke(rounds=2):
    torch.manual_seed(1234)
    torch.set_num_threads(1)
    DatasetCatalog.register_custom_dataset("bladesall_spectral_smoke", SpectralSmokeDataset)
    # Keep the attacker's proxy and actual defense settings explicit and aligned.
    dnc_sub_dim, dnc_num_iters, dnc_filter_frac = 4, 2, 1.0
    config = (
        FedavgConfig()
        .resources(num_remote_workers=0, num_gpus_per_worker=0, num_gpus_for_driver=0)
        .data(dataset_config={"custom_dataset": "bladesall_spectral_smoke"})
        .training(
            global_model=torch.nn.Linear(2, 2),
            server_config={
                "aggregator": {
                    "type": "DnC",
                    "num_byzantine": 1,
                    "sub_dim": dnc_sub_dim,
                    "num_iters": dnc_num_iters,
                    "filter_frac": dnc_filter_frac,
                },
                "optimizer": {"type": "SGD", "lr": 0.1},
            },
        )
        .client(client_config={"lr": 0.1, "num_batch_per_round": 1})
        .adversary(
            num_malicious_clients=1,
            adversary_config={
                "type": "blades.adversaries.spear_spectral_adversary.SPEARSpectralAdversary",
                "z_values": [0.0, 0.2, 0.5, 1.0],
                "spectral_penalties": [0.0, 1.0, 10.0],
                "spectral_rank": 1,
                "fit_subspaces": 2,
                "dnc_sub_dim": dnc_sub_dim,
                "dnc_num_iters": dnc_num_iters,
                "dnc_filter_frac": dnc_filter_frac,
                "search_banks": 2,
                "audit_banks": 2,
                "seed": 2026,
            },
        )
        .evaluation(evaluation_interval=1)
    )
    logdir = Path(__file__).resolve().parents[1] / "outputs" / "smoke_spectral_spear"
    logdir.mkdir(parents=True, exist_ok=True)
    trainer = config.build(logger_creator=lambda cfg: NoopLogger(cfg, str(logdir)))
    initial_weights = trainer.server.get_global_model().weight.detach().clone()
    results = []
    try:
        attack = next(callback for callback in trainer.callbacks
                      if isinstance(callback, SPEARSpectralAdversary))
        for round_index in range(rounds):
            result = trainer.train()
            diagnostics = attack.last_diagnostics
            assert attack.round_index == round_index + 1
            assert diagnostics["observed_benign"] == 4
            assert diagnostics["participating_malicious"] == 1
            results.append({
                "round": result["training_iteration"],
                "accuracy": float(result["acc_top_1"]),
                "loss": float(result["ce_loss"]),
                "attack_status": diagnostics["status"],
                "proxy_worst_damage": diagnostics["worst_damage"],
                "audit_retained_malicious": diagnostics["audit"]["retained_malicious"],
            })
        assert not torch.equal(initial_weights, trainer.server.get_global_model().weight)
        assert torch.isfinite(trainer.server.get_global_model().weight).all()
        return results
    finally:
        trainer.stop()


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2))
