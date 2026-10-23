"""Public-path compatibility and real local rounds after FedAvg unification.

The synthetic dataset is defined here; these tests need neither downloads nor
a Ray cluster. Fresh interpreters also exercise the formerly cyclic imports.
"""

import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import TensorDataset
import ray
from ray.tune.logger import NoopLogger

from blades.adversaries import Adversary
from blades.constants import CLIENT_ID, CLIENT_UPDATE
from blades.datasets import ClientDataset, DatasetCatalog, FLDataset
from blades.tasks import Classifier, Task
from blades.trainers.fedavg import (
    FedavgCallback,
    FedavgTrainer,
    FedavgTrainerCallbackList,
    FedavgTrainerConfig,
)


ROOT = Path(__file__).resolve().parents[1]


class TinyFederatedDataset(FLDataset):
    num_clients = 3

    def __init__(self):
        samples = TensorDataset(
            torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
            torch.tensor([1, 0, 0, 1]),
        )
        super().__init__([
            ClientDataset(uid, samples, samples, train_batch_size=4, test_batch_size=4)
            for uid in range(self.num_clients)
        ])


class FirstCallback(FedavgCallback):
    def __init__(self):
        super().__init__()
        self.events = []


class SecondCallback(FirstCallback):
    pass


class CustomCallbackList(FedavgTrainerCallbackList):
    pass


class CountingAdversary(Adversary):
    def __init__(self, magnitude=0.5):
        super().__init__()
        self.magnitude = magnitude
        self.calls = 0
        self.events = []

    def on_local_round_end(self, trainer):
        self.calls += 1
        self.events.append("attack")
        update = torch.full_like(self.get_benign_updates(trainer)[0], self.magnitude)
        malicious_ids = {client.client_id for client in self.clients}
        for result in trainer.local_results:
            if result[CLIENT_ID] in malicious_ids:
                result[CLIENT_UPDATE] = update.clone()


class CustomEvaluationTask(Task):
    def compile_eval_results(self, results):
        self.compilations.append(results)
        return {"custom_examples": sum(result["length"] for result in results)}


@contextmanager
def local_cpu_environment():
    rng = torch.get_rng_state()
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    num_threads, num_classes = torch.get_num_threads(), Classifier._num_classes
    torch.set_num_threads(1)
    Classifier._num_classes = None
    try:
        with (
            patch.object(ray, "init", side_effect=AssertionError("Ray cluster not needed")),
            patch.object(
                DatasetCatalog, "from_torch",
                side_effect=AssertionError("Dataset downloads are not allowed"),
            ),
        ):
            yield
    finally:
        torch.set_rng_state(rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
        torch.set_num_threads(num_threads)
        Classifier._num_classes = num_classes


def tiny_config(config_cls):
    torch.manual_seed(987)
    return (
        config_cls()
        .resources(num_remote_workers=0, num_gpus_per_worker=0, num_gpus_for_driver=0)
        .data(dataset_config={"custom_dataset": TinyFederatedDataset, "num_clients": 3})
        .training(
            global_model=torch.nn.Linear(2, 2),
            server_config={"aggregator": {"type": "Mean"}, "optimizer": {"lr": 0.1}},
        )
        .client(client_config={"lr": 0.1, "num_batch_per_round": 1, "momentum": 0.0})
        .evaluation(evaluation_interval=None)
    )


class FedavgUnificationTests(unittest.TestCase):
    def test_legacy_paths_pickle_names_inheritance_and_all_registry_bindings(self):
        from blades.algorithms import registry
        from blades.algorithms.fedavg import Fedavg, FedavgConfig, FedavgDP, FedavgDPConfig
        from blades.algorithms.fedavg.fedavg import Fedavg as old_fedavg
        from blades.algorithms.fedavg.fedavg_dp import FedavgDP as old_dp
        from blades.trainers import FedProxTrainer
        from blades.trainers.fedavg import adversarial, callbacks, fedavg, fedavg_dp
        from blades.trainers import fedavg as canonical

        expected = {
            "Fedavg": (Fedavg, adversarial.Fedavg, old_fedavg),
            "FedavgConfig": (FedavgConfig, adversarial.FedavgConfig),
            "FedavgDP": (FedavgDP, fedavg_dp.FedavgDP, old_dp),
            "FedavgDPConfig": (FedavgDPConfig, fedavg_dp.FedavgDPConfig),
        }
        for name, classes in expected.items():
            with self.subTest(public_name=name):
                for cls in classes:
                    self.assertIs(cls, getattr(canonical, name))
                self.assertIs(pickle.loads(pickle.dumps(classes[0])), classes[0])
        self.assertIs(fedavg.FedavgCallback, callbacks.FedavgCallback)
        self.assertIs(fedavg.FedavgTrainerCallbackList, callbacks.FedavgTrainerCallbackList)
        for module_name, name, cls in (
            ("blades.algorithms.fedavg.fedavg", "Fedavg", Fedavg),
            ("blades.algorithms.fedavg.fedavg_dp", "FedavgDP", FedavgDP),
            ("blades.trainers.fedavg.fedavg", "FedavgCallback", FedavgCallback),
        ):
            legacy_pickle = f"c{module_name}\n{name}\n.".encode("ascii")
            self.assertIs(pickle.loads(legacy_pickle), cls)
        self.assertTrue(issubclass(Fedavg, FedavgTrainer))
        self.assertTrue(issubclass(FedavgDP, Fedavg))
        self.assertTrue(issubclass(FedProxTrainer, FedavgTrainer))
        for name, cls in {
            "FEDAVG": Fedavg, "FEDAVG_BASE": FedavgTrainer,
            "FEDAVG_DP": FedavgDP, "FEDPROX": FedProxTrainer,
        }.items():
            registered, config = registry.ALGORITHMS[name]()
            self.assertIs(registered, cls)
            self.assertIs(config.algo_class, cls)
            self.assertIs(registry._get_algorithm_class(name), cls)

    def test_each_public_import_order_works_in_a_fresh_interpreter(self):
        orders = [
            ["blades.algorithms.fedavg", "blades.trainers", "blades.adversaries"],
            ["blades.trainers", "blades.adversaries", "blades.algorithms.fedavg"],
            ["blades.adversaries", "blades.algorithms.fedavg", "blades.trainers"],
        ]
        for order in orders:
            with self.subTest(first_import=order[0]):
                script = (
                    "import importlib\n"
                    f"for name in {json.dumps(order)}: importlib.import_module(name)\n"
                    "from blades.algorithms.fedavg import Fedavg, FedavgDP\n"
                    "from blades.trainers.fedavg import Fedavg as Canonical\n"
                    "from blades.trainers.fedavg import FedavgDP as CanonicalDP\n"
                    "import ray\n"
                    "assert Fedavg is Canonical and FedavgDP is CanonicalDP\n"
                    "assert not ray.is_initialized()\n"
                )
                result = subprocess.run(
                    [sys.executable, "-c", script], cwd=ROOT,
                    capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_repeated_callback_builds_preserve_frozen_specs_and_make_fresh_instances(self):
        from blades.algorithms.fedavg import FedavgConfig

        spec_variants = [
            (FirstCallback, [FirstCallback]),
            ({"type": FirstCallback}, [FirstCallback]),
            ([FirstCallback, {"type": SecondCallback}], [FirstCallback, SecondCallback]),
            ((FirstCallback, SecondCallback), [FirstCallback, SecondCallback]),
        ]
        for config_cls in (FedavgTrainerConfig, FedavgConfig):
            for specs, ordinary_types in spec_variants:
                with self.subTest(config=config_cls.__name__, specs=type(specs).__name__):
                    config = config_cls()
                    config.callbacks_config = specs
                    if isinstance(config, FedavgConfig):
                        config.adversary(
                            num_malicious_clients=1,
                            adversary_config={"type": CountingAdversary},
                        )
                    config.freeze()
                    before = copy.deepcopy(config.to_dict())
                    first = config.build_callbacks()
                    second = config.build_callbacks(CustomCallbackList)
                    third = config.build_callbacks()
                    expected = ordinary_types + (
                        [CountingAdversary] if isinstance(config, FedavgConfig) else []
                    )
                    self.assertEqual(config.to_dict(), before)
                    self.assertIs(config.callbacks_config, specs)
                    self.assertIs(type(second), CustomCallbackList)
                    for built in (first, second, third):
                        self.assertEqual([type(item) for item in built._callbacks], expected)
                    for a, b, c in zip(first._callbacks, second._callbacks, third._callbacks):
                        self.assertIsNot(a, b)
                        self.assertIsNot(b, c)
                        a.events.append("first-only")
                        self.assertEqual(b.events, [])
                        self.assertEqual(c.events, [])

    def test_two_real_local_rounds_without_attacks_match_the_base_trainer(self):
        from blades.algorithms.fedavg import FedavgConfig

        histories = []
        with local_cpu_environment(), tempfile.TemporaryDirectory() as directory:
            for config_cls in (FedavgTrainerConfig, FedavgConfig):
                config = tiny_config(config_cls)
                trainer = config.build(
                    logger_creator=lambda cfg: NoopLogger(cfg, directory)
                )
                try:
                    model = trainer.server.get_global_model()
                    weights = [{key: value.detach().clone() for key, value in model.state_dict().items()}]
                    for _ in range(2):
                        trainer.train()
                        weights.append({key: value.detach().clone() for key, value in model.state_dict().items()})
                    histories.append(weights)
                finally:
                    trainer.stop()
        for round_index, (base, attack_aware) in enumerate(zip(*histories)):
            for key in base:
                with self.subTest(round=round_index, parameter=key):
                    torch.testing.assert_close(base[key], attack_aware[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(histories[0][0]["weight"], histories[0][1]["weight"]))
        self.assertFalse(torch.equal(histories[0][1]["weight"], histories[0][2]["weight"]))

    def test_nonzero_attack_runs_once_before_aggregation_in_each_real_round(self):
        from blades.algorithms.fedavg import FedavgConfig

        events, uploads = [], []
        with local_cpu_environment(), tempfile.TemporaryDirectory() as directory:
            config = tiny_config(FedavgConfig).adversary(
                num_malicious_clients=1,
                adversary_config={"type": CountingAdversary, "magnitude": 0.5},
            )
            trainer = config.build(logger_creator=lambda cfg: NoopLogger(cfg, directory))
            try:
                attacks = [item for item in trainer.callbacks._callbacks if isinstance(item, CountingAdversary)]
                self.assertEqual(len(attacks), 1)
                attack = attacks[0]
                attack.events = events

                def mean_after_attack(updates):
                    events.append("aggregate")
                    uploads.append(torch.stack(updates).detach().clone())
                    return torch.stack(updates).mean(dim=0)

                trainer.server._aggregator = mean_after_attack
                for _ in range(2):
                    trainer.train()
                self.assertEqual(attack.calls, 2)
                self.assertEqual(events, ["attack", "aggregate", "attack", "aggregate"])
                for upload in uploads:
                    self.assertEqual(upload.shape[0], 3)
                    torch.testing.assert_close(upload[0], torch.full_like(upload[0], 0.5))
                    self.assertFalse(torch.equal(upload[1], upload[0]))
            finally:
                trainer.stop()

    def test_shared_evaluation_keeps_custom_base_and_legacy_attack_aggregation(self):
        from blades.algorithms.fedavg import Fedavg

        self.assertIs(Fedavg.evaluate, FedavgTrainer.evaluate)
        metrics = [
            {"ce_loss": 1.0, "acc_top_1": torch.tensor(0.25), "length": 2},
            {"ce_loss": 5.0, "acc_top_1": torch.tensor(0.75), "length": 6},
        ]
        custom_task = object.__new__(CustomEvaluationTask)
        custom_task.compilations = []
        model = torch.nn.Linear(2, 2)
        evaluated = []

        def client(index):
            def evaluate(loader):
                self.assertEqual(loader, f"loader-{index}")
                evaluated.append(index)
                return metrics[index]
            return SimpleNamespace(client_id=index, evaluate=evaluate)

        worker = SimpleNamespace(dataset=SimpleNamespace(
            get_client_dataset=lambda uid: SimpleNamespace(get_test_loader=lambda: f"loader-{uid}")
        ))
        results = []
        for cls in (FedavgTrainer, Fedavg):
            trainer = object.__new__(cls)
            trainer.server = SimpleNamespace(task=custom_task, get_global_model=lambda: model)
            trainer.client_manager = SimpleNamespace(testable_clients=[client(0), client(1)])
            trainer.worker_group = SimpleNamespace(
                sync_weights=Mock(),
                foreach_execution=lambda fn, clients: [fn(worker, item) for item in clients],
            )
            results.append(trainer.evaluate())
            trainer.worker_group.sync_weights.assert_called_once()
        self.assertEqual(results[0], {"custom_examples": 8})
        self.assertEqual(set(results[1]), {"ce_loss", "acc_top_1"})
        self.assertAlmostEqual(float(results[1]["ce_loss"]), 4.0)
        self.assertAlmostEqual(float(results[1]["acc_top_1"]), 0.625)
        self.assertEqual(len(custom_task.compilations), 1)
        self.assertEqual(evaluated, [0, 1, 0, 1])

    def test_subclass_default_configs_bind_to_the_requested_subclass(self):
        from blades.algorithms.fedavg import Fedavg, FedavgDP
        from blades.trainers import FedProxTrainer

        for parent in (FedavgTrainer, Fedavg, FedavgDP, FedProxTrainer):
            child = type(f"Derived{parent.__name__}", (parent,), {})
            with self.subTest(parent=parent.__name__):
                self.assertIs(child.get_default_config().algo_class, child)

    def test_dp_and_fedprox_keep_their_specialized_client_configuration(self):
        from blades.algorithms.fedavg import FedavgDPConfig
        from blades.clients import DPClient
        from blades.trainers.fedprox import FedProxTrainerConfig
        from blades.trainers.fedprox.fedprox import FedProxClientCallback

        dp = tiny_config(FedavgDPConfig)
        dp.dp_clip_threshold, dp.dp_privacy_epsilon = 0.7, 2.0
        dp.dataset_config["train_bs"] = 4
        dp.validate()
        dp.freeze()
        client_config = dp.get_client_config()
        self.assertAlmostEqual(dp.dp_privacy_sensitivity, 0.35)
        self.assertAlmostEqual(client_config.clip_threshold, 0.7)
        expected_noise = 0.35 * math.sqrt(2 * math.log(1.25 / dp.dp_privacy_delta)) / 2.0
        self.assertAlmostEqual(client_config.noise_factor, expected_noise)
        self.assertIsInstance(client_config.build(), DPClient)
        prox = FedProxTrainerConfig()
        prox.prox_mu = 0.37
        client = prox.get_client_config().build()
        callbacks = [item for item in client.callbacks._callbacks if isinstance(item, FedProxClientCallback)]
        self.assertEqual(len(callbacks), 1)
        self.assertAlmostEqual(callbacks[0].prox_mu, 0.37)
        self.assertIs(callbacks[0]._client, client)


if __name__ == "__main__":
    unittest.main()
