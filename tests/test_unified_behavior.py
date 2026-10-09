"""Exercise the merged package with real CPU objects and no data downloads.

These tests do not start a Ray cluster. The round fixture contains results only;
clients, configuration, callbacks, catalogs, adversaries, and aggregators are the
actual implementations shipped in this checkout.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.abc
import sys
import unittest
from uuid import uuid4


class RejectFedlibImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fedlib" or fullname.startswith("fedlib."):
            raise AssertionError(f"External backend import attempted: {fullname}")
        return None


@contextmanager
def without_external_backend():
    guard = RejectFedlibImports()
    sys.meta_path.insert(0, guard)
    try:
        yield
    finally:
        sys.meta_path.remove(guard)


with without_external_backend():
    import torch
    from ray.rllib.utils.from_config import from_config
    from ray.tune.registry import _global_registry, get_trainable_cls

    from blades.adversaries import AdversaryConfig, IPMAdversary
    from blades.aggregators import Mean, Median
    from blades.algorithms.fedavg import Fedavg, FedavgDP
    from blades.algorithms.registry import ALGORITHMS, _get_algorithm_class
    from blades.clients import (
        BaseClient,
        Client,
        ClientCallback,
        ClientCallbackList,
        ClientConfig,
        ClientProxy,
        ClippingCallback,
        DPClient,
        DPCliengConfig,
    )
    from blades.constants import BLADES_DATASET, BLADES_MODEL, CLIENT_ID, CLIENT_UPDATE
    from blades.datasets import DatasetCatalog, FLDataset, ToyFLDataset
    from blades.models.catalog import ModelCatalog
    from blades.tasks import Classifier
    from blades.trainers import ClientManager, FedavgTrainer, FedProxTrainer, TrainerConfig
    from blades.trainers.fedprox import FedProxTrainerConfig
    from blades.trainers.fedprox.fedprox import FedProxClientCallback


class TinyModel(torch.nn.Linear):
    """A user-defined model registered through the public catalog API."""

    def __init__(self):
        super().__init__(2, 2)


@dataclass
class RoundResults:
    client_manager: ClientManager
    local_results: list


class UnifiedBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.backend_guard = without_external_backend()
        self.backend_guard.__enter__()
        self.addCleanup(self.backend_guard.__exit__, None, None, None)
        self.assertFalse(
            any(name == "fedlib" or name.startswith("fedlib.") for name in sys.modules),
            "The tests must not rely on a previously imported external backend.",
        )

    def test_dynamic_client_config_builds_local_attack_aware_client(self):
        config = (
            ClientConfig("blades.clients.Client")
            .client_id("participant-7")
            .training(lr=0.25, num_batch_per_round=2)
        )
        client = config.build()

        self.assertIs(type(client), Client)
        self.assertIsInstance(client, BaseClient)
        self.assertIsInstance(client.callbacks, ClientCallbackList)
        self.assertEqual(client.client_id, "participant-7")
        self.assertEqual(client.config.lr, 0.25)
        self.assertFalse(client.is_malicious)
        config.lr = 9.0
        self.assertEqual(client.config.lr, 0.25, "Built client config must be independent.")

    def test_malicious_conversion_preserves_identity_and_restore_switch(self):
        client = ClientConfig("blades.clients.Client").client_id("attacker").build()
        with self.assertRaises(ValueError):
            client.attack()

        client.to_malicious(local_training=False)
        self.assertIsInstance(client, ClientProxy)
        self.assertTrue(client.is_malicious)
        self.assertEqual(client.client_id, "attacker")
        self.assertTrue(client._is_under_attack)
        self.assertIsInstance(client.benign_client, BaseClient)
        self.assertEqual(client.train_one_round(None), {CLIENT_ID: "attacker"})

        client.restore()
        self.assertFalse(client._is_under_attack)
        client.attack()
        self.assertTrue(client._is_under_attack)

    def test_base_callback_list_executes_blades_gradient_clipping(self):
        client = ClientConfig("blades.clients.Client").client_id("clipped").build()
        task = TrainerConfig().training(global_model=TinyModel()).get_task_spec().build("cpu")
        self.assertIsInstance(task, Classifier)
        callback = ClippingCallback(clip_threshold=0.5)
        callbacks = ClientCallbackList([ClientCallback()])
        callbacks.setup(client)
        callbacks.append(callback)
        self.assertIs(callback._client, client)

        for parameter in task.model.parameters():
            parameter.grad = torch.full_like(parameter, 20.0)
        callbacks.on_backward_end(task)
        gradient = torch.cat([parameter.grad.flatten() for parameter in task.model.parameters()])
        self.assertAlmostEqual(float(gradient.norm()), 0.5, places=5)

        data = torch.tensor([[1.0, 2.0]])
        labels = torch.tensor([1])
        returned_data, returned_labels = callbacks.on_train_batch_begin(data, labels)
        self.assertIs(returned_data, data)
        self.assertIs(returned_labels, labels)

    def test_dp_client_inherits_merged_client_and_clips_real_update(self):
        client = (
            DPCliengConfig()
            .client_id("private")
            .training(clip_threshold=1.0, noise_factor=0.0)
            .build()
        )
        self.assertIsInstance(client, DPClient)
        self.assertIsInstance(client, Client)
        self.assertIsInstance(client, BaseClient)
        client.pseudo_grad_vec = torch.tensor([3.0, 4.0])
        client.callbacks.on_train_round_end()
        torch.testing.assert_close(client.pseudo_grad_vec, torch.tensor([0.6, 0.8]))

    def test_algorithm_registry_and_cli_preserve_distinct_trainers(self):
        # Importing the CLI registers classes with Tune without starting Ray.
        from blades.train import _register_all

        _register_all()
        expected = {
            "FEDAVG": Fedavg,
            "FEDAVG_BASE": FedavgTrainer,
            "FEDPROX": FedProxTrainer,
            "FEDAVG_DP": FedavgDP,
        }
        for name, cls in expected.items():
            with self.subTest(algorithm=name):
                registered_cls, config = ALGORITHMS[name]()
                self.assertIs(registered_cls, cls)
                self.assertIs(config.algo_class, cls)
                self.assertIs(_get_algorithm_class(name), cls)
                self.assertIs(get_trainable_cls(name), cls)

    def test_fedprox_default_config_installs_requested_proximal_callback(self):
        config = FedProxTrainer.get_default_config()
        self.assertIsInstance(config, FedProxTrainerConfig)
        self.assertIs(config.algo_class, FedProxTrainer)
        config.update_from_dict({"prox_mu": 0.37})
        client = config.get_client_config().build()
        callbacks = [
            callback
            for callback in client.callbacks._callbacks
            if isinstance(callback, FedProxClientCallback)
        ]
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(callbacks[0].prox_mu, 0.37)
        self.assertIs(callbacks[0]._client, client)

    def test_custom_model_and_dotted_task_server_aggregator_config(self):
        name = f"unified_model_{uuid4().hex}"
        ModelCatalog.register_custom_model(name, TinyModel)
        self.addCleanup(_global_registry.unregister, BLADES_MODEL, name)
        model = ModelCatalog.get_model({"custom_model": name})
        self.assertIsInstance(model, TinyModel)
        self.assertEqual(tuple(model(torch.zeros(3, 2)).shape), (3, 2))

        config = TrainerConfig().training(
            global_model={"custom_model": name},
            server_config={"aggregator": {"type": "blades.aggregators.Median"}},
        )
        config.freeze()
        server = config.get_server_config().build(device="cpu")
        self.assertEqual(type(server).__module__, "blades.trainers.server")
        self.assertIsInstance(server.task, Classifier)
        self.assertIsInstance(server.get_global_model(), TinyModel)
        self.assertIsInstance(server.aggregator, Median)
        torch.testing.assert_close(
            server.aggregator([torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([99.0])]),
            torch.tensor([2.0]),
        )

    def test_custom_dataset_registration_and_dotted_path_need_no_download(self):
        name = f"unified_dataset_{uuid4().hex}"
        DatasetCatalog.register_custom_dataset(name, ToyFLDataset)
        self.addCleanup(_global_registry.unregister, BLADES_DATASET, name)
        for spec in (name, "blades.datasets.ToyFLDataset"):
            with self.subTest(dataset=spec):
                dataset = DatasetCatalog.get_dataset({"custom_dataset": spec})
                self.assertIsInstance(dataset, ToyFLDataset)
                self.assertIsInstance(dataset, FLDataset)
                self.assertEqual(len(dataset.client_ids), 3)
                train, test = dataset.to_torch_datasets()
                self.assertEqual(len(train), 3)
                self.assertEqual(len(test), 3)
                sample, target = train[0]
                self.assertEqual(tuple(sample.shape), (2,))
                self.assertIn(int(target), (0, 1))

    def test_ipm_attack_and_aggregators_share_real_clients_and_tensors(self):
        manager = ClientManager([], [], [], ClientConfig("blades.clients.Client"))
        for uid in range(3):
            manager.add_client(ClientConfig("blades.clients.Client").client_id(uid).build())
        manager.get_client_by_id(0).to_malicious(local_training=False)
        honest_first = torch.tensor([1.0, 3.0])
        honest_second = torch.tensor([3.0, 5.0])
        round_results = RoundResults(
            manager,
            [
                {CLIENT_ID: 0},
                {CLIENT_ID: 1, CLIENT_UPDATE: honest_first},
                {CLIENT_ID: 2, CLIENT_UPDATE: honest_second},
            ],
        )
        spec = AdversaryConfig().update_from_dict(
            {"type": "blades.adversaries.IPMAdversary", "scale": 2.0}
        ).pre_build()
        adversary = from_config(spec)
        self.assertIsInstance(adversary, IPMAdversary)
        adversary.on_local_round_end(round_results)

        torch.testing.assert_close(round_results.local_results[0][CLIENT_UPDATE], torch.tensor([-4.0, -8.0]))
        torch.testing.assert_close(honest_first, torch.tensor([1.0, 3.0]))
        torch.testing.assert_close(honest_second, torch.tensor([3.0, 5.0]))
        updates = [result[CLIENT_UPDATE] for result in round_results.local_results]
        torch.testing.assert_close(Mean()(updates), torch.zeros(2))
        torch.testing.assert_close(Median()(updates), torch.tensor([1.0, 3.0]))

        for uid in (1, 2):
            manager.get_client_by_id(uid).to_malicious(local_training=False)
        with self.assertRaisesRegex(ValueError, "No benign updates"):
            adversary.get_benign_updates(round_results)


if __name__ == "__main__":
    unittest.main()
