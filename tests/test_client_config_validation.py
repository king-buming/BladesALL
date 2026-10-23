"""Legacy client settings must not silently change an unrelated configuration."""

import unittest

import torch  # noqa: F401 - Initialize native libraries before Ray on Windows.
import yaml

from blades.clients import BaseClient, Client, ClientConfig, DPCliengConfig
from blades.trainers.fedavg import FedavgTrainerConfig


class CustomClient(Client):
    """An extension that explicitly consumes its own configuration fields."""

    def __init__(self, client_config):
        super().__init__(client_config)
        self.selected_optimizer = client_config.opt_identifier
        self.extension_option = client_config.extension_option


class ClientConfigValidationTests(unittest.TestCase):
    def test_default_and_training_configuration_still_build(self):
        for class_specifier in ("blades.clients.Client", Client, BaseClient):
            with self.subTest(class_specifier=class_specifier):
                config = ClientConfig(class_specifier).training(
                    lr=0.25, momentum=0.4, num_batch_per_round=3
                )
                client = config.build()
                self.assertEqual(client.config.lr, 0.25)
                self.assertEqual(client.config.momentum, 0.4)
                self.assertEqual(client.config.num_batch_per_round, 3)
                self.assertEqual(config.to_dict(), client.config.to_dict())
        DPCliengConfig().build()

    def test_default_dictionary_round_trip_preserves_training_values(self):
        original = ClientConfig("blades.clients.Client").training(lr=0.3, momentum=0.7)
        rebuilt = ClientConfig("blades.clients.Client").update_from_dict(original.to_dict())
        self.assertEqual(rebuilt.build().config.lr, 0.3)
        self.assertEqual(rebuilt.momentum, 0.7)

    def test_dataset_and_model_setters_give_actionable_errors(self):
        config = ClientConfig("blades.clients.Client")
        with self.assertWarns(DeprecationWarning):
            with self.assertRaisesRegex(ValueError, "TrainerConfig.data"):
                config.dataset("CIFAR10", train_batch_size=128)
        with self.assertWarns(DeprecationWarning):
            with self.assertRaisesRegex(ValueError, "global_model.*Task.loss"):
                config.model("resnet", loss="mse")

    def test_optimizer_setter_maps_supported_sgd_settings(self):
        config = ClientConfig("blades.clients.Client")
        with self.assertWarns(DeprecationWarning):
            config.optimizer("SGD", opt_kargs={"lr": 0.03, "momentum": 0.8})
        self.assertEqual(config.build().config.lr, 0.03)
        self.assertEqual(config.momentum, 0.8)

    def test_optimizer_setter_rejects_ignored_options(self):
        settings = (
            {"opt_identifier": "adam"},
            {"opt_kargs": {"weight_decay": 0.01}},
            {"lr_scheduler": "StepLR"},
            {"lr_scheduler_config": {}},
        )
        for options in settings:
            with self.subTest(options=options):
                with self.assertWarns(DeprecationWarning):
                    with self.assertRaisesRegex(ValueError, "Task.init_optimizer"):
                        ClientConfig("blades.clients.Client").optimizer(**options)

    def test_yaml_dictionary_cannot_bypass_legacy_validation(self):
        fragments = (
            "opt_identifier: adam",
            "opt_kargs: {weight_decay: 0.01}",
            "lr_scheduler: StepLR",
            "lr_scheduler_config: {step_size: 2}",
            "dataset_identifier: CIFAR10",
            "train_batch_size: 128",
            "model_identifier: resnet",
            "model_config: {hidden_dim: 32}",
            "loss_identifier: mse",
        )
        for fragment in fragments:
            with self.subTest(fragment=fragment):
                settings = yaml.safe_load(fragment)
                with self.assertRaisesRegex(ValueError, "TrainerConfig|Task"):
                    ClientConfig("blades.clients.Client").update_from_dict(settings)
                with self.assertRaisesRegex(ValueError, "TrainerConfig|Task"):
                    FedavgTrainerConfig().client(client_config=settings).get_client_config()

    def test_dictionary_sgd_options_apply_with_explicit_training_precedence(self):
        config = ClientConfig("blades.clients.Client").update_from_dict(
            {"opt_identifier": "sgd", "opt_kargs": {"lr": 0.02, "momentum": 0.5}}
        )
        self.assertEqual(config.build().config.lr, 0.02)
        self.assertEqual(config.momentum, 0.5)
        settings = {"lr": 0.4, "opt_kargs": {"lr": 0.8}}
        config.update_from_dict(settings)
        self.assertEqual(config.lr, 0.4)
        self.assertEqual(settings["opt_kargs"]["lr"], 0.8)

    def test_build_rejects_direct_unsupported_legacy_settings(self):
        config = ClientConfig("blades.clients.Client")
        config.opt_identifier = "adam"
        with self.assertRaisesRegex(ValueError, "Task.init_optimizer"):
            config.build()

    def test_direct_sgd_kwargs_cannot_be_ignored_silently(self):
        config = ClientConfig("blades.clients.Client")
        config.opt_kargs = {"lr": 0.7}
        with self.assertWarnsRegex(FutureWarning, "Assigning opt_kargs directly"):
            client = config.build()
        self.assertEqual(client.config.lr, 0.1)

    def test_direct_learning_rate_assignment_remains_available(self):
        config = ClientConfig("blades.clients.Client")
        config.lr = 0.25
        self.assertEqual(config.build().config.lr, 0.25)
        with self.assertWarns(DeprecationWarning):
            config.optimizer("sgd", opt_kargs={"lr": 0.6})
        config.training(lr=0.4)
        self.assertEqual(config.build().config.lr, 0.4)

    def test_custom_client_and_extra_fields_remain_available(self):
        config = ClientConfig(CustomClient).update_from_dict(
            {
                "opt_identifier": "extension_optimizer",
                "opt_kargs": {"custom_option": 2},
                "model_identifier": "extension_model",
                "dataset_identifier": "extension_dataset",
                "extension_option": 17,
            }
        )
        with self.assertWarns(DeprecationWarning):
            config.optimizer("custom_optimizer", opt_kargs={"custom_option": 3})
        with self.assertWarns(DeprecationWarning):
            config.dataset("custom_dataset", train_batch_size=7)
        with self.assertWarns(DeprecationWarning):
            config.model("custom_model", loss="custom_loss")
        client = config.build()
        self.assertEqual(client.selected_optimizer, "custom_optimizer")
        self.assertEqual(client.extension_option, 17)
        self.assertEqual(client.config.opt_kargs, {"custom_option": 3})
        self.assertEqual(client.config.train_batch_size, 7)
        self.assertEqual(client.config.model_identifier, "custom_model")
        builtin = ClientConfig("blades.clients.Client").update_from_dict(
            {"extension_option": "retained"}
        )
        self.assertEqual(builtin.build().config.extension_option, "retained")


if __name__ == "__main__":
    unittest.main()
