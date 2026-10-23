"""Attack-aware configuration and evaluation policy for the FedAvg engine."""

from typing import TYPE_CHECKING, Dict, Optional

from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override

from blades.utils.types import NotProvided

from .fedavg import FedavgTrainer, FedavgTrainerConfig

if TYPE_CHECKING:
    from blades.adversaries import AdversaryConfig


class FedavgConfig(FedavgTrainerConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or Fedavg)
        self.adversary_config = {}
        self.num_malicious_clients = 0

    def adversary(
        self,
        *,
        num_malicious_clients: Optional[int] = NotProvided,
        adversary_config: Optional[Dict] = NotProvided,
    ):
        if adversary_config is not NotProvided:
            deep_update(self.adversary_config, adversary_config, True)
        if num_malicious_clients is not NotProvided:
            self.num_malicious_clients = num_malicious_clients
        return self

    def get_adversary_config(self) -> "AdversaryConfig":
        if not self._is_frozen:
            raise ValueError(
                "Cannot call `get_adversary_config()` on an unfrozen "
                "FedavgTrainerConfig! Please call `freeze()` first."
            )
        # Adversaries depend on trainers; load them only after trainer imports finish.
        from blades.adversaries import Adversary, AdversaryConfig

        return (
            AdversaryConfig(
                adversary_cls=self.adversary_config.get("type", Adversary)
            )
            .update_from_dict(self.adversary_config)
            .pre_build()
        )

    def _get_callback_specs(self):
        return super()._get_callback_specs() + [self.get_adversary_config()]

    @override(FedavgTrainerConfig)
    def validate(self) -> None:
        super().validate()

        if (
            self.server_config.get("aggregator", None) is not None
            and self.server_config.get("aggregator").get("type") is not None
            and (
                "DnC" == self.server_config["aggregator"]["type"]
                or "Trimmedmean" in self.server_config["aggregator"]["type"]
                or "Multikrum" in self.server_config["aggregator"]["type"]
            )
        ):
            self.server_config["aggregator"]["num_byzantine"] = self.num_malicious_clients

        if self.num_malicious_clients > self.num_clients:
            raise ValueError(
                "`num_malicious_clients` must be smaller than or equal "
                "`num_clients`! Simulation makes no sense otherwise."
            )


class Fedavg(FedavgTrainer):
    """FedAvg with adversary callbacks and legacy classification metrics."""

    @classmethod
    def get_default_config(cls) -> FedavgConfig:
        return FedavgConfig(algo_class=cls)

    def _compile_eval_results(self, results):
        from blades.tasks.classifier import compile_classification_results

        return compile_classification_results(results)
