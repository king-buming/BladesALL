"""Callbacks shared by all FedAvg variants."""

from typing import TYPE_CHECKING

from blades.trainers.callbacks import TrainerCallback, TrainerCallbackList

if TYPE_CHECKING:
    from .fedavg import FedavgTrainer


class FedavgCallback(TrainerCallback):
    def on_local_round_end(self, trainer: "FedavgTrainer"):
        """Called after client training, before updates are aggregated."""


class FedavgTrainerCallbackList(TrainerCallbackList):
    def on_local_round_end(self, trainer: "FedavgTrainer"):
        for callback in self._callbacks:
            callback.on_local_round_end(trainer)


__all__ = ["FedavgCallback", "FedavgTrainerCallbackList"]
