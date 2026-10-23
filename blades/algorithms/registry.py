"""Registry of algorithm names for `rllib train --run=<alg_name>`"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass


def _import_fedavg():
    from blades.trainers.fedavg import Fedavg

    return Fedavg, Fedavg.get_default_config()


def _import_fedavgdp():
    from blades.trainers.fedavg import FedavgDP

    return FedavgDP, FedavgDP.get_default_config()


def _import_fedavg_base():
    from blades.trainers.fedavg import FedavgTrainer

    return FedavgTrainer, FedavgTrainer.get_default_config()


def _import_fedprox():
    from blades.trainers.fedprox import FedProxTrainer, FedProxTrainerConfig

    return FedProxTrainer, FedProxTrainerConfig()


ALGORITHMS = {
    "FEDAVG": _import_fedavg,
    "FEDAVG_DP": _import_fedavgdp,
    "FEDAVG_BASE": _import_fedavg_base,
    "FEDPROX": _import_fedprox,
}


def _get_algorithm_class(alg: str) -> type:
    # This helps us get around a circular import (tune calls rllib._register_all when
    # checking if a rllib Trainable is registered)
    if alg in ALGORITHMS:
        return ALGORITHMS[alg]()[0]
    elif alg == "script":
        from ray.tune import script_runner

        return script_runner.ScriptRunner
    elif alg == "__fake":
        from ray.rllib.algorithms.mock import _MockTrainer

        return _MockTrainer
    elif alg == "__sigmoid_fake_data":
        from ray.rllib.algorithms.mock import _SigmoidFakeData

        return _SigmoidFakeData
    elif alg == "__parameter_tuning":
        from ray.rllib.algorithms.mock import _ParameterTuningTrainer

        return _ParameterTuningTrainer
    else:
        raise Exception("Unknown algorithm {}.".format(alg))
