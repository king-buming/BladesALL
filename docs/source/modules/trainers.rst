Trainers and algorithms
=======================

``blades/trainers`` contains ``Trainer``, ``TrainerConfig``, the server,
client manager, and training callback interfaces. ``FedavgTrainer`` and
``FedProxTrainer`` supply the base federated training implementations.

All FedAvg implementations live in ``blades/trainers/fedavg``. The base
``FedavgTrainer`` owns training and evaluation dispatch; ``Fedavg`` adds attack
configuration and its legacy classification-metric policy, while ``FedavgDP``
adds private clients. These variants share the same training loop.

``blades/algorithms/fedavg`` re-exports the same classes for existing callers.
The unified algorithm registry is in ``blades/algorithms/registry.py`` and is
used by the command-line entry point.

For example:

.. code-block:: python

   from blades.trainers import TrainerConfig, FedavgTrainer
   from blades.trainers.fedavg import FedavgConfig

The executable experiment names are ``FEDAVG``, ``FEDAVG_BASE``, ``FEDAVG_DP``,
and ``FEDPROX``. See :doc:`../get_started/key_concepts` for their mapping.
