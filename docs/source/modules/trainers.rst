Trainers and algorithms
=======================

``blades/trainers`` contains ``Trainer``, ``TrainerConfig``, the server,
client manager, and training callback interfaces. ``FedavgTrainer`` and
``FedProxTrainer`` supply the base federated training implementations.

``blades/algorithms/fedavg`` adds attack-aware ``Fedavg`` / ``FedavgConfig`` and
``FedavgDP`` / ``FedavgDPConfig``. The unified algorithm registry is in
``blades/algorithms/registry.py`` and is used by the command-line entry point.

For example:

.. code-block:: python

   from blades.trainers import TrainerConfig, FedavgTrainer
   from blades.algorithms.fedavg import FedavgConfig

The executable experiment names are ``FEDAVG``, ``FEDAVG_BASE``, ``FEDAVG_DP``,
and ``FEDPROX``. See :doc:`../get_started/key_concepts` for their mapping.
