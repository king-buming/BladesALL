Overview
========

BladesALL brings the original Blades attack layer and local federated learning
implementation into a single package. Keep changes in the corresponding local
module:

* ``blades/adversaries``: attack implementations, including SPEAR and SPEAR-SL.
* ``blades/aggregators``: robust aggregation and defenses.
* ``blades/datasets``: datasets, client data, and data partitioning.
* ``blades/models``: model definitions and backbones.
* ``blades/trainers``: training loops, server configuration, and callbacks.
* ``blades/algorithms``: attack-aware FedAvg and differential privacy variants.
* ``blades/clients``, ``blades/tasks``, and ``blades/core``: execution support.

The original source projects are `Blades
<https://github.com/lishenghui/blades>`_ and `fedlib
<https://github.com/fedlib/fedlib>`_. Their attribution and Apache-2.0 licenses
are retained. Current usage and imports follow this project's ``blades``
namespace.

Research configurations are retained in ``blades/tuned_examples`` and
``blades/new_examples``. Inspect dataset, client count, GPU resources, and round
limits before running a full experiment. Historical notebooks and examples may
use older configuration APIs; start with ``examples/smoke_train.py`` for a small
local training run.
