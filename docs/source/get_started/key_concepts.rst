Key concepts
============

An experiment selects a trainer or algorithm, a model, a dataset, client
training settings, and a server aggregation rule. The trainer schedules local
client work, collects model updates, invokes configured attack callbacks, and
passes the updates to the server for aggregation.

``blades.clients.base_client.BaseClient`` provides ordinary local training.
``blades.clients.Client`` extends it with malicious-client behavior.
``ClientConfig``, ``ClientCallback``, and ``ClientCallbackList`` remain available
from ``blades.clients`` so client configuration and callbacks share the same
local implementation.

The unified registry supports these YAML ``run`` values:

.. list-table::
   :header-rows: 1

   * - Value
     - Implementation
   * - ``FEDAVG``
     - Attack-aware ``blades.algorithms.fedavg.Fedavg``
   * - ``FEDAVG_BASE``
     - Basic ``blades.trainers.FedavgTrainer``
   * - ``FEDAVG_DP``
     - ``blades.algorithms.fedavg.FedavgDP``
   * - ``FEDPROX``
     - ``blades.trainers.FedProxTrainer``

Configuration may resolve classes by dotted import paths. Use the local
namespace, for example ``blades.adversaries.SPEARAdversary`` or
``blades.tasks.Classifier``. Aggregators and datasets also support their catalog
names, such as ``Mean`` and ``FashionMNIST``.
