Aggregators
===========

Robust aggregation implementations live in ``blades/aggregators``. The public
``blades.aggregators`` module exports ``Mean``, ``Median``, ``Trimmedmean``,
``GeoMed``, ``DnC``, ``Centeredclipping``, ``Clippedclustering``, ``Multikrum``,
and ``Signguard``. Other local implementations can be selected by their full
class paths where supported by server configuration.

For example:

.. code-block:: python

   from blades.aggregators import Mean

Server configuration connects the chosen aggregator to the training loop.
When changing a defense, edit its implementation here and check the relevant
module tests plus a small training run.
