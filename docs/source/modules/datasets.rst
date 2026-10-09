Datasets and partitioners
=========================

``blades.datasets`` exposes ``DatasetCatalog``, ``FLDataset``, ``ClientDataset``,
and ``ToyFLDataset``. Dataset implementations and local data-loading code live
in ``blades/datasets``. ``ToyFLDataset`` is useful for checks that should not
download external datasets.

Partitioners live in ``blades/datasets/partitioners`` and are also exported from
``blades.datasets``:

* ``DatasetPartitioner`` defines the partitioning interface.
* ``IIDPartitioner`` distributes data independently across clients.
* ``DirichletPartitioner`` partitions by a Dirichlet distribution.
* ``ShardPartitioner`` distributes sorted data shards.

For example:

.. code-block:: python

   from blades.datasets import DatasetCatalog, ToyFLDataset
   from blades.datasets.partitioners import IIDPartitioner

Inspect each dataset's configuration and download behavior before running a
research experiment.
