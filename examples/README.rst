Examples
==========

Run these scripts from the BladesALL project root after an editable install.

``python -m examples.smoke_train`` runs a small CPU training experiment with
synthetic data and no dataset download. Start with this example when changing
the local training, attack, or aggregation code.

The other scripts and notebooks retain the original research examples. They
may require dataset downloads, GPU resources, or updates to older configuration
APIs. Plotting examples require the optional dependency group:

.. code-block:: console

   python -m pip install -e ".[examples]"

YAML experiments live in ``blades/tuned_examples`` and ``blades/new_examples``.
For example:

.. code-block:: console

   python -m blades file blades/tuned_examples/fedsgd_cnn_fashion_mnist.yaml
