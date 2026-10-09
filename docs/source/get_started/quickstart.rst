Quick start
===========

After an editable install, run this command from the project root:

.. code-block:: console

   python -m examples.smoke_train

This example uses synthetic data for a small CPU training run without dataset
downloads. It is the starting point for checking local source changes.

For command-line experiment configuration:

.. code-block:: console

   python -m blades --help
   python -m blades file --help
   python -m blades file blades/tuned_examples/fedsgd_cnn_fashion_mnist.yaml

The installed ``blades`` and ``bladesall`` commands expose the same interface.
The path after ``file`` is the experiment configuration file. Adjust the YAML's
dataset, resource settings, client count, and stopping conditions for your
machine before launching a research experiment.

Run the merge verification tests or build the distribution with:

.. code-block:: console

   python -m pytest tests
   python -m build

Additional inherited tests live in each module's ``tests`` subdirectory. Some
historical tests depend on external datasets or older configuration APIs.
