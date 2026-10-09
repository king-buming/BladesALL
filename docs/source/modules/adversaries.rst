Adversaries
===========

Attack implementations are local files in ``blades/adversaries``. The base
``Adversary`` class and ``AdversaryConfig`` define the attack callback interface.
The package exports the existing attacks, including ``SPEARAdversary``,
``SPEARSLAdversary``, ``ALIEAdversary``, ``IPMAdversary``, ``MinMaxAdversary``,
``NoiseAdversary``, ``LabelFlipAdversary``, and ``SignFlipAdversary``.

For example:

.. code-block:: python

   from blades.adversaries import SPEARAdversary
   from blades.algorithms.fedavg import FedavgConfig

Use the attack-aware ``FEDAVG`` algorithm and set the malicious client count
and adversary configuration for attack experiments. Dotted class paths in
configuration use ``blades.adversaries``, matching the local source package.
