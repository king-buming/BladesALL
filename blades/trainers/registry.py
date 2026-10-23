"""Compatibility aliases for :mod:`blades.algorithms.registry`.

``FEDAVG`` now has the unified, attack-aware algorithm semantics. Use
``FEDAVG_BASE`` for the original plain ``FedavgTrainer`` implementation.
"""

from blades.algorithms.registry import (
    ALGORITHMS,
    _get_algorithm_class,
    _import_fedavg,
    _import_fedprox,
)

# Keep the historical table and loader names without a second registry.
TRAINERS = ALGORITHMS

__all__ = [
    "TRAINERS", "ALGORITHMS", "_get_algorithm_class", "_import_fedavg", "_import_fedprox"
]
