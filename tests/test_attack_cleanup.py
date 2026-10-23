"""Behavioral regressions for removal of duplicated attack code and dead options."""

import warnings
from types import SimpleNamespace

import pytest
import torch

from blades.adversaries import SPEARAdversary, SPEARSLAdversary, SPEARSL2Adversary
from blades.aggregators import Signguard


@pytest.mark.parametrize("attack_class", [SPEARAdversary, SPEARSLAdversary])
@pytest.mark.parametrize("seed, expected_z", [(7, 1.1474609375), (2026, 1.1865234375)])
def test_shared_search_preserves_seeded_legacy_result_and_rng(attack_class, seed, expected_z):
    # Values were recorded from both original implementations before extraction.
    benign = torch.randn(
        5, 6, generator=torch.Generator().manual_seed(203), dtype=torch.float64
    ) + 0.3
    attack = attack_class()
    attack.num_clients = 7
    attack.num_byzantine = 2
    attack.dnc_sub_dim = 4
    attack.dnc_num_iters = 2
    attack.dnc_filter_frac = 1.0
    attack.z_max_standard = torch.tensor(0.75)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        assert attack.find_adaptive_z_max(benign) == expected_z
        actual_rng = torch.get_rng_state()

        torch.manual_seed(seed)
        # Ten search candidates, each with two independently sampled subspaces.
        for _ in range(20):
            torch.randperm(benign.shape[1])
        assert torch.equal(actual_rng, torch.get_rng_state())


@pytest.mark.parametrize("attack_class", [SPEARAdversary, SPEARSLAdversary])
def test_shared_search_keeps_subclass_simulator_dispatch(attack_class):
    class AcceptingAttack(attack_class):
        def simulate_dnc_filtering(self, all_updates, benign_mask):
            self.screened.append(all_updates.clone())
            assert benign_mask is None
            return torch.arange(all_updates.shape[0])

    attack = AcceptingAttack()
    attack.num_clients = 4
    attack.num_byzantine = 1
    attack.z_max_standard = torch.tensor(0.75)
    attack.screened = []
    benign = torch.tensor([[1.0, -1.0], [0.5, -2.0], [2.0, -0.5]])

    z = attack.find_adaptive_z_max(benign)

    assert z == 5.0 * (1 - 2 ** -10)
    assert len(attack.screened) == 10
    torch.testing.assert_close(attack.screened[-1][:-1], benign)
    torch.testing.assert_close(attack.screened[-1][-1], benign.mean(0) + benign.std(0) * z)


def test_sl2_initializes_clients_and_builds_a_finite_candidate():
    class Client:
        def to_malicious(self, *, local_training):
            self.is_malicious = True
            self.local_training = local_training

    attack = SPEARSL2Adversary()
    attack.clients = [Client()]
    attack.on_trainer_init(SimpleNamespace(config=SimpleNamespace(num_clients=4)))

    assert attack.num_clients == 4
    assert attack.num_byzantine == 1
    assert attack.clients[0].is_malicious
    assert attack.clients[0].local_training is False
    benign = torch.tensor([[1.0, -1.0], [0.5, -2.0], [2.0, -0.5]])
    candidate = attack._build_candidate(0.5, benign, attack._benign_sign_stats(benign))
    assert torch.isfinite(candidate).all()
    assert candidate.norm() <= benign.norm(dim=1).median() + 1e-6


@pytest.mark.parametrize(
    "args, kwargs, deprecated_names",
    [
        ((), {"max_tau": 1e5}, ["max_tau"]),
        ((), {"linkage": "average"}, ["linkage"]),
        (("mean", 0.01, "single"), {}, ["max_tau", "linkage"]),
    ],
)
def test_signguard_explicit_legacy_options_warn_without_changing_aggregation(
    args, kwargs, deprecated_names
):
    # Three consistent sign vectors and one outlier give a nonempty selection.
    updates = [
        torch.tensor([1.0, -1.0, 0.0]),
        torch.tensor([2.0, -1.0, 0.0]),
        torch.tensor([1.0, -2.0, 0.0]),
        torch.tensor([-1.0, 1.0, 1.0]),
    ]
    expected = Signguard()(updates)

    with pytest.warns(DeprecationWarning, match="have no effect.*will be removed") as caught:
        legacy = Signguard(*args, **kwargs)

    assert len(caught) == 1
    for name in deprecated_names:
        assert name in str(caught[0].message)
    torch.testing.assert_close(legacy(updates), expected, rtol=0, atol=0)


def test_signguard_default_construction_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Signguard()
    assert not caught


def test_signguard_retains_invalid_linkage_validation():
    with pytest.raises(AssertionError):
        Signguard(linkage="ward")
