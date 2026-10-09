"""Round-level contracts for the norm-preserving SignLoader variant."""

from types import SimpleNamespace

import pytest
import torch

from blades.adversaries.spear_sl_sphere_adversary import SPEARSLSphereAdversary
from blades.constants import CLIENT_ID, CLIENT_UPDATE


def test_round_submits_a_final_vector_that_was_actually_screened():
    class RecordedAdversary(SPEARSLSphereAdversary):
        def _count_dnc_passes(self, benign, candidate, num_malicious, subsets):
            self.screened.append(candidate.clone())
            return super()._count_dnc_passes(benign, candidate, num_malicious, subsets)

    attack = RecordedAdversary(z_values=(0.0, 0.5, 1.0), dnc_num_iters=2,
                               dnc_sub_dim=3, max_relative_l2=2.0)
    attack.screened = []
    benign = torch.tensor([[1., -.8, .1], [.8, -1., -.1], [1.1, -.9, .2]])
    clients = [SimpleNamespace(is_malicious=i >= 3) for i in range(5)]
    results = [{CLIENT_ID: i, CLIENT_UPDATE: benign[i].clone() if i < 3
                else torch.zeros(3)} for i in range(5)]
    trainer = SimpleNamespace(
        local_results=results,
        client_manager=SimpleNamespace(get_client_by_id=lambda i: clients[i]),
    )
    attack.on_local_round_end(trainer)
    actual = results[3][CLIENT_UPDATE]
    assert any(torch.equal(actual, candidate) for candidate in attack.screened)
    assert torch.equal(actual, results[4][CLIENT_UPDATE])
    assert actual.data_ptr() != results[4][CLIENT_UPDATE].data_ptr()
    assert torch.equal(torch.stack([r[CLIENT_UPDATE] for r in results[:3]]), benign)
    counts = tuple(int(mask.sum()) for mask in (actual > 0, actual == 0, actual < 0))
    assert counts == tuple(attack.last_diagnostics['target_counts'])
    z = attack.last_diagnostics['z']
    original = benign.mean(0) + z * benign.std(0)
    torch.testing.assert_close(actual.norm(), original.norm(), rtol=3e-6, atol=1e-7)
    assert attack.last_diagnostics['relative_l2'] <= 2.0 + 1e-6


def test_no_benign_results_needed_when_no_malicious_client_participates():
    attack = SPEARSLSphereAdversary()
    trainer = SimpleNamespace(local_results=[], client_manager=SimpleNamespace())
    attack.on_local_round_end(trainer)


def test_single_benign_update_has_finite_output():
    attack = SPEARSLSphereAdversary(z_values=(0., 1.), dnc_num_iters=1)
    result = attack.generate_update(torch.tensor([[1., -2., 0.]]), 1)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.tensor([1., -2., 0.]))


def test_infeasible_distance_budget_is_not_silently_violated():
    # Mean [0,0], std >0 and one positive/one negative required.
    # z=1 gives two equal positives. A one-negative unit vector must move >1R.
    attack = SPEARSLSphereAdversary(z_values=(1.,), max_relative_l2=0.01)
    with pytest.raises(ValueError, match='budget'):
        attack.generate_update(torch.tensor([[1., -1.], [-1., 1.]]), 1)


def test_limited_information_records_only_observed_population():
    attack = SPEARSLSphereAdversary(z_values=(0.,), visible_benign_limit=1,
                                   dnc_num_iters=1, max_relative_l2=2.)
    torch.manual_seed(71)
    result = attack.generate_update(torch.tensor([[1., -2.], [2., -3.], [3., -1.]]), 1)
    assert torch.isfinite(result).all()
    assert attack.last_diagnostics['observed_benign'] == 1


def test_a_failed_proxy_is_reported_instead_of_claiming_acceptance():
    class RejectingAdversary(SPEARSLSphereAdversary):
        def _count_dnc_passes(self, benign, candidate, num_malicious, subsets):
            return 0

    attack = RejectingAdversary(z_values=(0.,), max_relative_l2=2.)
    result = attack.generate_update(torch.tensor([[1., -2.], [2., -3.]]), 1)
    assert torch.isfinite(result).all()
    assert attack.last_diagnostics['proxy_all_passed'] is False
    assert attack.last_diagnostics['proxy_passed'] == 0


def test_config_can_load_new_class_by_module_path():
    from ray.rllib.utils.from_config import from_config

    attack = from_config({'type': 'blades.adversaries.spear_sl_sphere_adversary.'
                         'SPEARSLSphereAdversary', 'z_values': [0., 1.]})
    assert isinstance(attack, SPEARSLSphereAdversary)
