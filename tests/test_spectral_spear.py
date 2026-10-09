"""Mathematical and randomized-screening contracts for the spectral extension."""

import importlib

import numpy as np
import pytest
import torch


def core():
    name = 'blades.adversaries.spectral_spear'
    assert importlib.util.find_spec(name) is not None, 'spectral SPEAR core is missing'
    return importlib.import_module(name)


def test_ellipsoid_direction_matches_dense_optimum_and_avoids_dominant_axis():
    m = core()
    benign = torch.tensor([[11., 1., 1.], [-9., 1., 1.],
                           [1., 1.2, 1.], [1., .8, 1.]], dtype=torch.float64)
    metric = m.fit_spectral_metric(benign, (torch.arange(3),), rank=1)
    mean = benign.mean(0)
    direction = m.spectral_direction(mean, metric, penalty=10., radius=2.)
    dense = torch.eye(3, dtype=benign.dtype)
    dense += 10. * metric.factors @ metric.factors.T
    expected = torch.linalg.solve(dense, -mean)
    expected *= 2. / torch.sqrt(expected @ dense @ expected)
    torch.testing.assert_close(direction, expected)
    torch.testing.assert_close(direction @ dense @ direction, mean.new_tensor(4.))
    assert abs(direction[0]) < abs(direction[1]) / 5
    assert mean @ direction < 0
    # The closed-form value bounds every sampled feasible competitor.
    points = torch.randn(100, 3, generator=torch.Generator().manual_seed(7), dtype=mean.dtype)
    points *= (2. / torch.sqrt(torch.einsum('bi,ij,bj->b', points, dense, points)))[:, None]
    assert torch.all(points @ -mean <= direction @ -mean + 1e-10)


def test_sparse_metric_keeps_unsampled_coordinates_and_zero_penalty_ablation():
    m = core()
    benign = torch.tensor([[4., 1., 2., 1.], [-2., 1., 2., 1.]], dtype=torch.float64)
    metric = m.fit_spectral_metric(benign, (torch.tensor([0, 2]),), rank=1)
    assert metric.coordinates.tolist() == [0, 2]
    assert metric.factors.shape == (2, 1)
    mu = benign.mean(0)
    direction = m.spectral_direction(mu, metric, penalty=0., radius=3.)
    torch.testing.assert_close(direction, -3. * mu / mu.norm())


def test_large_penalty_respects_radius_in_uploaded_float32():
    m = core()
    b = torch.tensor([[2.], [0.]])
    metric = m.fit_spectral_metric(b, (torch.arange(1),), rank=1)
    direction = m.spectral_direction(b.mean(0), metric, penalty=1e7, radius=1.)
    energy = direction.double().square().sum() * (1. + 1e7)
    assert energy <= 1. + 2e-6
    torch.testing.assert_close(energy, torch.tensor(1., dtype=torch.float64), rtol=3e-6, atol=1e-9)


def test_constant_sampled_coordinates_do_not_create_roundoff_geometry():
    m = core()
    b = torch.full((7, 5), .1)
    b[:, 4] = torch.arange(7)
    metric = m.fit_spectral_metric(b, (torch.arange(4),), rank=2)
    assert metric.factors.shape == (4, 0)


def test_overlapping_unsorted_subspaces_match_dense_nonzero_penalty():
    m = core()
    b = torch.randn(8, 7, generator=torch.Generator().manual_seed(19), dtype=torch.float64)
    metric = m.fit_spectral_metric(b, (torch.tensor([5, 0, 2]), torch.tensor([2, 3, 0])), rank=2)
    lifted = torch.zeros(7, metric.factors.shape[1], dtype=b.dtype)
    lifted[metric.coordinates] = metric.factors
    h = torch.eye(7, dtype=b.dtype) + 7. * lifted @ lifted.T
    mu = b.mean(0)
    expected = torch.linalg.solve(h, -mu)
    expected *= 2. / torch.sqrt(expected @ h @ expected)
    actual = m.spectral_direction(mu, metric, penalty=7., radius=2.)
    torch.testing.assert_close(actual, expected)


def test_dnc_nonmonotone_witness_is_not_treated_as_binary_boundary():
    m = core()
    b = torch.tensor([[0., 2.], [2., -3.], [1., 2.], [-3., 0.]], dtype=torch.float64)
    flags = []
    for z in (.7, .8, 1., 1.5):
        u = b.mean(0) + z * b.std(0)
        mixed = torch.cat((b, u.repeat(2, 1)))
        mask = m.dnc_select(mixed, (torch.arange(2),), drop_count=2)
        flags.append(bool(mask[-2:].all()))
    assert flags == [True, False, True, False]


def test_proxy_matches_existing_dnc_for_fixed_full_coordinate_bank(monkeypatch):
    m = core()
    from blades.aggregators.aggregators import DnC
    updates = torch.randn(9, 4, generator=torch.Generator().manual_seed(13))
    monkeypatch.setattr(torch, 'randperm', lambda d: torch.arange(d))
    selected = m.dnc_select(updates, (torch.arange(4),), drop_count=2)
    expected = DnC(2, sub_dim=4, num_iters=1)(list(updates))
    torch.testing.assert_close(updates[selected].mean(0), expected)


def test_proxy_parity_includes_intersection_ties_and_malicious_row_order(monkeypatch):
    m = core()
    from blades.aggregators.aggregators import DnC
    b = torch.tensor([[0., 2., .2], [2., -3., -.2], [1., 2., .3], [-3., 0., .1]])
    candidate = b.mean(0) + .7 * b.std(0)
    mixed = torch.stack((candidate, b[0], b[1], candidate, b[2], b[3]))
    spaces = (torch.tensor([0, 1]), torch.tensor([2, 0]))
    iterator = iter(spaces)
    monkeypatch.setattr(torch, 'randperm', lambda d: next(iterator))
    selected = m.dnc_select(mixed, spaces, drop_count=2)
    expected = DnC(2, sub_dim=2, num_iters=2)(list(mixed))
    actual = mixed[selected].mean(0) if selected.any() else torch.zeros(3)
    torch.testing.assert_close(actual, expected)


def test_search_evaluates_whole_grid_and_holdout_cannot_change_selection():
    m = core()
    b = torch.tensor([[1., 4., 1., -.1], [1., -4., 1., .1],
                      [1.1, 2., .9, .1], [.9, -2., 1.1, -.1]], dtype=torch.float64)
    kwargs = dict(z_values=(0., .1, .3), spectral_penalties=(0., 10.),
                  dnc_sub_dim=3, dnc_num_iters=2, search_banks=2, fit_subspaces=2)
    u, report = m.search_spectral_update(b, 1, config=m.SpectralSpearConfig(**kwargs, audit_banks=1))
    v, other = m.search_spectral_update(b, 1, config=m.SpectralSpearConfig(**kwargs, audit_banks=7))
    assert len(report['evaluations']) == 5  # one mean + two nonzero z per penalty
    assert len(report['audit']['retained_malicious']) == 1
    assert len(other['audit']['retained_malicious']) == 7
    torch.testing.assert_close(u, v, rtol=0, atol=0)
    eligible = [e for e in report['evaluations'] if e['feasible']]
    if eligible:
        assert report['worst_damage'] == max(e['worst_damage'] for e in eligible)
        assert report['status'] == 'proxy_feasible'
    assert torch.isfinite(u).all()


def test_random_streams_are_local_and_reproducible():
    m = core()
    b = torch.randn(8, 9, generator=torch.Generator().manual_seed(123))
    cfg = m.SpectralSpearConfig(z_values=(0., .1), spectral_penalties=(1.,),
                               dnc_sub_dim=4, dnc_num_iters=1, search_banks=1, audit_banks=1)
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    a, report = m.search_spectral_update(b, 2, config=cfg, round_index=2)
    c, other = m.search_spectral_update(b, 2, config=cfg, round_index=2)
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert numpy_state[0] == np.random.get_state()[0]
    assert numpy_state[2:] == np.random.get_state()[2:]
    assert torch.equal(a, c)
    assert report == other


@pytest.mark.parametrize('values', [[[1., -2.]], [[0., 0.], [0., 0.]]])
def test_degenerate_data_returns_finite_honestly_reported_fallback(values):
    m = core()
    b = torch.tensor(values)
    u, report = m.search_spectral_update(b, 1, config=m.SpectralSpearConfig(
        z_values=(0.,), dnc_num_iters=1, search_banks=1, audit_banks=1))
    torch.testing.assert_close(u, b.mean(0))
    assert report['status'] in ('proxy_feasible', 'no_feasible_candidate')
    if report['status'] == 'no_feasible_candidate':
        assert report['selected'] is None


@pytest.mark.parametrize('options', [dict(z_values=()), dict(z_values=(-1.,)),
    dict(spectral_penalties=(float('nan'),)), dict(dnc_num_iters=0),
    dict(spectral_rank=True), dict(min_survival=0.), dict(seed=-1), dict(audit_banks=0)])
def test_invalid_search_config_is_rejected(options):
    m = core()
    with pytest.raises(ValueError):
        m.SpectralSpearConfig(**options)


def test_invalid_inputs_fail_explicitly():
    m = core()
    cfg = m.SpectralSpearConfig()
    for b in (torch.empty(0, 2), torch.tensor([[float('nan')]]), torch.ones(2, 3, dtype=torch.int64)):
        with pytest.raises((ValueError, TypeError)):
            m.search_spectral_update(b, 1, config=cfg)
    with pytest.raises(ValueError, match='positions'):
        m.search_spectral_update(torch.ones(2, 3), 2, config=cfg, malicious_positions=[0, 0])
