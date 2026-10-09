"""Independent feasibility and optimality contracts for the sphere projection."""

import math
from itertools import product

import pytest
import torch

from blades.adversaries.spear_sl_sphere_adversary import project_sign_sphere


def test_an_already_feasible_vector_is_unchanged_without_direction_bias():
    g = torch.tensor([3.0, -4.0, 0.0], dtype=torch.float64)
    actual = project_sign_sphere(g, (1, 1, 1))
    torch.testing.assert_close(actual, g, rtol=1e-12, atol=1e-12)


def _assert_feasible(actual, g, counts, epsilon_rel=1e-6):
    """Check the public constraints without relying on projection internals."""
    assert actual.shape == g.shape
    assert actual.dtype == g.dtype
    assert actual.device == g.device
    assert bool(torch.isfinite(actual).all())
    observed = tuple(int(mask.sum()) for mask in
                     (actual > 0, actual == 0, actual < 0))
    assert observed == counts
    radius = math.hypot(*g.tolist())
    actual_radius = math.hypot(*actual.tolist())
    tolerance = 3e-6 if g.dtype == torch.float32 else 3e-12
    assert math.isclose(actual_radius, radius, rel_tol=tolerance, abs_tol=0.0)
    nonzero = actual[actual != 0].abs()
    if nonzero.numel():
        floor = epsilon_rel * radius / math.sqrt(counts[0] + counts[2])
        assert float(nonzero.min()) >= floor * (1.0 - tolerance)


def test_one_active_coordinate_uses_the_best_location_and_entire_radius():
    g = torch.tensor([3.0, 4.0], dtype=torch.float64)
    actual = project_sign_sphere(g, (1, 1, 0))
    torch.testing.assert_close(actual, torch.tensor([0., 5.], dtype=g.dtype),
                               rtol=1e-12, atol=1e-12)


def test_required_adverse_sign_stays_at_the_nonzero_floor():
    g = torch.tensor([2.0, 1.0, 3.0], dtype=torch.float64)
    epsilon_rel = 0.2
    floor = epsilon_rel * math.sqrt(14.0 / 2.0)
    expected = torch.tensor([0., -floor, math.sqrt(14.0 - floor**2)],
                            dtype=g.dtype)
    actual = project_sign_sphere(g, (1, 1, 1), epsilon_rel=epsilon_rel)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_all_adverse_scores_place_excess_norm_on_the_least_adverse_coordinate():
    # Every allowed coordinate points against g. Spreading the spare norm
    # proportionally would maximize distance instead of minimizing it.
    g = torch.tensor([1.0, 2.0], dtype=torch.float64)
    epsilon_rel = 0.2
    floor = epsilon_rel * math.sqrt(5.0 / 2.0)
    expected = torch.tensor([-math.sqrt(5.0 - floor**2), -floor], dtype=g.dtype)
    actual = project_sign_sphere(g, (0, 0, 2), epsilon_rel=epsilon_rel)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("epsilon_rel", [1e-6, 0.25, 0.999])
def test_every_small_sign_composition_preserves_radius_and_exact_counts(dtype, epsilon_rel):
    # Covers tied entries, zero entries, and every nonempty support in d=1..6.
    seed_values = [1.0, -2.0, 0.0, 0.5, 1.0, -0.2]
    for dimension in range(1, 7):
        g = torch.tensor(seed_values[:dimension], dtype=dtype)
        for positive in range(dimension + 1):
            for negative in range(dimension - positive + 1):
                if positive + negative == 0:
                    continue
                counts = (positive, dimension - positive - negative, negative)
                actual = project_sign_sphere(g, counts, epsilon_rel=epsilon_rel)
                _assert_feasible(actual, g, counts, epsilon_rel)


def _circle_candidates(counts, radius, epsilon_rel):
    """Enumerate feasible arcs, including their exact endpoints and both axes."""
    active = counts[0] + counts[2]
    if active == 1:
        sign = 1.0 if counts[0] else -1.0
        return torch.tensor([[sign * radius, 0.], [0., sign * radius]],
                            dtype=torch.float64)
    low = math.asin(epsilon_rel / math.sqrt(2.0))
    angles = torch.linspace(low, math.pi / 2.0 - low, 16385, dtype=torch.float64)
    magnitudes = radius * torch.stack((angles.cos(), angles.sin()), dim=1)
    candidates = []
    for signs in product((-1.0, 1.0), repeat=2):
        if signs.count(1.0) == counts[0]:
            candidates.append(magnitudes * torch.tensor(signs, dtype=torch.float64))
    return torch.cat(candidates)


@pytest.mark.parametrize("counts", [(2, 0, 0), (1, 0, 1), (0, 0, 2),
                                     (1, 1, 0), (0, 1, 1)])
@pytest.mark.parametrize("g_values,direction_values,weight", [
    ([1., 2.], [-2., 1.], 0.0),
    ([-3., 0.4], [2., -1.], 0.9),
    ([1., 2.], [-1., -2.], 3.0),
    ([1., -0.01], [0.1, 3.], 2.0),
])
def test_objective_matches_dense_independent_circle_search(
        counts, g_values, direction_values, weight):
    g = torch.tensor(g_values, dtype=torch.float64)
    direction = torch.tensor(direction_values, dtype=torch.float64)
    epsilon_rel = 0.17
    actual = project_sign_sphere(g, counts, direction=direction,
                                 direction_weight=weight, epsilon_rel=epsilon_rel)
    _assert_feasible(actual, g, counts, epsilon_rel)
    radius = math.hypot(*g_values)
    q = g / radius + weight * direction / math.hypot(*direction_values)
    reference = _circle_candidates(counts, radius, epsilon_rel)
    reference_best = float((reference @ q).max())
    actual_objective = float(actual @ q)
    scale = radius * math.hypot(*q.tolist())
    # The finite arc grid may slightly underestimate an interior maximum.
    assert actual_objective >= reference_best - 1e-11 * max(1.0, scale)
    assert actual_objective <= reference_best + 5e-8 * max(1.0, scale)


def test_increasing_direction_weight_never_reduces_directional_alignment():
    g = torch.tensor([2., -1., 0.5, -3., 1.2], dtype=torch.float64)
    direction = torch.tensor([-1., 2., -4., 0.2, 0.1], dtype=torch.float64)
    alignments, distances = [], []
    for weight in [0., 0.01, 0.1, 0.3, 0.7, 1., 2., 5., 20.]:
        actual = project_sign_sphere(g, (2, 1, 2), direction=direction,
                                     direction_weight=weight, epsilon_rel=0.1)
        _assert_feasible(actual, g, (2, 1, 2), 0.1)
        alignments.append(float(actual @ direction))
        distances.append(float(torch.linalg.vector_norm(actual - g)))
    assert all(b >= a - 1e-10 for a, b in zip(alignments, alignments[1:]))
    assert all(b >= a - 1e-10 for a, b in zip(distances, distances[1:]))
    assert alignments[-1] > alignments[0] + 1.0
    assert distances[-1] > distances[0] + 0.1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_does_not_mutate_noncontiguous_inputs_or_the_direction(dtype):
    storage = torch.tensor([2., 99., -1., 88., 0.5, 77.], dtype=dtype)
    g = storage[::2]
    direction = torch.tensor([-1., 3., 2.], dtype=dtype)
    storage_before, direction_before = storage.clone(), direction.clone()
    actual = project_sign_sphere(g, (1, 1, 1), direction=direction,
                                 direction_weight=0.7)
    _assert_feasible(actual, g, (1, 1, 1))
    assert torch.equal(storage, storage_before)
    assert torch.equal(direction, direction_before)


def test_zero_direction_and_zero_weight_have_no_effect():
    g = torch.tensor([1., -4., 2.], dtype=torch.float64)
    expected = project_sign_sphere(g, (1, 1, 1))
    with_zero_direction = project_sign_sphere(
        g, (1, 1, 1), direction=torch.zeros_like(g), direction_weight=9.)
    with_zero_weight = project_sign_sphere(
        g, (1, 1, 1), direction=torch.tensor([-10., 3., 7.], dtype=g.dtype),
        direction_weight=0.)
    torch.testing.assert_close(with_zero_direction, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(with_zero_weight, expected, rtol=1e-12, atol=1e-12)


def test_direction_normalization_makes_positive_scaling_irrelevant():
    g = torch.tensor([1., -4., 2.], dtype=torch.float64)
    direction = torch.tensor([-2., 1., 3.], dtype=g.dtype)
    actual = project_sign_sphere(g, (1, 1, 1), direction=direction,
                                 direction_weight=0.7)
    scaled = project_sign_sphere(g, (1, 1, 1), direction=19.0 * direction,
                                 direction_weight=0.7)
    torch.testing.assert_close(scaled, actual, rtol=1e-12, atol=1e-12)


def test_finite_float32_direction_with_larger_than_float32_norm_is_normalized():
    g = torch.tensor([1., -2.])
    expected = project_sign_sphere(g, (2, 0, 0), direction=torch.ones(2),
                                   direction_weight=3.)
    actual = project_sign_sphere(g, (2, 0, 0),
                                 direction=torch.tensor([3e38, 3e38]), direction_weight=3.)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=1e-7)


def test_finite_float32_vector_can_have_a_norm_larger_than_float32_maximum():
    g = torch.tensor([3e38, -3e38])
    actual = project_sign_sphere(g, (1, 0, 1))
    _assert_feasible(actual, g, (1, 0, 1))
    torch.testing.assert_close(actual, g, rtol=3e-6, atol=0.)


@pytest.mark.parametrize("dtype,scale", [
    (torch.float32, 1e-25), (torch.float32, 1e25),
    (torch.float64, 1e-200), (torch.float64, 1e200),
])
def test_large_and_small_finite_magnitudes_remain_feasible(dtype, scale):
    g = torch.tensor([3.0 * scale, -4.0 * scale, 0.2 * scale], dtype=dtype)
    actual = project_sign_sphere(g, (1, 1, 1))
    _assert_feasible(actual, g, (1, 1, 1))


def test_zero_norm_is_feasible_only_with_all_zero_counts():
    g = torch.zeros(3, dtype=torch.float64)
    actual = project_sign_sphere(g, (0, 3, 0))
    assert torch.equal(actual, g)
    assert actual.dtype == g.dtype
    with pytest.raises(ValueError):
        project_sign_sphere(g, (1, 2, 0))


def test_nonzero_norm_cannot_have_all_zero_counts():
    with pytest.raises(ValueError):
        project_sign_sphere(torch.tensor([1., 0., 0.]), (0, 3, 0))


@pytest.mark.parametrize("counts", [
    (1, 1), (1, 0, 0, 1), (-1, 1, 2), (2, 0, 1),
    (1.0, 0, 1), (True, 0, 1), (1, "0", 1),
])
def test_invalid_count_specifications_are_rejected(counts):
    with pytest.raises((ValueError, TypeError)):
        project_sign_sphere(torch.tensor([1., -2.]), counts)


@pytest.mark.parametrize("g", [
    torch.empty(0), torch.tensor(1.), torch.ones(1, 2),
    torch.tensor([float("nan"), 1.]), torch.tensor([float("inf"), 1.]),
    torch.tensor([1, 2]), torch.ones(2, dtype=torch.float16),
    torch.ones(2, dtype=torch.bfloat16), torch.ones(2, dtype=torch.complex64),
    [1., 2.],
])
def test_unsupported_or_nonfinite_input_is_rejected(g):
    with pytest.raises((ValueError, TypeError)):
        project_sign_sphere(g, (1, 0, 1))


@pytest.mark.parametrize("epsilon_rel", [0., -1., 1., 1.1, float("nan"), float("inf")])
def test_invalid_epsilon_is_rejected(epsilon_rel):
    with pytest.raises((ValueError, TypeError)):
        project_sign_sphere(torch.tensor([1., -2.]), (1, 0, 1),
                            epsilon_rel=epsilon_rel)


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf")])
def test_invalid_direction_weight_is_rejected(weight):
    with pytest.raises((ValueError, TypeError)):
        project_sign_sphere(torch.tensor([1., -2.]), (1, 0, 1),
                            direction=torch.tensor([2., 1.]), direction_weight=weight)


@pytest.mark.parametrize("direction", [
    torch.ones(3), torch.ones(1, 2), torch.tensor([float("nan"), 1.]),
    torch.tensor([1., float("inf")]),
])
def test_invalid_direction_is_rejected(direction):
    with pytest.raises((ValueError, TypeError)):
        project_sign_sphere(torch.tensor([1., -2.]), (1, 0, 1),
                            direction=direction, direction_weight=0.5)
