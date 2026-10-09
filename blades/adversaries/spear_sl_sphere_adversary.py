"""SPEAR with sign-histogram projection onto the original L2 sphere.

This is an experimental alternative to ``spear_sl_adversary.py``. It preserves
the pre-alignment vector's norm, not its distance to every benign vector, and
measures SPEAR-direction retention separately. See docs/signloader_sphere.md.
Load by the full module path; the original adversary is left intact.
"""

import logging
import math
from numbers import Integral

import torch

from blades.constants import CLIENT_ID, CLIENT_UPDATE
from .adversary import Adversary


logger = logging.getLogger(__name__)


def _norm(vector):
    # Rescale before squaring: even float64 accumulation alone can overflow
    # for finite 1e200 entries or underflow for finite 1e-200 entries.
    scale = vector.abs().max().item()
    if scale == 0:
        return 0.
    return torch.linalg.vector_norm(vector.to(torch.float64) / scale).item() * scale


def _validate_vector(vector, name):
    if not isinstance(vector, torch.Tensor) or vector.ndim != 1 or not vector.numel():
        raise ValueError(f'{name} must be a nonempty one-dimensional tensor')
    if vector.dtype not in (torch.float32, torch.float64):
        raise TypeError(f'{name} must have dtype float32 or float64')
    if not torch.isfinite(vector).all():
        raise ValueError(f'{name} must contain only finite values')


@torch.no_grad()
def project_sign_sphere(g, target_counts, *, direction=None,
                        direction_weight=0.0, epsilon_rel=1e-6):
    r"""Maximize q^T x with prescribed sign counts and ||x||_2 = ||g||_2.

    Counts are (positive, zero, negative). Define
    q = g/||g|| + direction_weight * direction/||direction||,
    omitting a zero direction. Each nonzero |x_j| is at least
    epsilon_rel * ||g|| / sqrt(number_of_nonzeros).

    At weight zero this is the globally closest feasible vector to g.
    At positive weights this trades distance for a *linear projection* onto
    direction; it does not guarantee training damage or cosine preservation.
    Sorting q jointly selects signs and coordinates. Given those signs, the
    optimal magnitudes are a clipped positive part rescaled to the sphere.
    """
    _validate_vector(g, 'g')
    if (len(target_counts) != 3
            or any(not isinstance(c, Integral) or isinstance(c, bool) or c < 0
                   for c in target_counts)
            or sum(target_counts) != g.numel()):
        raise ValueError('target_counts must be three nonnegative integers summing to dimension')
    if not math.isfinite(direction_weight) or direction_weight < 0:
        raise ValueError('direction_weight must be finite and nonnegative')
    if not math.isfinite(epsilon_rel) or not 0 < epsilon_rel < 1:
        raise ValueError('epsilon_rel must lie strictly between zero and one')
    if direction is not None:
        _validate_vector(direction, 'direction')
        if direction.shape != g.shape or direction.device != g.device or direction.dtype != g.dtype:
            raise ValueError('direction must have the same shape, device and dtype as g')

    positive, zero, negative = target_counts
    active_count = positive + negative
    radius = _norm(g)
    if radius == 0:
        if active_count:
            raise ValueError('infeasible: zero norm with nonzero target signs')
        return g.clone()
    if not active_count:
        raise ValueError('infeasible: nonzero norm with an all-zero target')

    # A finite float32 vector can have a norm larger than float32's maximum.
    # Keep the scalar division wide, then restore the working dtype.
    q = (g.double() / radius).to(g.dtype)
    if direction is not None and direction_weight:
        direction_norm = _norm(direction)
        if direction_norm:
            # Dividing the objective by a positive constant leaves its optimum
            # unchanged and avoids overflow for large, finite weights.
            scale = max(1.0, direction_weight)
            direction_unit = (direction.double() / direction_norm).to(g.dtype)
            q = q / scale + (direction_weight / scale) * direction_unit

    order = torch.argsort(q, stable=True)
    indices = torch.cat((order[:negative], order[negative + zero:]))
    signs = torch.cat((-torch.ones(negative, device=g.device, dtype=g.dtype),
                       torch.ones(positive, device=g.device, dtype=g.dtype)))
    signed_scores = signs * q[indices]
    floor = epsilon_rel / math.sqrt(active_count)
    favorable = signed_scores.clamp_min(0)
    favorable_norm = _norm(favorable)
    if favorable_norm:
        unit = favorable / favorable_norm
        # Work on a unit sphere: avoids squaring radius in low precision.
        low = g.new_tensor(0.)
        high = g.new_tensor(1.)
        for _ in range(56 if g.dtype == torch.float64 else 32):
            middle = (low + high) / 2
            amplitudes = (middle * unit).clamp_min(floor)
            too_small = torch.dot(amplitudes, amplitudes) < 1
            low = torch.where(too_small, middle, low)
            high = torch.where(too_small, high, middle)
        amplitudes = (high * unit).clamp_min(floor)
    else:
        # All scores are nonpositive. A global optimum concentrates the energy
        # above the common floor in the least unfavorable coordinate.
        amplitudes = torch.full_like(signed_scores, floor)
        amplitudes[torch.argmax(signed_scores)] = math.sqrt(
            1 - (active_count - 1) * floor * floor)

    result = torch.zeros_like(g)
    result[indices] = (signs.double() * amplitudes.double() * radius).to(g.dtype)
    if not torch.isfinite(result).all() or (result[indices] == 0).any():
        raise ValueError('requested sphere/signs are not representable in this dtype')
    return result


class SPEARSLSphereAdversary(Adversary):
    """Norm-preserving SignLoader with explicit distance and direction metrics.

    ``max_relative_l2`` is a hard budget on ||x-g||/||g||. The direction target
    is a preference: failure to find it is recorded in ``last_diagnostics``.
    DnC is a local proxy only; server masks and other defenses may differ.
    ``visible_benign_limit`` subsamples visible benign updates for an
    information-limited ablation, not a reconstruction from local data.
    """

    def __init__(self, z_values=None, direction_weights=(0., .25, 1., 4.),
                 retention_target=.9, max_relative_l2=.5, epsilon_rel=1e-6,
                 dnc_sub_dim=10000, dnc_num_iters=5, dnc_filter_frac=1.,
                 visible_benign_limit=None):
        super().__init__()
        self.z_values = tuple(z_values if z_values is not None else
                              (i * .5 for i in range(11)))
        self.direction_weights = tuple(sorted(set((0.,) + tuple(direction_weights))))
        for name, values in [('z_values', self.z_values),
                             ('direction_weights', self.direction_weights)]:
            if not values or any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError(f'{name} must contain finite nonnegative numbers')
        if not math.isfinite(retention_target) or not 0 <= retention_target <= 1:
            raise ValueError('retention_target must be between zero and one')
        if not math.isfinite(max_relative_l2) or not 0 <= max_relative_l2 <= 2:
            raise ValueError('max_relative_l2 must be between zero and two')
        if not math.isfinite(epsilon_rel) or not 0 < epsilon_rel < 1:
            raise ValueError('epsilon_rel must lie strictly between zero and one')
        for name, value in [('dnc_sub_dim', dnc_sub_dim), ('dnc_num_iters', dnc_num_iters)]:
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(dnc_filter_frac) or dnc_filter_frac < 0:
            raise ValueError('dnc_filter_frac must be finite and nonnegative')
        if visible_benign_limit is not None and (
                not isinstance(visible_benign_limit, Integral)
                or isinstance(visible_benign_limit, bool) or visible_benign_limit < 1):
            raise ValueError('visible_benign_limit must be a positive integer or None')
        self.retention_target = retention_target
        self.max_relative_l2 = max_relative_l2
        self.epsilon_rel = epsilon_rel
        self.dnc_sub_dim = dnc_sub_dim
        self.dnc_num_iters = dnc_num_iters
        self.dnc_filter_frac = dnc_filter_frac
        self.visible_benign_limit = visible_benign_limit
        self.last_diagnostics = {}

    @staticmethod
    def _get_average_sign_stats(benign):
        # Largest-remainder rounding avoids always assigning the remainder to +.
        counts = torch.stack(((benign > 0).sum(1), (benign == 0).sum(1),
                              (benign < 0).sum(1)), dim=1).double().mean(0)
        rounded = counts.floor().long()
        remainder = benign.shape[1] - int(rounded.sum())
        order = torch.argsort(counts - rounded, descending=True, stable=True)
        rounded[order[:remainder]] += 1
        return tuple(rounded.tolist())

    def _count_dnc_passes(self, benign, candidate, num_malicious, subsets):
        updates = torch.cat((benign, candidate.unsqueeze(0).expand(num_malicious, -1)))
        n = updates.shape[0]
        keep_count = max(0, n - int(self.dnc_filter_frac * num_malicious))
        trusted = torch.ones(n, dtype=torch.bool, device=updates.device)
        for indices in subsets:
            selected = updates[:, indices]
            centered = selected - selected.mean(0)
            # Numerical failure is not replaced with an unrelated random test.
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            scores = (centered @ vh[0]).square()
            keep = torch.argsort(scores, stable=True)[:keep_count]
            current = torch.zeros_like(trusted)
            current[keep] = True
            trusted &= current
        return int(trusted[benign.shape[0]:].sum())

    def _metrics(self, x, original, mean, direction_unit, weight, z):
        radius = _norm(original)
        relative = _norm(x.double() - original.double()) / radius if radius else 0.
        projection = torch.dot((x - mean).double(), direction_unit.double()).item()
        original_projection = torch.dot((original - mean).double(), direction_unit.double()).item()
        # A zero original displacement has no meaningful retention ratio.
        retention = projection / original_projection if original_projection > 0 else None
        target = self.retention_target * original_projection
        tolerance = 32 * torch.finfo(x.dtype).eps * max(radius, _norm(mean))
        return dict(z=float(z), direction_weight=float(weight),
                    original_norm=radius, final_norm=_norm(x),
                    relative_l2=relative, direction_projection=projection,
                    original_direction_projection=original_projection,
                    direction_retention=retention,
                    retention_target_met=projection + tolerance >= target)

    @torch.no_grad()
    def generate_update(self, benign_updates, num_malicious):
        self.last_diagnostics = {}
        if (not isinstance(benign_updates, torch.Tensor) or benign_updates.ndim != 2
                or not all(benign_updates.shape)):
            raise ValueError('benign_updates must be a nonempty matrix')
        _validate_vector(benign_updates.reshape(-1), 'benign_updates')
        if not isinstance(num_malicious, Integral) or isinstance(num_malicious, bool) or num_malicious < 1:
            raise ValueError('num_malicious must be a positive integer')
        benign = benign_updates
        if self.visible_benign_limit is not None and benign.shape[0] > self.visible_benign_limit:
            ids = torch.randperm(benign.shape[0], device=benign.device)[:self.visible_benign_limit]
            benign = benign[ids]
        mean = benign.mean(0)
        std = benign.std(0) if benign.shape[0] > 1 else torch.zeros_like(mean)
        std_norm = _norm(std)
        direction_unit = ((std.double() / std_norm).to(std.dtype) if std_norm
                          else torch.zeros_like(std))
        counts = self._get_average_sign_stats(benign)
        # Common local masks make candidate comparisons consistent. They do not
        # assume knowledge of the actual server's random subsets.
        subsets = [torch.randperm(benign.shape[1], device=benign.device)[:self.dnc_sub_dim]
                   for _ in range(self.dnc_num_iters)]
        best = None
        best_key = None
        evaluations = 0
        for z in self.z_values:
            original = mean + z * std
            # These two cases are mathematically infeasible, not search failures.
            if (_norm(original) == 0) != (counts[0] + counts[2] == 0):
                continue
            candidates = []
            for weight in self.direction_weights:
                x = project_sign_sphere(original, counts, direction=std,
                                        direction_weight=weight, epsilon_rel=self.epsilon_rel)
                metrics = self._metrics(x, original, mean, direction_unit, weight, z)
                if metrics['relative_l2'] > self.max_relative_l2 + 8 * torch.finfo(x.dtype).eps:
                    # At lambda=0 the distance is globally minimal. If it fails,
                    # no vector at this radius can satisfy the sign/distance constraints.
                    if weight == 0:
                        break
                    continue
                if not any(torch.equal(x, other[0]) for other in candidates):
                    candidates.append((x, metrics))
            if not candidates:
                continue
            closest = candidates[0]
            target_met = [c for c in candidates if c[1]['retention_target_met']]
            preferred = (min(target_met, key=lambda c: c[1]['relative_l2']) if target_met else
                         max(candidates, key=lambda c: c[1]['direction_projection']))
            to_screen = [preferred]
            if not torch.equal(preferred[0], closest[0]):
                to_screen.append(closest)
            for x, metrics in to_screen:
                passed = self._count_dnc_passes(benign, x, num_malicious, subsets)
                evaluations += 1
                # Survival first, then actual retained displacement, not nominal z.
                key = (passed, metrics['direction_projection'], -metrics['relative_l2'])
                if best_key is None or key > best_key:
                    best_key = key
                    best = (x, dict(metrics, proxy_passed=passed,
                                    proxy_all_passed=passed == num_malicious))
                if passed == num_malicious:
                    break
        if best is None:
            raise ValueError('No sign/norm candidate meets the L2 distance budget on z_values; '
                             'increase max_relative_l2 or change z_values after inspecting the data')
        result, diagnostics = best
        self.last_diagnostics = dict(diagnostics, target_counts=counts,
                                     observed_benign=benign.shape[0],
                                     participating_malicious=num_malicious,
                                     dnc_evaluations=evaluations)
        if not diagnostics['proxy_all_passed']:
            logger.warning('SignLoader sphere: no screened candidate passed the DnC proxy for all clients')
        return result.clone()

    def on_local_round_end(self, trainer):
        malicious_results = [r for r in trainer.local_results
                             if trainer.client_manager.get_client_by_id(r[CLIENT_ID]).is_malicious]
        if not malicious_results:
            self.last_diagnostics = {}
            return
        update = self.generate_update(self.get_benign_updates(trainer), len(malicious_results))
        for result in malicious_results:
            result[CLIENT_UPDATE] = update.clone()
