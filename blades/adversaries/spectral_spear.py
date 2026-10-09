"""Experimental SVD-conditioned directions and finite DnC screening.

The ellipsoidal direction subproblem has a closed-form optimum. Neither that
surrogate nor acceptance on sampled proxy banks guarantees server acceptance
or training damage. See docs/spear_spectral.md for the mathematical contract.
"""

from dataclasses import asdict, dataclass
import math
from numbers import Integral

import numpy as np
import torch


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _finite_nonnegative(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')


@dataclass(frozen=True)
class SpectralSpearConfig:
    z_values: tuple = (0., .5, 1., 2., 4.)
    spectral_penalties: tuple = (0., 1., 10.)
    spectral_rank: int = 2
    fit_subspaces: int = 3
    dnc_sub_dim: int = 10000
    dnc_num_iters: int = 5
    search_banks: int = 2
    audit_banks: int = 4
    dnc_filter_frac: float = 1.
    seed: int = 2026
    min_survival: float = 1.

    def __post_init__(self):
        for name in ('z_values', 'spectral_penalties'):
            values = tuple(getattr(self, name))
            if not values:
                raise ValueError(f'{name} must be nonempty')
            for value in values:
                _finite_nonnegative(value, name)
            object.__setattr__(self, name, tuple(sorted(set(float(v) for v in values))))
        for name in ('spectral_rank', 'fit_subspaces', 'dnc_sub_dim',
                     'dnc_num_iters', 'search_banks', 'audit_banks'):
            _positive_int(getattr(self, name), name)
        _positive_int(self.seed, 'seed', minimum=0)
        _finite_nonnegative(self.dnc_filter_frac, 'dnc_filter_frac')
        if not math.isfinite(self.min_survival) or not 0 < self.min_survival <= 1:
            raise ValueError('min_survival must be in (0, 1]')


def _matrix(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or not all(tensor.shape):
        raise ValueError('updates must be a nonempty matrix')
    if tensor.dtype not in (torch.float32, torch.float64):
        raise TypeError('updates must have dtype float32 or float64')
    if not torch.isfinite(tensor).all():
        raise ValueError('updates must be finite')


def _finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise ValueError(f'{name} is not finite; rescale inputs')
    return tensor


def _spaces(subspaces, dimension, device):
    spaces = tuple(subspaces)
    if not spaces:
        raise ValueError('subspaces must be nonempty')
    checked = []
    for ids in spaces:
        ids = torch.as_tensor(ids, device=device)
        if (ids.ndim != 1 or not ids.numel() or ids.dtype != torch.int64
                or int(ids.min()) < 0 or int(ids.max()) >= dimension
                or len(torch.unique(ids)) != len(ids)):
            raise ValueError('subspaces must contain distinct valid int64 coordinates')
        checked.append(ids)
    return tuple(checked)


@dataclass(frozen=True)
class SpectralMetric:
    dimension: int
    coordinates: torch.Tensor
    factors: torch.Tensor


@torch.no_grad()
def fit_spectral_metric(benign, subspaces, rank=2):
    r"""Build H = I + penalty A A^T without allocating a d-by-d matrix.

    Each bank contributes lifted top-r benign right singular vectors weighted
    by s_j/s_1/sqrt(number of subspaces). Zero numerical singular modes are
    omitted. A is stored only on the union of sampled coordinates.
    """
    _matrix(benign)
    _positive_int(rank, 'rank')
    spaces = _spaces(subspaces, benign.shape[1], benign.device)
    coordinates = torch.unique(torch.cat(spaces), sorted=True)
    pieces = []
    for ids in spaces:
        # Anchoring first makes constant coordinates exactly zero even when
        # their floating-point mean would have a small rounding error.
        shifted = _finite(benign[:, ids] - benign[0, ids], 'shifted data')
        centered = _finite(shifted - shifted.mean(0), 'centered data')
        # No arbitrary random-direction substitution on numerical failure.
        _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        if singular[0] == 0:
            continue
        threshold = torch.finfo(benign.dtype).eps * max(centered.shape) * singular[0]
        count = min(rank, int((singular > threshold).sum()))
        if count:
            local = (vh[:count].T * (singular[:count] / singular[0])) / math.sqrt(len(spaces))
            pieces.append((torch.searchsorted(coordinates, ids), local))
    width = sum(local.shape[1] for _, local in pieces)
    factors = benign.new_zeros((len(coordinates), width))
    offset = 0
    for locations, local in pieces:
        factors[locations, offset:offset + local.shape[1]] = local
        offset += local.shape[1]
    return SpectralMetric(benign.shape[1], coordinates, _finite(factors, 'spectral factors'))


@torch.no_grad()
def spectral_direction(mean, metric, penalty, radius):
    r"""Argmax -mean^T delta subject to delta^T H delta <= radius^2.

    Woodbury solves only a (rank * number_of_subspaces)-square system. The
    returned direction includes coordinates outside the sampled union, where
    H is the identity. This is a soft spectral penalty, not a nullspace claim.
    """
    _finite_nonnegative(penalty, 'penalty')
    _finite_nonnegative(radius, 'radius')
    if mean.ndim != 1 or mean.numel() != metric.dimension:
        raise ValueError('mean and metric dimensions differ')
    _matrix(mean.unsqueeze(0))
    wide_mean = mean.double()
    norm = _finite(torch.linalg.vector_norm(wide_mean), 'mean norm')
    if norm == 0 or radius == 0:
        return torch.zeros_like(mean)
    q = -wide_mean / norm
    result = q.clone()
    # Woodbury involves subtraction of nearly equal vectors at large penalty.
    # Solve and accumulate in float64, even for float32 uploads.
    factors = metric.factors.double()
    if penalty and factors.shape[1]:
        scaled = factors * math.sqrt(penalty)
        system = torch.eye(scaled.shape[1], dtype=torch.float64, device=mean.device) + scaled.T @ scaled
        coeff = torch.linalg.solve(system, scaled.T @ q[metric.coordinates])
        result[metric.coordinates] -= scaled @ coeff
    energy = _finite(torch.dot(q, result), 'ellipsoid inverse energy')
    if energy <= 0:
        raise ValueError('ellipsoid solve lost positive definiteness; reduce penalty')
    # Normalize with actual quadratic energy rather than assuming the rounded
    # inverse product still satisfies x^T H x = q^T x exactly.
    h_norm = torch.sqrt(result.square().sum() +
                        penalty * (factors.T @ result[metric.coordinates]).square().sum())
    direction = _finite((result * (radius / h_norm)).to(mean.dtype), 'spectral direction')
    uploaded = direction.double()
    actual_norm = _finite(torch.sqrt(uploaded.square().sum() +
                         penalty * (factors.T @ uploaded[metric.coordinates]).square().sum()),
                         'uploaded ellipsoid norm')
    if actual_norm > radius:
        # Leave a tiny rounding margin when casting back to the upload dtype.
        scale = (radius / actual_norm) * (1. - 2. * torch.finfo(mean.dtype).eps)
        direction = (uploaded * scale).to(mean.dtype)
    return direction


@torch.no_grad()
def dnc_select(updates, subspaces, drop_count):
    """Local DnC semantics: mixed centering, top-1 squared score, intersection.

    NumPy's default argsort follows this checkout's DnC tie policy. Exact and
    near ties may still depend on BLAS/device/library versions.
    """
    _matrix(updates)
    _positive_int(drop_count, 'drop_count', minimum=0)
    if drop_count >= updates.shape[0]:
        raise ValueError('drop_count must be smaller than the participating population')
    spaces = _spaces(subspaces, updates.shape[1], updates.device)
    trusted = torch.ones(updates.shape[0], dtype=torch.bool, device=updates.device)
    for ids in spaces:
        sub = updates[:, ids]
        centered = _finite(sub - sub.mean(0), 'centered mixed data')
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        # Match the existing defense's per-row dot-product scoring.
        scores = _finite(torch.stack([torch.dot(row, vh[0]).square() for row in centered]), 'DnC scores')
        order = np.argsort(scores.cpu().double().numpy())
        keep = torch.as_tensor(order[:updates.shape[0] - drop_count], device=updates.device)
        current = torch.zeros_like(trusted)
        current[keep] = True
        trusted &= current
    return trusted


def _generator(seed, round_index, stream):
    # Separate, deterministic CPU streams never consume training/server RNG.
    value = (seed + 1000003 * round_index + 15485863 * stream) % (2**63 - 1)
    return torch.Generator(device='cpu').manual_seed(value), value


def _sample(dimension, count, sub_dim, generator, device):
    return tuple(torch.randperm(dimension, generator=generator)[:min(dimension, sub_dim)].to(device)
                 for _ in range(count))


def _aggregate(updates, mask):
    return updates[mask].mean(0) if mask.any() else torch.zeros_like(updates[0])


def _population(benign, candidate, positions):
    n = benign.shape[0] + positions.numel()
    malicious = torch.zeros(n, dtype=torch.bool, device=benign.device)
    malicious[positions] = True
    mixed = benign.new_empty((n, benign.shape[1]))
    mixed[~malicious] = benign
    mixed[malicious] = candidate
    return mixed, malicious


def _evaluate(mixed, malicious, banks, drop_count, reference, mean_unit):
    retained, damages = [], []
    for bank, ref in zip(banks, reference):
        mask = dnc_select(mixed, bank, drop_count)
        retained.append(int(mask[malicious].sum()))
        aggregate = _finite(_aggregate(mixed, mask), 'aggregate')
        damage = _finite(-torch.dot(mean_unit, aggregate - ref), 'damage proxy')
        damages.append(float(damage))
    count = int(malicious.sum())
    return dict(retained_malicious=retained,
                all_survival_rate=sum(p == count for p in retained) / len(banks),
                damage_by_bank=damages, worst_damage=min(damages))


@torch.no_grad()
def search_spectral_update(benign, num_malicious, *, config=None, round_index=0,
                           malicious_positions=None):
    """Return (update, JSON-compatible diagnostics) under full benign visibility.

    All candidates share search banks, but recompute mixed SVD for each. An
    independent bank evaluates the selected vector once, without reselection.
    No feasible candidate => benign mean with an explicit failed-search status.
    """
    config = config or SpectralSpearConfig()
    _matrix(benign)
    _positive_int(num_malicious, 'num_malicious')
    _positive_int(round_index, 'round_index', minimum=0)
    n_benign, dimension = benign.shape
    n = n_benign + num_malicious
    drop_count = int(config.dnc_filter_frac * num_malicious)
    if drop_count >= n:
        raise ValueError('filter fraction removes the entire participating population')
    if malicious_positions is None:
        positions = torch.arange(n_benign, n, device=benign.device)
    else:
        positions = torch.as_tensor(malicious_positions, device=benign.device)
        if (positions.ndim != 1 or positions.dtype != torch.int64
                or positions.numel() != num_malicious
                or int(positions.min()) < 0 or int(positions.max()) >= n
                or positions.unique().numel() != num_malicious):
            raise ValueError('malicious_positions must be distinct participating row positions')
    mean = _finite(benign.mean(0), 'mean')
    std = benign.std(0) if n_benign > 1 else torch.zeros_like(mean)
    radius = float(_finite(std.norm(), 'standard-deviation norm'))
    mean_norm = _finite(mean.norm(), 'mean norm')
    mean_unit = mean / mean_norm if mean_norm > 0 else torch.zeros_like(mean)
    generators, seeds = {}, {}
    for stream, name in enumerate(('fit', 'search', 'audit'), start=1):
        generators[name], seeds[name] = _generator(config.seed, round_index, stream)
    fit_spaces = _sample(dimension, config.fit_subspaces, config.dnc_sub_dim,
                         generators['fit'], benign.device)
    metric = fit_spectral_metric(benign, fit_spaces, config.spectral_rank)
    banks = tuple(_sample(dimension, config.dnc_num_iters, config.dnc_sub_dim,
                          generators['search'], benign.device) for _ in range(config.search_banks))
    reference_mixed, malicious = _population(benign, mean, positions)
    references = [_aggregate(reference_mixed, dnc_select(reference_mixed, bank, drop_count))
                  for bank in banks]
    evaluations = []
    best_update, best_entry, best_key = None, None, None
    mean_evaluated = False
    for penalty in config.spectral_penalties:
        direction = spectral_direction(mean, metric, penalty, radius)
        for z in config.z_values:
            if z == 0 and mean_evaluated:
                continue
            mean_evaluated |= z == 0
            candidate = _finite(mean + z * direction, 'candidate')
            mixed, _ = _population(benign, candidate, positions)
            entry = _evaluate(mixed, malicious, banks, drop_count, references, mean_unit)
            entry.update(z=float(z), spectral_penalty=float(penalty),
                         perturbation_norm=float((candidate - mean).norm()),
                         feasible=entry['all_survival_rate'] >= config.min_survival)
            evaluations.append(entry)
            # Worst-bank directional damage, then survival, then smaller update.
            key = (entry['worst_damage'], entry['all_survival_rate'], -entry['perturbation_norm'])
            if entry['feasible'] and (best_key is None or key > best_key):
                best_update, best_entry, best_key = candidate.clone(), dict(entry), key
    selected = best_update is not None
    if not selected:
        best_update = mean.clone()
    # Audit is diagnostic only. Changing audit_banks cannot change selection.
    audit_banks = tuple(_sample(dimension, config.dnc_num_iters, config.dnc_sub_dim,
                                generators['audit'], benign.device) for _ in range(config.audit_banks))
    audit_references = [_aggregate(reference_mixed, dnc_select(reference_mixed, bank, drop_count))
                        for bank in audit_banks]
    final_mixed, _ = _population(benign, best_update, positions)
    audit = _evaluate(final_mixed, malicious, audit_banks, drop_count, audit_references, mean_unit)
    diagnostics = dict(
        method='SPEAR-Spectral experimental',
        status='proxy_feasible' if selected else 'no_feasible_candidate',
        selected=best_entry, worst_damage=best_entry['worst_damage'] if selected else None,
        evaluations=evaluations, audit=audit, config=asdict(config),
        round_index=int(round_index), seeds=seeds, observed_benign=n_benign,
        participating_malicious=num_malicious, malicious_positions=positions.tolist(),
        dimension=dimension, metric_columns=metric.factors.shape[1],
        metric_coordinates=metric.coordinates.numel(),
        knowledge_model='all participating benign updates and identities',
        objective='worst-bank anti-mean shift relative to mean-replacement control',
        svd_calls=(config.fit_subspaces +
                   (len(evaluations) + 1) * config.search_banks * config.dnc_num_iters +
                   2 * config.audit_banks * config.dnc_num_iters),
    )
    return best_update, diagnostics
