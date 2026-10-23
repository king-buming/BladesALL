"""Tensor clipping and compatibility helpers for historical public imports."""

from collections.abc import Mapping
from typing import Iterable, Union

import ray
import torch
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping as _ray_apply_grad_clipping,
    convert_to_non_torch_type as convert_to_non_torch_type,
    set_torch_seed as set_torch_seed,
)

FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38
_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def apply_grad_clipping(policy, optimizer, loss):
    """Use Ray clipping while retaining the legacy empty result for zero norms."""
    info = _ray_apply_grad_clipping(policy, optimizer, loss)
    return info if info.get("grad_gnorm", 0) > 0 else {}


def get_device(config):
    """Legacy config-based device selection, retained for external callers.

    Ray 2.39 no longer provides an equivalent config-dictionary helper.
    Current workers select their device from ``WorkerGroupScalingConfig``.
    """
    worker_idx = config.get("worker_index", 0)
    if (
        not config["_fake_gpus"]
        and ray._private.worker._mode() == ray._private.worker.LOCAL_MODE
    ):
        num_gpus = 0
    else:
        num_gpus = config["num_gpus" if worker_idx == 0 else "num_gpus_per_worker"]
    gpu_ids = list(range(torch.cuda.device_count()))
    if config["_fake_gpus"] or num_gpus == 0 or not gpu_ids:
        return torch.device("cpu")
    if ray._private.worker._mode() == ray._private.worker.WORKER_MODE:
        gpu_ids = ray.get_gpu_ids()
    if len(gpu_ids) < num_gpus:
        raise ValueError(
            "TorchPolicy was not able to find enough GPU IDs! Found "
            f"{gpu_ids}, but num_gpus={num_gpus}."
        )
    return torch.device("cuda")


def vector_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    """Copy a flat vector into trainable parameters, leaving frozen ones intact."""
    torch.nn.utils.vector_to_parameters(
        vec, (parameter for parameter in parameters if parameter.requires_grad)
    )


def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    """Flatten only trainable parameters, preserving the historical ordering."""
    return torch.nn.utils.parameters_to_vector(
        parameter for parameter in parameters if parameter.requires_grad
    )


def _parameter_list(parameters):
    if isinstance(parameters, torch.Tensor):
        return [parameters]
    return list(parameters.values() if isinstance(parameters, Mapping) else parameters)


def _norm_and_clip_coefficient(parameters, max_norm, norm_type, error_if_nonfinite):
    """Shared legacy norm calculation; keep its dtype and rounding behavior."""
    device = parameters[0].device
    if norm_type == torch.inf:
        norms = [parameter.detach().abs().max().to(device) for parameter in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(parameter.detach(), norm_type).to(device)
                    for parameter in parameters
                    if parameter.dtype != torch.int64
                ]
            ),
            norm_type,
        )
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm, torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)


def clip_para_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """Clip parameter values in place and return their original combined norm.

    The legacy mapping input is supported, along with tensors and iterables.
    Integer state entries remain unchanged.
    """
    parameters = _parameter_list(parameters)
    max_norm, norm_type = float(max_norm), float(norm_type)
    if not parameters:
        return torch.tensor(0.0)
    total_norm, coefficient = _norm_and_clip_coefficient(
        parameters, max_norm, norm_type, error_if_nonfinite
    )
    for parameter in parameters:
        if parameter.dtype != torch.int64:
            parameter.detach().mul_(coefficient.to(parameter.device))
    return total_norm


def clip_tensor_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """Clip an update tensor in place and return the clipped tensor.

    For legacy iterable inputs, the norm covers all entries but only the first
    non-int64 tensor is changed and returned. Use ``clip_para_norm_`` to change
    every parameter. Aggregators and DP clients pass a single update tensor.
    """
    parameters = _parameter_list(parameters)
    max_norm, norm_type = float(max_norm), float(norm_type)
    if not parameters:
        return torch.tensor(0.0)
    _, coefficient = _norm_and_clip_coefficient(
        parameters, max_norm, norm_type, error_if_nonfinite
    )
    for parameter in parameters:
        if parameter.dtype != torch.int64:
            return parameter.detach().mul_(coefficient.to(parameter.device))


__all__ = [
    "FLOAT_MIN",
    "FLOAT_MAX",
    "apply_grad_clipping",
    "convert_to_non_torch_type",
    "get_device",
    "vector_to_parameters",
    "parameters_to_vector",
    "set_torch_seed",
    "clip_para_norm_",
    "clip_tensor_norm_",
]
