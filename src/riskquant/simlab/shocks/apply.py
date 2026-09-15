# src/riskquant/simlab/shocks/apply.py
"""
Apply shocks to base levels.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.simlab.scenarios.base import ScenarioSetBatch
from riskquant.simlab.shocks.shock_kernel import _kernel_apply_shocks

# ============================================================
# Python dispatch wrapper
# ============================================================


def apply_shocks_to_base_levels(
    *,
    base_levels: npt.NDArray[np.float64],
    batch: ScenarioSetBatch,
    out: npt.NDArray[np.float64],
) -> None:
    """
    Hot-path entry. Validates nothing: contract enforcement lives in
    `validate_apply_shocks_buffers` (run once at runner build time).

    `out` MUST be pre-allocated by the caller — `(N, B)` float64,
    C-contiguous — and is mutated in place.

    dtype seam: `shock_scale` is float32 upstream, float64 in the kernel
    -------------------------------------------------------------------
    `ProxyResolution.shock_scale` is float32 by contract (`proxy_audit`
    enforces it), and the scenario generators pass that array through to
    `ScenarioSetBatch` untouched. `_kernel_apply_shocks`, however, is
    layout-and-dtype pinned to a float64 `shock_scale`, because the scale
    multiplies the shock INSIDE `exp(shock * scale)` and the whole
    level-reconstruction arithmetic is done in f64 (the same reason
    `base_levels` and `out` are f64: a small shock against a large level is
    a cancelling operation that must not lose significance before it is
    narrowed downstream).

    Both contracts are correct on their own terms, so the promotion belongs
    here, at the execution boundary, rather than in either owner. The cast
    is `(N,)`-sized — once per batch, against `N * B` kernel work — so it is
    not a hot-path cost. A no-op when a caller already supplies float64.
    """
    shock_scale = np.ascontiguousarray(batch.shock_scale, dtype=np.float64)

    # Layout note (RQ-AUD-017, consumer side): a trailing batch of exactly one
    # scenario is reported by numpy as both C- and F-contiguous, and Numba
    # resolves that ambiguity to 'C'. `_kernel_apply_shocks` therefore carries
    # BOTH layout overloads rather than forcing a copy here; see its signature.
    _kernel_apply_shocks(
        base_levels,
        batch.shocks,
        batch.required_to_shock_uix,
        shock_scale,
        batch.required_shift_type_id,
        out,
    )


def validate_apply_shocks_buffers(
    *,
    base_levels: npt.NDArray[np.float64],
    shocks: npt.NDArray[np.float32],
    out: npt.NDArray[np.float64],
) -> None:
    """
    One-shot layout/dtype validation. Run at runner construction time
    so the per-batch hot path can trust the contract.

    Raises
    ------
    ValueError
        On any layout or dtype mismatch with the kernel signature.
    """
    if base_levels.dtype != np.float64 or base_levels.ndim != 1:
        raise ValueError("base_levels must be 1-D float64.")
    if not base_levels.flags.c_contiguous:
        raise ValueError("base_levels must be C-contiguous.")

    if shocks.dtype != np.float32 or shocks.ndim != 2:
        raise ValueError("shocks must be 2-D float32.")
    if not shocks.flags.f_contiguous:
        raise ValueError(
            "shocks must be F-contiguous. The generator is responsible "
            "for emitting `order='F'` so the apply kernel reads down "
            "the scenario axis with unit stride."
        )

    if out.dtype != np.float64 or out.ndim != 2:
        raise ValueError("out must be 2-D float64.")
    if not out.flags.c_contiguous:
        raise ValueError("out must be C-contiguous.")
    if out.shape != (base_levels.shape[0], shocks.shape[0]):
        raise ValueError(
            f"out shape {out.shape} does not match "
            f"(N={base_levels.shape[0]}, B={shocks.shape[0]})."
        )
