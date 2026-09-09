# src/riskquant/simlab/scenarios/parametric.py
"""
Parametric Monte Carlo scenario generator.

Supports Gaussian-Scale-Mixture (GSM) distributions: standard normal,
Student-t, Laplace, variance gamma, and a regime-switching Gaussian
mixture (crisis-vol multipliers).

Pipeline per batch
------------------
1. Draw raw (B, K_total) i.i.d. normals into a pre-allocated work
   buffer; `rng.standard_normal(out=...)` writes in place.
2. Apply the (K_total x K_req) Cholesky-transpose slice via
   `np.matmul(work, L_T, out=correlated)` to induce cross-asset
   correlation. K_total appears on the contracted axis so the full
   correlation structure is preserved even though only K_req shocks
   are emitted.
3. Distribution-specific row scaling for the GSM mixing variable.
4. Multiply by per-factor base vols.
5. Down-cast into the pre-allocated F-contig float32 shocks buffer.

Implementation
--------------
- `build_shock_universe` replaces the previous dict-loop path; gathers
  Cholesky columns / vols / GMM multipliers via vectorised `searchsorted`.
- `work`, `correlated`, and `shocks_buf` are pre-allocated once at
  `batch_size`; the streaming loop only re-slices.
- Shocks always emitted as float32.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from riskquant.simlab.builder.shock_universe import build_shock_universe
from riskquant.simlab.scenarios.base import (
    ScenarioGenerator,
    ScenarioSetBatch,
)

DistributionKind = Literal[
    "standard_normal",
    "student_t",
    "laplace",
    "variance_gamma",
    "regime_switching_gmm",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ParametricGenerator(ScenarioGenerator):
    """
    Parametric shock-space generator over Gaussian-Scale-Mixtures.
    Shocks are always emitted as float32.
    """

    available_factor_ids: np.ndarray
    cholesky_matrix: np.ndarray
    volatilities: np.ndarray
    total_scenarios: int
    distribution_kind: DistributionKind
    seed: int | None = field(default=None)
    student_t_df: float | None = field(default=None)
    vg_nu: float | None = field(default=None)
    prob_crisis: float | None = field(default=None)
    crisis_vol_multipliers: float | np.ndarray | None = field(default=None)

    # Derived in __post_init__.
    _cholesky_transpose: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        available_factor_ids = np.asarray(self.available_factor_ids, dtype=np.int32)
        if available_factor_ids.ndim != 1:
            raise ValueError("available_factor_ids must be 1D")

        cholesky_matrix = np.asarray(self.cholesky_matrix, dtype=np.float64)
        if cholesky_matrix.ndim != 2:
            raise ValueError("cholesky_matrix must be 2D")
        if cholesky_matrix.shape[0] != cholesky_matrix.shape[1]:
            raise ValueError("cholesky_matrix must be square")
        if available_factor_ids.shape[0] != cholesky_matrix.shape[0]:
            raise ValueError(
                "available_factor_ids length must match cholesky_matrix dimension"
            )
        if not np.isfinite(cholesky_matrix).all():
            raise ValueError("cholesky_matrix must contain only finite values")

        volatilities = np.asarray(self.volatilities, dtype=np.float64)
        if volatilities.ndim != 1:
            raise ValueError("volatilities must be 1D")
        if volatilities.shape[0] != cholesky_matrix.shape[0]:
            raise ValueError("volatilities length must match cholesky_matrix dimension")
        if not np.isfinite(volatilities).all():
            raise ValueError("volatilities must contain only finite values")
        if np.any(volatilities <= 0.0):
            raise ValueError("volatilities must be strictly positive")

        total_scenarios = int(self.total_scenarios)
        if total_scenarios <= 0:
            raise ValueError("total_scenarios must be positive")

        if self.seed is not None and not isinstance(self.seed, (int, np.integer)):
            raise TypeError("seed must be an integer or None")

        if self.distribution_kind == "student_t":
            if self.student_t_df is None:
                raise ValueError("student_t_df is required for student_t")
            if not np.isfinite(self.student_t_df) or self.student_t_df <= 2.0:
                raise ValueError("student_t_df must be finite and greater than 2")

        if self.distribution_kind == "variance_gamma":
            if self.vg_nu is None:
                raise ValueError("vg_nu is required for variance_gamma")
            if not np.isfinite(self.vg_nu) or self.vg_nu <= 0.0:
                raise ValueError("vg_nu must be finite and positive")

        if self.distribution_kind == "regime_switching_gmm":
            if self.prob_crisis is None:
                raise ValueError("prob_crisis is required for regime_switching_gmm")
            if not (np.isfinite(self.prob_crisis) and 0.0 <= self.prob_crisis <= 1.0):
                raise ValueError("prob_crisis must be in [0, 1]")
            if self.crisis_vol_multipliers is None:
                raise ValueError(
                    "crisis_vol_multipliers is required for regime_switching_gmm"
                )
            if isinstance(
                self.crisis_vol_multipliers,
                (int, float, np.integer, np.floating),
            ):
                crisis_mult: float | np.ndarray = float(self.crisis_vol_multipliers)
                if not np.isfinite(crisis_mult) or crisis_mult <= 0.0:
                    raise ValueError(
                        "crisis_vol_multipliers must be finite and positive"
                    )
            else:
                crisis_mult = np.asarray(self.crisis_vol_multipliers, dtype=np.float64)
                if crisis_mult.ndim != 1:
                    raise ValueError("crisis_vol_multipliers must be scalar or 1D")
                if crisis_mult.shape[0] != cholesky_matrix.shape[0]:
                    raise ValueError(
                        "crisis_vol_multipliers length must match cholesky_matrix dimension"
                    )
                if not np.isfinite(crisis_mult).all() or np.any(crisis_mult <= 0.0):
                    raise ValueError(
                        "crisis_vol_multipliers must contain only positive finite values"
                    )
            object.__setattr__(self, "crisis_vol_multipliers", crisis_mult)

        object.__setattr__(self, "available_factor_ids", available_factor_ids)
        object.__setattr__(
            self, "cholesky_matrix", np.ascontiguousarray(cholesky_matrix)
        )
        object.__setattr__(self, "volatilities", volatilities)
        object.__setattr__(self, "total_scenarios", total_scenarios)
        object.__setattr__(
            self, "_cholesky_transpose", np.ascontiguousarray(cholesky_matrix.T)
        )

    # ---- Factory helpers (classmethods unchanged in spirit; thinned) ----

    @classmethod
    def standard_normal(
        cls,
        available_factor_ids: np.ndarray,
        cholesky_matrix: np.ndarray,
        volatilities: np.ndarray,
        total_scenarios: int,
        seed: int | None = None,
    ) -> ParametricGenerator:
        return cls(
            available_factor_ids=available_factor_ids,
            cholesky_matrix=cholesky_matrix,
            volatilities=volatilities,
            total_scenarios=total_scenarios,
            distribution_kind="standard_normal",
            seed=seed,
        )

    @classmethod
    def student_t(
        cls,
        student_t_df: float,
        available_factor_ids: np.ndarray,
        cholesky_matrix: np.ndarray,
        volatilities: np.ndarray,
        total_scenarios: int,
        seed: int | None = None,
    ) -> ParametricGenerator:
        return cls(
            available_factor_ids=available_factor_ids,
            cholesky_matrix=cholesky_matrix,
            volatilities=volatilities,
            total_scenarios=total_scenarios,
            distribution_kind="student_t",
            student_t_df=float(student_t_df),
            seed=seed,
        )

    @classmethod
    def laplace(
        cls,
        available_factor_ids: np.ndarray,
        cholesky_matrix: np.ndarray,
        volatilities: np.ndarray,
        total_scenarios: int,
        seed: int | None = None,
    ) -> ParametricGenerator:
        return cls(
            available_factor_ids=available_factor_ids,
            cholesky_matrix=cholesky_matrix,
            volatilities=volatilities,
            total_scenarios=total_scenarios,
            distribution_kind="laplace",
            seed=seed,
        )

    @classmethod
    def variance_gamma(
        cls,
        vg_nu: float,
        available_factor_ids: np.ndarray,
        cholesky_matrix: np.ndarray,
        volatilities: np.ndarray,
        total_scenarios: int,
        seed: int | None = None,
    ) -> ParametricGenerator:
        return cls(
            available_factor_ids=available_factor_ids,
            cholesky_matrix=cholesky_matrix,
            volatilities=volatilities,
            total_scenarios=total_scenarios,
            distribution_kind="variance_gamma",
            seed=seed,
            vg_nu=float(vg_nu),
        )

    @classmethod
    def regime_switching_gmm(
        cls,
        prob_crisis: float,
        crisis_vol_multipliers: float | np.ndarray,
        available_factor_ids: np.ndarray,
        cholesky_matrix: np.ndarray,
        volatilities: np.ndarray,
        total_scenarios: int,
        seed: int | None = None,
    ) -> ParametricGenerator:
        return cls(
            available_factor_ids=available_factor_ids,
            cholesky_matrix=cholesky_matrix,
            volatilities=volatilities,
            total_scenarios=total_scenarios,
            distribution_kind="regime_switching_gmm",
            seed=seed,
            prob_crisis=float(prob_crisis),
            crisis_vol_multipliers=crisis_vol_multipliers,
        )

    # ---- Distribution scaling ----

    def _apply_distribution_scaling(
        self,
        *,
        rng: np.random.Generator,
        shocks: np.ndarray,
        gmm_sliced_multipliers: float | np.ndarray | None,
    ) -> None:
        kind = self.distribution_kind
        if kind == "standard_normal":
            return

        if kind == "student_t":
            assert self.student_t_df is not None
            chi_square = rng.chisquare(df=self.student_t_df, size=shocks.shape[0])
            row_scale = np.sqrt((self.student_t_df - 2.0) / chi_square)
            np.multiply(shocks, row_scale[:, None], out=shocks)
            return

        if kind == "laplace":
            exponential_mix = rng.exponential(scale=1.0, size=shocks.shape[0])
            row_scale = np.sqrt(exponential_mix)
            np.multiply(shocks, row_scale[:, None], out=shocks)
            return

        if kind == "variance_gamma":
            assert self.vg_nu is not None
            gamma_mix = rng.gamma(
                shape=1.0 / self.vg_nu, scale=self.vg_nu, size=shocks.shape[0]
            )
            row_scale = np.sqrt(gamma_mix)
            np.multiply(shocks, row_scale[:, None], out=shocks)
            return

        if kind == "regime_switching_gmm":
            assert self.prob_crisis is not None
            assert gmm_sliced_multipliers is not None
            crisis_mask = rng.random(shocks.shape[0]) < self.prob_crisis
            if crisis_mask.any():
                shocks[crisis_mask, :] *= gmm_sliced_multipliers

    # ---- Streaming entry ----

    def iter_batches(
        self,
        *,
        required_shift_type_id: np.ndarray,
        source_factor_ids: np.ndarray,
        shock_scale: np.ndarray,
        batch_size: int,
    ) -> Iterator[ScenarioSetBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        shock_universe = build_shock_universe(
            required_shift_type_id=required_shift_type_id,
            source_factor_ids=source_factor_ids,
            available_factor_ids=self.available_factor_ids,
        )
        plan = shock_universe.compute_plan
        k_req = plan.cols_sorted.shape[0]
        k_total = self.cholesky_matrix.shape[0]

        # Cholesky-transpose column slice is C-contig per column for
        # the trailing matmul. Vols and GMM multipliers gather likewise.
        col_indices = plan.cols_sorted
        l_t_sliced = np.ascontiguousarray(self._cholesky_transpose[:, col_indices])
        vols_sliced = np.ascontiguousarray(self.volatilities[col_indices])

        gmm_mults_sliced: float | np.ndarray | None = None
        if self.distribution_kind == "regime_switching_gmm":
            cvm = self.crisis_vol_multipliers
            if isinstance(cvm, np.ndarray):
                gmm_mults_sliced = np.ascontiguousarray(cvm[col_indices])
            else:
                gmm_mults_sliced = float(cvm) if cvm is not None else None

        rng = np.random.default_rng(self.seed)

        # ---- Pre-allocate hot-path buffers ONCE ----
        work = np.empty((batch_size, k_total), dtype=np.float64, order="C")
        correlated = np.empty((batch_size, k_req), dtype=np.float64, order="C")
        shocks_buf = np.empty((batch_size, k_req), dtype=np.float32, order="F")

        scenario_offset = 0
        while scenario_offset < self.total_scenarios:
            scenario_count = min(batch_size, self.total_scenarios - scenario_offset)

            work_view = work[:scenario_count, :]
            correlated_view = correlated[:scenario_count, :]
            shocks_view = shocks_buf[:scenario_count, :]

            # 1. Raw normals → work
            rng.standard_normal(size=work_view.shape, out=work_view)

            # 2. Correlation induction (B, K_total) @ (K_total, K_req)
            np.matmul(work_view, l_t_sliced, out=correlated_view)

            # 3. GSM row scaling (in place)
            self._apply_distribution_scaling(
                rng=rng,
                shocks=correlated_view,
                gmm_sliced_multipliers=gmm_mults_sliced,
            )

            # 4. Per-factor base vols (broadcast across rows)
            np.multiply(correlated_view, vols_sliced, out=correlated_view)

            # 5. Cast into the F-contig float32 streaming buffer.
            # `np.copyto` honours the buffer's F-layout writeback path.
            # np.copyto itself does not require shocks_view to be contiguous
            # (it writes element-wise through the view's strides either way),
            # but the *yielded* ScenarioSetBatch.shocks contract requires
            # genuine F-contiguity. A slice of an F-contiguous buffer along
            # its first axis is only itself F-contiguous when the slice
            # covers every row; the final (partial) batch is neither C nor
            # F contiguous. Materialise a contiguous copy only in that case.
            # (RQ-AUD-017.)
            np.copyto(shocks_view, correlated_view, casting="same_kind")
            shocks_out = (
                shocks_view
                if shocks_view.flags["F_CONTIGUOUS"]
                else np.asfortranarray(shocks_view)
            )

            yield ScenarioSetBatch(
                shocks=shocks_out,
                required_to_shock_uix=shock_universe.required_to_shock_uix,
                shock_scale=shock_scale,
                required_shift_type_id=required_shift_type_id,
                scenario_offset=scenario_offset,
                scenario_count_total=self.total_scenarios,
            )

            scenario_offset += scenario_count
