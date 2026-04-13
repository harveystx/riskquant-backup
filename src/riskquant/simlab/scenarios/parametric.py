# src/riskquant/simlab/scenarios/parametric.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal

import numpy as np

from riskquant.simlab.scenarios.base import (
    ScenarioGenerator,
    ScenarioSetBatch,
    build_shock_key_plan,
)

DistributionKind = Literal[
    "standard_normal",
    "student_t",
    "laplace",
    "variance_gamma",
    "regime_switching_gmm",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ParametricMCScenarioGenerator(ScenarioGenerator):
    """
    Parametric shock-space scenario generator strictly supporting Gaussian Scale Mixtures.
    """

    factor_ids: np.ndarray
    cholesky_matrix: np.ndarray
    volatilities: np.ndarray
    total_scenarios: int
    distribution_kind: DistributionKind
    seed: int | None = None
    student_t_df: float | None = None
    vg_nu: float | None = None
    prob_crisis: float | None = None
    crisis_vol_multipliers: float | np.ndarray | None = None
    shock_dtype: type[np.float32] | type[np.float64] = np.float32
    _cholesky_transpose: np.ndarray = field(init=False, repr=False)
    _volatilities64: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        factor_ids = np.asarray(self.factor_ids, dtype=np.int32)
        if factor_ids.ndim != 1:
            raise ValueError("factor_ids must be 1D")

        cholesky_matrix = np.asarray(self.cholesky_matrix, dtype=np.float64)
        if cholesky_matrix.ndim != 2:
            raise ValueError("cholesky_matrix must be 2D")
        if cholesky_matrix.shape[0] != cholesky_matrix.shape[1]:
            raise ValueError("cholesky_matrix must be square")
        if factor_ids.shape[0] != cholesky_matrix.shape[0]:
            raise ValueError(
                "factor_ids length must match cholesky_matrix dimension"
            )
        if not np.isfinite(cholesky_matrix).all():
            raise ValueError("cholesky_matrix must contain only finite values")

        volatilities = np.asarray(self.volatilities, dtype=np.float64)
        if volatilities.ndim != 1:
            raise ValueError("volatilities must be 1D")
        if volatilities.shape[0] != cholesky_matrix.shape[0]:
            raise ValueError(
                "volatilities length must match cholesky_matrix dimension"
            )
        if not np.isfinite(volatilities).all():
            raise ValueError("volatilities must contain only finite values")
        if np.any(volatilities <= 0.0):
            raise ValueError("volatilities must be strictly positive")

        total_scenarios = int(self.total_scenarios)
        if total_scenarios <= 0:
            raise ValueError("total_scenarios must be positive")

        if self.seed is not None and not isinstance(
            self.seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer or None")

        if self.shock_dtype not in (np.float32, np.float64):
            raise TypeError("shock_dtype must be np.float32 or np.float64")

        if self.distribution_kind == "student_t":
            if self.student_t_df is None:
                raise ValueError("student_t_df is required for student_t")
            if not np.isfinite(self.student_t_df) or self.student_t_df <= 2.0:
                raise ValueError(
                    "student_t_df must be finite and greater than 2"
                )

        if self.distribution_kind == "variance_gamma":
            if self.vg_nu is None:
                raise ValueError("vg_nu is required for variance_gamma")
            if not np.isfinite(self.vg_nu) or self.vg_nu <= 0.0:
                raise ValueError("vg_nu must be finite and positive")

        if self.distribution_kind == "regime_switching_gmm":
            if self.prob_crisis is None:
                raise ValueError(
                    "prob_crisis is required for regime_switching_gmm"
                )
            if not np.isfinite(self.prob_crisis):
                raise ValueError("prob_crisis must be finite")
            if not 0.0 <= self.prob_crisis <= 1.0:
                raise ValueError("prob_crisis must be between 0 and 1")
            if self.crisis_vol_multipliers is None:
                raise ValueError(
                    "crisis_vol_multipliers is required for regime_switching_gmm"
                )

            if isinstance(
                self.crisis_vol_multipliers,
                (int, float, np.integer, np.floating),
            ):
                crisis_mult = float(self.crisis_vol_multipliers)
                if not np.isfinite(crisis_mult) or crisis_mult <= 0.0:
                    raise ValueError(
                        "crisis_vol_multipliers must be finite and positive"
                    )
                object.__setattr__(self, "crisis_vol_multipliers", crisis_mult)
            else:
                crisis_mult = np.asarray(
                    self.crisis_vol_multipliers, dtype=np.float64
                )
                if crisis_mult.ndim != 1:
                    raise ValueError(
                        "crisis_vol_multipliers must be scalar or 1D"
                    )
                if crisis_mult.shape[0] != cholesky_matrix.shape[0]:
                    raise ValueError(
                        "crisis_vol_multipliers length must match cholesky_matrix dimension"
                    )
                if not np.isfinite(crisis_mult).all() or np.any(
                    crisis_mult <= 0.0
                ):
                    raise ValueError(
                        "crisis_vol_multipliers must contain only positive finite values"
                    )
                object.__setattr__(self, "crisis_vol_multipliers", crisis_mult)

        object.__setattr__(self, "factor_ids", factor_ids)
        object.__setattr__(
            self, "cholesky_matrix", np.ascontiguousarray(cholesky_matrix)
        )
        object.__setattr__(self, "volatilities", volatilities)
        object.__setattr__(self, "total_scenarios", total_scenarios)
        object.__setattr__(
            self, "_cholesky_transpose", np.ascontiguousarray(cholesky_matrix.T)
        )
        object.__setattr__(self, "_volatilities64", volatilities)

    @classmethod
    def standard_normal(
        cls,
        factor_ids: np.ndarray,
        cholesky_matrix: np.ndarray,
        volatilities: np.ndarray,
        total_scenarios: int,
        seed: int | None = None,
        shock_dtype: type[np.float32] | type[np.float64] = np.float32,
    ) -> ParametricMCScenarioGenerator:
        return cls(
            factor_ids=factor_ids,
            cholesky_matrix=cholesky_matrix,
            volatilities=volatilities,
            total_scenarios=total_scenarios,
            distribution_kind="standard_normal",
            seed=seed,
            shock_dtype=shock_dtype,
        )

    @classmethod
    def student_t(
        cls,
        df: float,
        factor_ids: np.ndarray,
        cholesky_matrix: np.ndarray,
        volatilities: np.ndarray,
        total_scenarios: int,
        seed: int | None = None,
        shock_dtype: type[np.float32] | type[np.float64] = np.float32,
    ) -> ParametricMCScenarioGenerator:
        return cls(
            factor_ids=factor_ids,
            cholesky_matrix=cholesky_matrix,
            volatilities=volatilities,
            total_scenarios=total_scenarios,
            distribution_kind="student_t",
            seed=seed,
            student_t_df=float(df),
            shock_dtype=shock_dtype,
        )

    @classmethod
    def laplace(
        cls,
        factor_ids: np.ndarray,
        cholesky_matrix: np.ndarray,
        volatilities: np.ndarray,
        total_scenarios: int,
        seed: int | None = None,
        shock_dtype: type[np.float32] | type[np.float64] = np.float32,
    ) -> ParametricMCScenarioGenerator:
        return cls(
            factor_ids=factor_ids,
            cholesky_matrix=cholesky_matrix,
            volatilities=volatilities,
            total_scenarios=total_scenarios,
            distribution_kind="laplace",
            seed=seed,
            shock_dtype=shock_dtype,
        )

    @classmethod
    def variance_gamma(
        cls,
        vg_nu: float,
        factor_ids: np.ndarray,
        cholesky_matrix: np.ndarray,
        volatilities: np.ndarray,
        total_scenarios: int,
        seed: int | None = None,
        shock_dtype: type[np.float32] | type[np.float64] = np.float32,
    ) -> ParametricMCScenarioGenerator:
        return cls(
            factor_ids=factor_ids,
            cholesky_matrix=cholesky_matrix,
            volatilities=volatilities,
            total_scenarios=total_scenarios,
            distribution_kind="variance_gamma",
            seed=seed,
            vg_nu=float(vg_nu),
            shock_dtype=shock_dtype,
        )

    @classmethod
    def regime_switching_gmm(
        cls,
        prob_crisis: float,
        crisis_vol_multipliers: float | np.ndarray,
        factor_ids: np.ndarray,
        cholesky_matrix: np.ndarray,
        volatilities: np.ndarray,
        total_scenarios: int,
        seed: int | None = None,
        shock_dtype: type[np.float32] | type[np.float64] = np.float32,
    ) -> ParametricMCScenarioGenerator:
        return cls(
            factor_ids=factor_ids,
            cholesky_matrix=cholesky_matrix,
            volatilities=volatilities,
            total_scenarios=total_scenarios,
            distribution_kind="regime_switching_gmm",
            seed=seed,
            prob_crisis=float(prob_crisis),
            crisis_vol_multipliers=crisis_vol_multipliers,
            shock_dtype=shock_dtype,
        )

    def _apply_distribution_scaling(
        self,
        *,
        rng: np.random.Generator,
        shocks: np.ndarray,
        gmm_sliced_multipliers: float | np.ndarray | None,
    ) -> None:
        if self.distribution_kind == "standard_normal":
            return

        elif self.distribution_kind == "student_t":
            assert self.student_t_df is not None
            chi_square = rng.chisquare(
                df=self.student_t_df, size=shocks.shape[0]
            )
            row_scale = np.sqrt((self.student_t_df - 2.0) / chi_square)
            np.multiply(shocks, row_scale[:, None], out=shocks)

        elif self.distribution_kind == "laplace":
            exponential_mix = rng.exponential(scale=1.0, size=shocks.shape[0])
            row_scale = np.sqrt(exponential_mix)
            np.multiply(shocks, row_scale[:, None], out=shocks)

        elif self.distribution_kind == "variance_gamma":
            assert self.vg_nu is not None
            gamma_mix = rng.gamma(
                shape=1.0 / self.vg_nu, scale=self.vg_nu, size=shocks.shape[0]
            )
            row_scale = np.sqrt(gamma_mix)
            np.multiply(shocks, row_scale[:, None], out=shocks)

        elif self.distribution_kind == "regime_switching_gmm":
            assert self.prob_crisis is not None
            assert gmm_sliced_multipliers is not None
            crisis_mask = rng.random(shocks.shape[0]) < self.prob_crisis
            if np.any(crisis_mask):
                shocks[crisis_mask, :] *= gmm_sliced_multipliers

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

        shock_source_ids, _, req_to_uix = build_shock_key_plan(
            required_shift_type_id=required_shift_type_id,
            source_factor_ids=source_factor_ids,
        )

        K_total = int(self.cholesky_matrix.shape[0])
        K_req = int(shock_source_ids.shape[0])

        id_to_col = {int(fid): i for i, fid in enumerate(self.factor_ids)}
        col_indices = np.empty(K_req, dtype=np.int32)

        for i in range(K_req):
            fid = int(shock_source_ids[i])
            idx = id_to_col.get(fid)
            if idx is None:
                raise KeyError(
                    f"Required factor_id {fid} not found in parametric universe."
                )
            col_indices[i] = idx

        # --- Pre-slice the mathematical parameters ---

        # Slice cholesky
        L_T_sliced = np.ascontiguousarray(
            self._cholesky_transpose[:, col_indices]
        )

        # Slice vols
        vols_sliced = np.ascontiguousarray(self._volatilities64[col_indices])

        # Slice GMM multipliers
        gmm_mults_sliced = None
        if self.distribution_kind == "regime_switching_gmm":
            assert self.crisis_vol_multipliers is not None

            # Type-checker safe: Explicitly isolate the ndarray for slicing
            if isinstance(self.crisis_vol_multipliers, np.ndarray):
                gmm_mults_sliced = np.ascontiguousarray(
                    self.crisis_vol_multipliers[col_indices]
                )
            else:
                # Both floats and ints fall safely here
                gmm_mults_sliced = float(self.crisis_vol_multipliers)

        rng = np.random.default_rng(self.seed)

        # Raw normals MUST remain K_total wide to ensure correct cross-asset correlation
        work = np.empty((batch_size, K_total), dtype=np.float64, order="C")

        # Correlated output is now directly allocated at K_req width
        correlated = np.empty((batch_size, K_req), dtype=np.float64, order="C")

        scenario_offset = 0

        while scenario_offset < self.total_scenarios:
            scenario_count = min(
                batch_size, self.total_scenarios - scenario_offset
            )

            work_view = work[:scenario_count, :]
            correlated_view = correlated[:scenario_count, :]

            # 1. Draw raw (B, K_total)
            rng.standard_normal(size=work_view.shape, out=work_view)

            # 2. Optimized Matmul: (B, K_total) @ (K_total, K_req) -> (B, K_req)
            np.matmul(work_view, L_T_sliced, out=correlated_view)

            # 3. Apply distribution scaling and base volatilities to the smaller matrix
            self._apply_distribution_scaling(
                rng=rng,
                shocks=correlated_view,
                gmm_sliced_multipliers=gmm_mults_sliced,
            )
            np.multiply(correlated_view, vols_sliced, out=correlated_view)

            # 4. Final copy to Fortran contiguous target
            shocks = np.empty(
                (scenario_count, K_req), dtype=self.shock_dtype, order="F"
            )
            np.copyto(shocks, correlated_view, casting="same_kind")

            yield ScenarioSetBatch(
                shocks=shocks,
                required_to_shock_uix=req_to_uix,
                shock_scale=shock_scale,
                required_shift_type_id=required_shift_type_id,
                scenario_offset=scenario_offset,
                scenario_count_total=self.total_scenarios,
            )

            scenario_offset += scenario_count
