# src/riskquant/credit/masters/factor_universe/compiled.py
"""
Compiled credit-factor universe data.
"""

from __future__ import annotations

__all__ = [
    "CompiledCreditFactorUniverse",
]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from riskquant.credit.masters.enums.credit_factor_kind import CreditFactorKind


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledCreditFactorUniverse:
    """
    Canonical credit-factor universe.

    UIX is implicit:

        factor_ids[i]
        factor_names[i]
        factor_kind[i]
        factor_chi_nu[i]
        factor_beta_alpha[i]
        factor_beta_beta[i]

    all describe factor UIX=i.

    Example
    -------
    The universe is stored in structure-of-arrays form. Conceptually it is
    equivalent to the following table:

    +-----+-----------+-------------------+------------------+--------+-------+------+
    | UIX | Factor ID | Factor Name       | Factor Kind      | Nu     | Alpha | Beta |
    +=====+===========+===================+==================+========+=======+======+
    | 0   | 1001      | GLOBAL_CREDIT     | SYSTEMIC         |   -    |   -   |   -  |
    +-----+-----------+-------------------+------------------+--------+-------+------+
    | 1   | 1002      | IDIO_OBLIGOR_001  | IDIOSYNCRATIC    |   -    |   -   |   -  |
    +-----+-----------+-------------------+------------------+--------+-------+------+
    | 2   | 1003      | CHI_GLOBAL        | CHI              |  4.0   |   -   |   -  |
    +-----+-----------+-------------------+------------------+--------+-------+------+
    | 3   | 1004      | U_OBLIGOR_001     | UNIFORM          |   -    |   -   |   -  |
    +-----+-----------+-------------------+------------------+--------+-------+------+
    | 4   | 1005      | RECESSION_FLAG    | STRESS_INDICATOR |   -    |   -   |   -  |
    +-----+-----------+-------------------+------------------+--------+-------+------+
    | 5   | 1006      | PD_ENVIRONMENT    | BETA_PD          |   -    | 2.0   | 18.0 |
    +-----+-----------+-------------------+------------------+--------+-------+------+

    which is represented internally as:

        factor_ids
            = [1001, 1002, 1003, 1004, 1005, 1006]

        factor_names
            = [
                "GLOBAL_CREDIT",
                "IDIO_OBLIGOR_001",
                "CHI_GLOBAL",
                "U_OBLIGOR_001",
                "RECESSION_FLAG",
                "BETA_PD_ENVIRONMENT",
            ]

        factor_kind
            = [
                SYSTEMIC,
                IDIOSYNCRATIC,
                CHI,
                UNIFORM,
                STRESS_INDICATOR,
                BETA_PD,
            ]

    UIX is therefore simply the row coordinate into the factor matrix:

        credit_factor_matrix[0, :]  -> GLOBAL_CREDIT
        credit_factor_matrix[2, :]  -> CHI_GLOBAL
        credit_factor_matrix[5, :]  -> BETA_PD_ENVIRONMENT

    `factor_ids` is the stable identity axis (sorted ascending, unique); UIX is
    the engine-owned coordinate — the row index into the `(F, S)`
    `CreditFactorMatrix`.

    `factor_chi_nu` carries nu > 0 for CHI factors and the `NO_CHI_NU`
    sentinel (NOT NaN) elsewhere.

    `factor_beta_alpha` and `factor_beta_beta` carry the Beta(alpha, beta)
    shape parameters (both > 0) for BETA_PD factors and the
    `NO_BETA_PARAM` sentinel elsewhere.

    The derived accessors are plain ``property``: a slotted dataclass has no
    ``__dict__`` for ``functools.cached_property`` to write into, and these are
    setup-time projections (generator config, ingest maps), never hot-path.
    """

    factor_ids: npt.NDArray[np.int32]
    factor_names: npt.NDArray[np.str_]
    factor_kind: npt.NDArray[np.int8]
    factor_chi_nu: npt.NDArray[np.float64]
    factor_beta_alpha: npt.NDArray[np.float64]
    factor_beta_beta: npt.NDArray[np.float64]

    @property
    def n_factors(self) -> int:
        return int(self.factor_ids.size)

    @property
    def factor_id_to_uix(self) -> dict[int, int]:
        return {int(factor_id): uix for uix, factor_id in enumerate(self.factor_ids)}

    @property
    def factor_name_to_uix(self) -> dict[str, int]:
        return {str(name): uix for uix, name in enumerate(self.factor_names)}

    @property
    def chi_uix(self) -> npt.NDArray[np.int32]:
        return np.flatnonzero(self.factor_kind == CreditFactorKind.CHI).astype(np.int32)

    @property
    def systemic_uix(self) -> npt.NDArray[np.int32]:
        return np.flatnonzero(self.factor_kind == CreditFactorKind.SYSTEMIC).astype(
            np.int32
        )

    @property
    def idiosyncratic_uix(self) -> npt.NDArray[np.int32]:
        return np.flatnonzero(
            self.factor_kind == CreditFactorKind.IDIOSYNCRATIC
        ).astype(np.int32)

    @property
    def uniform_uix(self) -> npt.NDArray[np.int32]:
        return np.flatnonzero(self.factor_kind == CreditFactorKind.UNIFORM).astype(
            np.int32
        )

    @property
    def stress_indicator_uix(self) -> npt.NDArray[np.int32]:
        return np.flatnonzero(
            self.factor_kind == CreditFactorKind.STRESS_INDICATOR
        ).astype(np.int32)

    @property
    def chi_nu_by_uix(self) -> dict[int, float]:
        return {int(uix): float(self.factor_chi_nu[uix]) for uix in self.chi_uix}

    @property
    def chi_nu_vector(self) -> npt.NDArray[np.float64]:
        """
        nu values aligned to chi_uix.
        """
        return np.ascontiguousarray(
            self.factor_chi_nu[self.chi_uix],
            dtype=np.float64,
        )

    @property
    def beta_pd_uix(self) -> npt.NDArray[np.int32]:
        return np.flatnonzero(self.factor_kind == CreditFactorKind.BETA_PD).astype(
            np.int32
        )

    @property
    def beta_alpha(self) -> npt.NDArray[np.float64]:
        """alpha values aligned to beta_pd_uix."""
        return np.ascontiguousarray(
            self.factor_beta_alpha[self.beta_pd_uix],
            dtype=np.float64,
        )

    @property
    def beta_beta(self) -> npt.NDArray[np.float64]:
        """beta values aligned to beta_pd_uix."""
        return np.ascontiguousarray(
            self.factor_beta_beta[self.beta_pd_uix],
            dtype=np.float64,
        )

    @property
    def beta_params_by_uix(self) -> dict[int, tuple[float, float]]:
        return {
            int(uix): (
                float(self.factor_beta_alpha[uix]),
                float(self.factor_beta_beta[uix]),
            )
            for uix in self.beta_pd_uix
        }

    def get_uix_from_factor_id(self, factor_id: int) -> int:
        return self.factor_id_to_uix[factor_id]

    def get_uix_from_factor_name(self, factor_name: str) -> int:
        return self.factor_name_to_uix[factor_name]
