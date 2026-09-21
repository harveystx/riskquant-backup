# src/riskquant/solutions/risk_run/schemas.py
"""
Contracts of the production scenario-VaR entry point.

`RiskRun` is the production result: the VaR/ES numbers and the immutable
identity of the computation that produced them, in ONE frozen object. They are
built together and serialized together, so a persisted risk number cannot be
detached from its identity.

Solution tier only. Nothing here enters a generator, a kernel or a Numba
signature (D001).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from riskquant.solutions.run_identity.schemas import RunManifest
from riskquant.solutions.sensitivity_pnl.schemas import SensitivityPnlProvenance
from riskquant.solutions.sensitivity_pnl.taylor_contract import TaylorBaseBinding

__all__ = [
    "NonReproducibleRunError",
    "ReportingSpec",
    "ReproducibilityMode",
    "RiskRun",
]

#: `reproducible` (default): a stochastic model must carry an explicit seed, or
#: the run is refused before any numerical work. `exploratory`: `seed=None` is
#: permitted and the result is marked non-reproducible.
ReproducibilityMode = Literal["reproducible", "exploratory"]


class NonReproducibleRunError(ValueError):
    """A reproducible run was requested for a configuration that cannot be
    reproduced (a stochastic model with `seed=None`)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ReportingSpec:
    """
    Reporting-currency conversion inputs, on the book's position order.

    `settlement_ccys[i]` is the settlement currency of `book.position_uids[i]`.
    `fx_factor_rows[j]` is the row of the required-factor axis carrying the FX
    rate named `fx_factor_names[j]`; the quotation direction is derived from the
    name by the FX authority, never stated here. An identity conversion is a
    position whose settlement currency equals `reporting_ccy`.
    """

    reporting_ccy: str
    settlement_ccys: tuple[str, ...]
    #: D081: each position's BASE value in its own settlement currency (V_0),
    #: in `book.position_uids` order. Required — ordinary-MTM reporting P&L is
    #: dV*q_s + V_0*(q_s - q_0), and the translation term needs V_0. No default:
    #: a Taylor-only book that cannot state its base values cannot report in
    #: a foreign currency.
    base_settlement_values: npt.NDArray[np.float64]
    #: P1-A: the base the V_0 above (and the book's Greeks) were produced at.
    #: The run requires it to equal the calculator's certified binding, and
    #: re-derives the levels and V_0 digests from the run's own inputs.
    base_binding: TaylorBaseBinding
    fx_factor_names: tuple[str, ...] = ()
    fx_factor_rows: npt.NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )

    def __post_init__(self) -> None:
        rows = self.fx_factor_rows
        if rows.ndim != 1 or rows.dtype != np.int32:
            raise TypeError("fx_factor_rows must be 1-D int32.")
        if rows.shape[0] != len(self.fx_factor_names):
            raise ValueError(
                f"fx_factor_rows has {rows.shape[0]} rows but "
                f"{len(self.fx_factor_names)} fx_factor_names."
            )
        if len(self.settlement_ccys) == 0:
            raise ValueError("settlement_ccys must be non-empty.")
        v0 = self.base_settlement_values
        if v0.ndim != 1 or v0.dtype != np.float64:
            raise TypeError("base_settlement_values must be 1-D float64.")
        if v0.shape[0] != len(self.settlement_ccys):
            raise ValueError(
                f"base_settlement_values has {v0.shape[0]} value(s) but "
                f"{len(self.settlement_ccys)} settlement_ccys."
            )
        if not np.isfinite(v0).all():
            raise ValueError("base_settlement_values must be finite.")


@dataclass(frozen=True, slots=True, kw_only=True)
class RiskRun:
    """
    Production result: risk numbers inseparable from their run identity.

    `var` / `es` hold one entry per position, in `position_uids` order, followed
    by the portfolio aggregate as the LAST entry. `identity` is
    `manifest.identity()`, computed once and stored so the two cannot drift.
    `reproducible` is False only for an exploratory run of a stochastic model
    with `seed=None`.
    """

    position_uids: npt.NDArray[np.int64]
    var: npt.NDArray[np.float64]
    es: npt.NDArray[np.float64]
    reporting_ccy: str
    manifest: RunManifest
    identity: str
    reproducible: bool
    pnl_provenance: SensitivityPnlProvenance

    def __post_init__(self) -> None:
        if self.identity != self.manifest.identity():
            raise ValueError("identity does not match manifest.")
        # A persisted artifact must not be able to state a currency its own
        # identity contradicts: the manifest is the authority.
        if self.reporting_ccy != self.manifest.reporting.reporting_currency:
            raise ValueError(
                f"reporting_ccy {self.reporting_ccy!r} contradicts the manifest "
                f"({self.manifest.reporting.reporting_currency!r})."
            )
        n = self.position_uids.shape[0] + 1
        if self.var.shape != (n,) or self.es.shape != (n,):
            raise ValueError("var/es must have one entry per position plus portfolio.")

    @property
    def portfolio_var(self) -> float:
        return float(self.var[-1])

    @property
    def portfolio_es(self) -> float:
        return float(self.es[-1])

    def to_json(self) -> str:
        """
        Serializable artifact. Risk numbers are written as exact IEEE-754 hex so
        the round trip is bitwise; the manifest travels inside, so the artifact
        is self-identifying.
        """
        return json.dumps(
            {
                "position_uids": [int(u) for u in self.position_uids],
                "var_hex": [float(v).hex() for v in self.var],
                "es_hex": [float(v).hex() for v in self.es],
                "reporting_ccy": self.reporting_ccy,
                "identity": self.identity,
                "reproducible": self.reproducible,
                "manifest": json.loads(self.manifest.to_json()),
                "pnl_provenance": self.pnl_provenance.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str) -> RiskRun:
        d = json.loads(payload)
        manifest = RunManifest.model_validate(d["manifest"])
        if d["identity"] != manifest.identity():
            raise ValueError("persisted identity does not match its manifest.")
        return cls(
            position_uids=np.asarray(d["position_uids"], dtype=np.int64),
            var=np.asarray([float.fromhex(h) for h in d["var_hex"]], dtype=np.float64),
            es=np.asarray([float.fromhex(h) for h in d["es_hex"]], dtype=np.float64),
            reporting_ccy=d["reporting_ccy"],
            manifest=manifest,
            identity=d["identity"],
            reproducible=bool(d["reproducible"]),
            pnl_provenance=SensitivityPnlProvenance.model_validate(d["pnl_provenance"]),
        )
