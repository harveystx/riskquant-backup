# src/riskquant/solutions/risk_run/schemas.py
"""
Contracts of the production scenario-VaR entry point.

`RiskRun` is the production result: the VaR/ES numbers and the immutable
identity of the computation that produced them, in ONE frozen object. They are
built together and serialized together, so a persisted risk number cannot be
detached from its identity.

Methodology composition (D081 §11). `MethodologyDeclaration` partitions an
EXPLICIT production universe U of economic position UIDs into FULL_REVAL,
TAYLOR/SUPPLIED and TAYLOR/DERIVED. `FullRevalBlock` carries the compiled
valuation representation the FULL route prices; `TaylorBlock` carries one
certified Taylor representation and the V_0 it is bound to. `ReportingSpec`
carries reporting metadata only. Each fact has exactly one owner here.

Solution tier only. Nothing here enters a generator, a kernel or a Numba
signature (D001).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from riskquant.market_data.assemble import UniverseMapping
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.solutions.run_identity.hashing import digest_named_arrays
from riskquant.solutions.run_identity.schemas import RunManifest
from riskquant.solutions.sensitivity_pnl.api import SensitivityPnlCalculator
from riskquant.solutions.sensitivity_pnl.schemas import SensitivityPnlProvenance
from riskquant.solutions.sensitivity_pnl.taylor_contract import TaylorBaseBinding
from riskquant.universe import UixMapper

__all__ = [
    "CompositionContractError",
    "FullRevalBlock",
    "MethodologyDeclaration",
    "NonReproducibleRunError",
    "ReportingSpec",
    "ReproducibilityMode",
    "RiskRun",
    "TaylorBlock",
]

#: `reproducible` (default): a stochastic model must carry an explicit seed, or
#: the run is refused before any numerical work. `exploratory`: `seed=None` is
#: permitted and the result is marked non-reproducible.
ReproducibilityMode = Literal["reproducible", "exploratory"]


class NonReproducibleRunError(ValueError):
    """A reproducible run was requested for a configuration that cannot be
    reproduced (a stochastic model with `seed=None`)."""


class CompositionContractError(ValueError):
    """The methodology composition violates D081 §11: the declared partition,
    the supplied blocks, the factor axis or the economic UID universe disagree."""


def _owned_uid_array(value: object, *, name: str) -> npt.NDArray[np.int64]:
    """
    Take ownership of one UID set: defensive int64 copy, ascending, unique,
    read-only.

    The copy severs every alias to the caller's buffer, so a later write to
    the caller's array cannot change the declaration or its digest. A float
    UID is refused rather than cast: the cast would truncate it silently.
    """
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise CompositionContractError(f"{name} must be 1-D; got ndim={raw.ndim}.")
    if raw.size == 0:
        owned = np.empty(0, dtype=np.int64)
    else:
        if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.integer):
            raise CompositionContractError(
                f"{name} must hold integer UIDs; got dtype {raw.dtype}."
            )
        if raw.dtype.kind == "u" and int(raw.max()) > np.iinfo(np.int64).max:
            raise CompositionContractError(f"{name} holds a UID above the int64 range.")
        owned = np.sort(raw.astype(np.int64, copy=True), kind="stable")
    repeated = np.unique(owned[1:][owned[1:] == owned[:-1]])
    if repeated.size:
        raise CompositionContractError(
            f"{name} declares {repeated.size} UID(s) more than once. "
            f"First 10: {repeated[:10].tolist()}."
        )
    owned = np.ascontiguousarray(owned)
    owned.setflags(write=False)
    return owned


_UID_FIELDS: tuple[str, ...] = (
    "production_position_uids",
    "full_reval_uids",
    "taylor_supplied_uids",
    "taylor_derived_uids",
)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class MethodologyDeclaration:
    """
    D081 §11 partition of the production universe.

    `production_position_uids` is U, the authoritative set of ECONOMIC position
    UIDs of the run. It is stated explicitly and is never computed as the union
    of the partitions, so the partition is checked against U, not against
    itself. The run then reconciles U against every artifact that carries
    position identity (the compiled portfolio, each Taylor book).

    Construction proves U_F ⊔ U_S ⊔ U_D = U: no duplicate, no overlap, no
    undeclared UID, no extra UID, no empty U. There is no default methodology.
    Every stored array is an owned, ascending, read-only int64 copy.
    """

    production_position_uids: npt.NDArray[np.int64]
    full_reval_uids: npt.NDArray[np.int64]
    taylor_supplied_uids: npt.NDArray[np.int64]
    taylor_derived_uids: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        for name in _UID_FIELDS:
            object.__setattr__(
                self, name, _owned_uid_array(getattr(self, name), name=name)
            )
        universe = self.production_position_uids
        if universe.size == 0:
            raise CompositionContractError(
                "production_position_uids is empty: a production run needs at "
                "least one economic position."
            )
        partitions = {
            "FULL_REVAL": self.full_reval_uids,
            "TAYLOR/SUPPLIED": self.taylor_supplied_uids,
            "TAYLOR/DERIVED": self.taylor_derived_uids,
        }
        labels = tuple(partitions)
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                both = np.intersect1d(partitions[a], partitions[b], assume_unique=True)
                if both.size:
                    raise CompositionContractError(
                        f"{both.size} UID(s) are declared both {a} and {b}; a "
                        f"position has exactly one methodology. First 10: "
                        f"{both[:10].tolist()}."
                    )
        declared = np.concatenate(tuple(partitions.values()))
        undeclared = np.setdiff1d(universe, declared, assume_unique=True)
        extra = np.setdiff1d(declared, universe, assume_unique=True)
        if undeclared.size or extra.size:
            raise CompositionContractError(
                "the methodology partition does not cover the production universe "
                f"exactly: {undeclared.size} UID(s) of U have no methodology "
                f"(first 10: {undeclared[:10].tolist()}) and {extra.size} declared "
                f"UID(s) are not in U (first 10: {extra[:10].tolist()}). There is "
                "no default methodology."
            )

    def blocks(
        self,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """`(U_F, U_S, U_D)`, ascending and read-only: the PnlGrid block order."""
        return (
            self.full_reval_uids,
            self.taylor_supplied_uids,
            self.taylor_derived_uids,
        )

    def digest(self) -> str:
        """Order-canonical content digest of U and the partition."""
        return digest_named_arrays(**{
            name: getattr(self, name) for name in _UID_FIELDS
        })


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class FullRevalBlock:
    """
    The compiled valuation representation the FULL_REVAL route prices.

    The as-of is the mapper's (D067) and the discounting mode is the universe
    map's (D080); the run reads both from here and takes no second copy.
    """

    portfolio: CompiledPortfolio
    uix_mapper: UixMapper
    universe_map: UniverseMapping

    def __post_init__(self) -> None:
        if self.portfolio.uix_fingerprint != self.uix_mapper.fingerprint:
            raise CompositionContractError(
                "the compiled portfolio was stamped by a different UixMapper "
                f"(portfolio {self.portfolio.uix_fingerprint[:12]}..., mapper "
                f"{self.uix_mapper.fingerprint[:12]}...); its uixs index foreign "
                "universes (AUD-02, D064)."
            )
        if self.uix_mapper.as_of_ord32 is None:
            raise CompositionContractError(
                "the UixMapper carries no as-of; the FULL route values at the "
                "mapper's as-of and cannot infer one (D067)."
            )


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class TaylorBlock:
    """
    One Taylor representation entering production P&L.

    `base_settlement_values` is V_0 in settlement currency, in
    `calculator.position_uids` order (ascending). `expected_binding` is the
    base the caller states the representation is certified at; the run
    requires the calculator's certified binding to equal it (P1-A) and
    re-derives the base-levels and V_0 digests from its own inputs.
    """

    calculator: SensitivityPnlCalculator
    base_settlement_values: npt.NDArray[np.float64]
    expected_binding: TaylorBaseBinding

    def __post_init__(self) -> None:
        v0 = self.base_settlement_values
        if not isinstance(v0, np.ndarray) or v0.ndim != 1 or v0.dtype != np.float64:
            raise TypeError("base_settlement_values must be a 1-D float64 array.")
        num_positions = int(self.calculator.num_positions)
        if v0.shape[0] != num_positions:
            raise ValueError(
                f"base_settlement_values has {v0.shape[0]} value(s) but the book "
                f"has {num_positions} position(s)."
            )
        if not np.isfinite(v0).all():
            raise ValueError("base_settlement_values must be finite.")
        owned = np.array(v0, dtype=np.float64, copy=True, order="C")
        owned.setflags(write=False)
        object.__setattr__(self, "base_settlement_values", owned)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReportingSpec:
    """
    Reporting-currency conversion inputs, on the production economic axis.

    `position_uids` is the alignment key: it must equal the run's production
    universe U in canonical ascending order, and `settlement_ccys[i]` is the
    settlement currency of `position_uids[i]`. `fx_factor_rows[j]` is the row
    of the required-factor axis carrying the FX rate named `fx_factor_names[j]`;
    the quotation direction is derived from the name by the FX authority,
    never stated here. Base values and Taylor bindings are NOT reporting
    metadata; they live on the methodology blocks.
    """

    reporting_ccy: str
    position_uids: npt.NDArray[np.int64]
    settlement_ccys: tuple[str, ...]
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
        uids = self.position_uids
        if not isinstance(uids, np.ndarray) or uids.ndim != 1 or uids.dtype != np.int64:
            raise TypeError("position_uids must be a 1-D int64 array.")
        if uids.shape[0] != len(self.settlement_ccys):
            raise ValueError(
                f"position_uids has {uids.shape[0]} UID(s) but "
                f"{len(self.settlement_ccys)} settlement_ccys."
            )
        owned = np.array(uids, dtype=np.int64, copy=True, order="C")
        owned.setflags(write=False)
        object.__setattr__(self, "position_uids", owned)


@dataclass(frozen=True, slots=True, kw_only=True)
class RiskRun:
    """
    Production result: risk numbers inseparable from their run identity.

    `position_uids` is the production universe U, ascending: one entry per
    economic position, whatever its methodology or its number of valuation
    rows. `var` / `es` hold one entry per position, in `position_uids` order,
    followed by the portfolio aggregate as the LAST entry. `identity` is
    `manifest.identity()`, computed once and stored so the two cannot drift.
    `reproducible` is False only for an exploratory run of a stochastic model
    with `seed=None`. `pnl_provenance` stamps the TAYLOR/DERIVED block; it is
    None when the run has no Taylor block.
    """

    position_uids: npt.NDArray[np.int64]
    var: npt.NDArray[np.float64]
    es: npt.NDArray[np.float64]
    reporting_ccy: str
    manifest: RunManifest
    identity: str
    reproducible: bool
    pnl_provenance: SensitivityPnlProvenance | None

    def __post_init__(self) -> None:
        if self.identity != self.manifest.identity():
            raise ValueError("identity does not match manifest.")
        # A production result must name its methodology composition (D081 §11).
        if self.manifest.pnl.composition is None:
            raise ValueError(
                "manifest carries no methodology composition; a production "
                "RiskRun cannot be detached from its U_F/U_S/U_D partition."
            )
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
                "pnl_provenance": None
                if self.pnl_provenance is None
                else self.pnl_provenance.model_dump(mode="json"),
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
        provenance = d["pnl_provenance"]
        return cls(
            position_uids=np.asarray(d["position_uids"], dtype=np.int64),
            var=np.asarray([float.fromhex(h) for h in d["var_hex"]], dtype=np.float64),
            es=np.asarray([float.fromhex(h) for h in d["es_hex"]], dtype=np.float64),
            reporting_ccy=d["reporting_ccy"],
            manifest=manifest,
            identity=d["identity"],
            reproducible=bool(d["reproducible"]),
            pnl_provenance=None
            if provenance is None
            else SensitivityPnlProvenance.model_validate(provenance),
        )
