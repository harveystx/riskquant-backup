# src/riskquant/solutions/risk_run/schemas.py
"""
Contracts of the production scenario-VaR entry point.

`RiskRun` is the production result: the VaR/ES numbers and the immutable
identity of the computation that produced them, in ONE frozen object. They are
built together and serialized together, so a persisted risk number cannot be
detached from its identity.

Identity versus result integrity (D083). `identity` is `manifest.identity()`:
the identity of the CALCULATION DEFINITION (D040, D081 §7). It never commits
to an output. `result_digest` is the integrity of the RESULT ARTIFACT: it binds
the identity to the stored `position_uids`, `var`, `es` and `reproducible`, so
a persisted result cannot state numbers its identity did not carry at build
time. It is an unkeyed content digest: integrity and consistency, NOT
authentication. Numerical authority beyond self-consistency is reproduction
under a matching environment (D041/D042).

Methodology composition (D081 §11). `MethodologyDeclaration` partitions an
EXPLICIT production universe U of economic position UIDs into FULL_REVAL,
TAYLOR/SUPPLIED and TAYLOR/DERIVED. `ValuationContext` is the run-owned base
valuation context (P1-B): the run derives its as-of, static-configuration and
valuation-contract identity from it, and computes EVERY V_0 from it.
`TaylorBlock` carries one certified Taylor representation only. `ReportingSpec`
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

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.portfolio.engine import compile_portfolio_from_ingest_batch
from riskquant.portfolio.fx import validate_ccy
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.solutions.run_identity.hashing import digest_named_arrays
from riskquant.solutions.run_identity.schemas import RunManifest
from riskquant.solutions.sensitivity_pnl.api import SensitivityPnlCalculator
from riskquant.solutions.sensitivity_pnl.schemas import SensitivityPnlProvenance
from riskquant.universe import UixMapper

__all__ = [
    "CompositionContractError",
    "MethodologyDeclaration",
    "NonReproducibleRunError",
    "ReportingSpec",
    "ReproducibilityMode",
    "ResultIntegrityError",
    "RiskRun",
    "RunFactorAxis",
    "TaylorBlock",
    "ValuationContext",
]

#: `reproducible` (default): a stochastic model must carry an explicit seed, or
#: the run is refused before any numerical work. `exploratory`: `seed=None` is
#: permitted and the result is marked non-reproducible.
ReproducibilityMode = Literal["reproducible", "exploratory"]


class NonReproducibleRunError(ValueError):
    """A reproducible run was requested for a configuration that cannot be
    reproduced (a stochastic model with `seed=None`)."""


class ResultIntegrityError(ValueError):
    """A persisted `RiskRun` fails RESULT INTEGRITY: its `result_digest` is
    absent or does not bind its identity to its stored position UIDs, VaR, ES
    and reproducibility flag (D083)."""


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
class ValuationContext:
    """
    The run-owned base valuation context (P1-B). Required for every run.

    The run owns its identity through this object: the as-of is the mapper's
    (D067), the discounting mode is stated here (D080), the universe map is
    BUILT by the run from `static_config`, and every V_0, FULL_REVAL and
    TAYLOR/DERIVED alike, is valued by the run from `portfolio` at the base
    levels. Nothing on this object is echoed back as a claimed digest.

    Provenance (N1): the caller supplies the canonical ingest output
    (`IngestBatch`, from `compile_raw_df_into_ingest_batch`), never a compiled
    portfolio. `portfolio` is COMPILED HERE, with this context's own mapper, by
    the canonical compiler, so every valuation row's parent economic UID comes
    from the ingest batch by construction.
    """

    static_config: RiskStaticConfig
    ingest_batch: IngestBatch
    uix_mapper: UixMapper
    discounting_mode: DiscountingMode
    portfolio: CompiledPortfolio = field(init=False)

    def __post_init__(self) -> None:
        if self.uix_mapper.as_of_ord32 is None:
            raise CompositionContractError(
                "the UixMapper carries no as-of; the run values at the mapper's "
                "as-of and cannot infer one (D067)."
            )
        if not isinstance(self.discounting_mode, DiscountingMode):
            raise TypeError("discounting_mode must be a DiscountingMode.")
        object.__setattr__(
            self,
            "portfolio",
            compile_portfolio_from_ingest_batch(
                ingest_batch=self.ingest_batch, uix_mapper=self.uix_mapper
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class TaylorBlock:
    """
    One certified Taylor representation entering production P&L.

    It carries NO base value and NO expected binding (P1-B): the run values
    V_0 from its own `ValuationContext` and requires the calculator's certified
    binding to equal the binding it derives itself (as-of, base levels,
    discounting mode, static configuration, valuation contract, V_0).
    """

    calculator: SensitivityPnlCalculator


@dataclass(frozen=True, slots=True, kw_only=True)
class ReportingSpec:
    """
    The caller-owned reporting choice: the single reporting currency (D085).

    Everything else the D081 conversion needs is derived by the run, never
    stated here: each position's settlement currency from its compiled
    economic state, and its settlement -> reporting FX factor, orientation and
    run-axis row from the static configuration's FX authority (D060).
    """

    reporting_ccy: str

    def __post_init__(self) -> None:
        validate_ccy(self.reporting_ccy)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class RunFactorAxis:
    """
    Canonical descriptor of the run axis of one production run (D082 as
    amended, D085): which factor sits on which row, and nothing else.

    `factor_ids[:num_valuation_factors]` is the valuation prefix, the universe
    map's `required_factor_ids` row for row (also the Taylor axis). The
    remaining rows are the reporting-only FX suffix: the reporting FX factors
    the valuation prefix lacks, ascending by rf_id. `shift_type_ids` is
    row-aligned. Every axis-aligned run input (factor ids, shift types, source
    factor ids, shock scale, base levels) is bound to exactly this order.

    Ownership: frozen fields; the arrays are private C-contiguous copies of
    exact dtype (int32 / int8), 1-D and READ-ONLY. The descriptor is never
    execution storage: `run_scenario_var_es` takes its own writable execution
    copies of whatever arrays it receives, so these may be passed to it as-is.
    """

    factor_ids: npt.NDArray[np.int32]
    shift_type_ids: npt.NDArray[np.int8]
    num_valuation_factors: int

    def __post_init__(self) -> None:
        for name, dtype in (("factor_ids", np.int32), ("shift_type_ids", np.int8)):
            value = getattr(self, name)
            if (
                not isinstance(value, np.ndarray)
                or value.ndim != 1
                or value.dtype != dtype
            ):
                raise TypeError(f"{name} must be a 1-D {np.dtype(dtype).name} array.")
            owned = np.array(value, dtype=dtype, copy=True, order="C")
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)
        k = self.factor_ids.shape[0]
        n_val = self.num_valuation_factors
        if isinstance(n_val, bool) or not isinstance(n_val, int):
            raise TypeError("num_valuation_factors must be an int.")
        if self.shift_type_ids.shape[0] != k:
            raise ValueError("shift_type_ids must be row-aligned with factor_ids.")
        if not 0 <= self.num_valuation_factors <= k:
            raise ValueError("num_valuation_factors must lie in [0, K].")
        if np.unique(self.factor_ids).shape[0] != k:
            raise ValueError("the run axis must not repeat a factor id.")
        suffix = self.factor_ids[self.num_valuation_factors :]
        if suffix.size > 1 and not bool((suffix[1:] > suffix[:-1]).all()):
            raise ValueError("the reporting suffix must be ascending by rf_id.")

    @property
    def reporting_factor_ids(self) -> npt.NDArray[np.int32]:
        """The reporting-only FX suffix (possibly empty)."""
        return self.factor_ids[self.num_valuation_factors :]


#: Keys of the persisted `RiskRun` JSON. A payload missing one is malformed and
#: fails as such, never as an incidental `KeyError`.
_RISK_RUN_JSON_KEYS: tuple[str, ...] = (
    "es_hex",
    "identity",
    "manifest",
    "pnl_provenance",
    "position_uids",
    "reporting_ccy",
    "reproducible",
    "result_digest",
    "var_hex",
)


def _owned_result_array(
    value: npt.NDArray[np.generic], *, name: str, dtype: type[np.generic]
) -> npt.NDArray[np.generic]:
    """
    Take ownership of one result array: exact dtype, 1-D, C-ordered, read-only.

    The dtype is REQUIRED, never cast: `result_digest` folds dtype and bytes in,
    so a coerced array would describe a different buffer than the one handed
    over. The copy severs every alias, so a later write to the caller's buffer
    cannot make the stored numbers disagree with their digest.
    """
    if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype != dtype:
        raise TypeError(f"{name} must be a 1-D {np.dtype(dtype).name} array.")
    owned = np.array(value, dtype=dtype, copy=True, order="C")
    owned.setflags(write=False)
    return owned


def _result_digest(
    *,
    identity: str,
    position_uids: npt.NDArray[np.int64],
    var: npt.NDArray[np.float64],
    es: npt.NDArray[np.float64],
    reproducible: bool,
) -> str:
    """
    Content digest of one result artifact, bound to its calculation identity.

    `digest_named_arrays` folds name, dtype, shape and exact bytes of every
    member in, so a one-ULP change, a permuted or replaced UID and a flipped
    flag each change it. `identity` is a member, so a result cannot be
    transplanted onto another calculation, and a relabelled manifest with a
    recomputed identity no longer matches the stored digest.
    """
    return digest_named_arrays(
        identity=np.frombuffer(identity.encode("ascii"), dtype=np.uint8),
        position_uids=position_uids,
        var=var,
        es=es,
        reproducible=np.asarray([reproducible], dtype=np.bool_),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RiskRun:
    """
    Production result: risk numbers inseparable from their run identity.

    `position_uids` is the production universe U, ascending: one entry per
    economic position, whatever its methodology or its number of valuation
    rows. `var` / `es` hold one entry per position, in `position_uids` order,
    followed by the portfolio aggregate as the LAST entry; every entry is
    finite. `identity` is `manifest.identity()`, computed once and stored so
    the two cannot drift. `reproducible` is False only for an exploratory run
    of a stochastic model with `seed=None`. `pnl_provenance` stamps the
    TAYLOR/DERIVED block; it is None when the run has no Taylor block.

    `result_digest` is DERIVED here from `identity`, `position_uids`, `var`,
    `es` and `reproducible` (D083). It is the persisted-result integrity
    check, not a second calculation identity: it never enters the manifest.
    The stored arrays are owned read-only copies, so the digest keeps
    describing them.
    """

    position_uids: npt.NDArray[np.int64]
    var: npt.NDArray[np.float64]
    es: npt.NDArray[np.float64]
    reporting_ccy: str
    manifest: RunManifest
    identity: str
    reproducible: bool
    pnl_provenance: SensitivityPnlProvenance | None
    result_digest: str = field(init=False)

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
        if not isinstance(self.reproducible, bool):
            raise TypeError("reproducible must be a bool.")
        for name, dtype in (
            ("position_uids", np.int64),
            ("var", np.float64),
            ("es", np.float64),
        ):
            object.__setattr__(
                self,
                name,
                _owned_result_array(getattr(self, name), name=name, dtype=dtype),
            )
        n = self.position_uids.shape[0] + 1
        if self.var.shape != (n,) or self.es.shape != (n,):
            raise ValueError("var/es must have one entry per position plus portfolio.")
        # A non-finite risk number is an error state, never a result: a digest
        # over it would certify an internally consistent but invalid artifact.
        for name in ("var", "es"):
            finite = np.isfinite(getattr(self, name))
            if not bool(finite.all()):
                bad = np.flatnonzero(~finite)
                raise ValueError(
                    f"{name} holds {bad.size} non-finite entr"
                    f"{'y' if bad.size == 1 else 'ies'} (first 10 indices: "
                    f"{bad[:10].tolist()}); a RiskRun carries finite risk numbers only."
                )
        object.__setattr__(
            self,
            "result_digest",
            _result_digest(
                identity=self.identity,
                position_uids=self.position_uids,
                var=self.var,
                es=self.es,
                reproducible=self.reproducible,
            ),
        )

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
        is self-identifying; `result_digest` travels inside, so the stored
        result is integrity-checked on load (D083).
        """
        return json.dumps(
            {
                "position_uids": [int(u) for u in self.position_uids],
                "var_hex": [float(v).hex() for v in self.var],
                "es_hex": [float(v).hex() for v in self.es],
                "reporting_ccy": self.reporting_ccy,
                "identity": self.identity,
                "reproducible": self.reproducible,
                "result_digest": self.result_digest,
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
        """
        Load and VERIFY a persisted artifact.

        Order: required keys, identity against the manifest, the `RiskRun`
        construction contract (shape, dtype, finiteness, currency), then
        RESULT INTEGRITY: the digest recomputed from the parsed result must
        equal the persisted `result_digest`. Any failure raises; a payload is
        never repaired.
        """
        d = json.loads(payload)
        if not isinstance(d, dict):
            raise ValueError("malformed persisted RiskRun: not a JSON object.")
        missing = [k for k in _RISK_RUN_JSON_KEYS if k not in d]
        if "result_digest" in missing:
            raise ResultIntegrityError(
                "RESULT INTEGRITY: the persisted RiskRun carries no result_digest; "
                "its stored result cannot be verified against its identity."
            )
        if missing:
            raise ValueError(f"malformed persisted RiskRun: missing key(s) {missing}.")
        manifest = RunManifest.model_validate(d["manifest"])
        if d["identity"] != manifest.identity():
            raise ValueError("persisted identity does not match its manifest.")
        if not isinstance(d["reproducible"], bool):
            raise ValueError("malformed persisted RiskRun: reproducible is not a bool.")
        provenance = d["pnl_provenance"]
        run = cls(
            position_uids=np.asarray(d["position_uids"], dtype=np.int64),
            var=np.asarray([float.fromhex(h) for h in d["var_hex"]], dtype=np.float64),
            es=np.asarray([float.fromhex(h) for h in d["es_hex"]], dtype=np.float64),
            reporting_ccy=d["reporting_ccy"],
            manifest=manifest,
            identity=d["identity"],
            reproducible=d["reproducible"],
            pnl_provenance=None
            if provenance is None
            else SensitivityPnlProvenance.model_validate(provenance),
        )
        if run.result_digest != d["result_digest"]:
            raise ResultIntegrityError(
                "RESULT INTEGRITY: the persisted result_digest does not match the "
                "digest of the stored identity, position_uids, var, es and "
                "reproducible flag; the stored result is not the result this "
                "artifact was built with."
            )
        return run

