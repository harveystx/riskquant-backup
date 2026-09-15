# src/riskquant/solutions/run_identity/schemas.py
"""
Immutable identity of a risk run: the minimum set of inputs that, held fixed,
determine the final VaR/ES bit-for-bit in a pinned numerical environment.

Solution tier, Pydantic v2, `extra="forbid"`. The manifest is never passed into
a generator, a kernel or a Numba signature; it is built ALONGSIDE the run from
the same inputs and carries only primitives and hex digests (D001).

Identity versus record
----------------------
`RunManifest` is the mathematical identity: every field in it can change the
number. Operational metadata (timestamp, operator, host, job id) is deliberately
absent, so re-running the same computation on another day or host yields the
same identity.

Snapshots are IDENTIFIED, not stored: a digest names the data, it does not
contain it. A manifest stays small regardless of matrix size.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "DataIdentity",
    "EngineIdentity",
    "ExecutionIdentity",
    "PnlIdentity",
    "ReportingIdentity",
    "RiskMeasureIdentity",
    "RunManifest",
    "ShockIdentity",
]

# Schema version of THIS contract. Deliberately distinct from the engine
# version: the manifest layout and the software evolve independently.
MANIFEST_SCHEMA_VERSION = "1"

_STRICT = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class EngineIdentity(BaseModel):
    """Numerical environment. Explains whether a rerun can be bitwise identical.
    NumPy gives no cross-version guarantee for `Generator` distributions, so a
    differing `numpy` here downgrades the reproducibility claim."""

    model_config = _STRICT
    riskquant_version: str
    python_version: str
    numpy_version: str


class DataIdentity(BaseModel):
    """Content digest of the construction data. `kind` distinguishes a historical
    matrix from parametric data so two digests can never be confused."""

    model_config = _STRICT
    kind: Literal["risk_factor_matrix", "parametric_risk_data"]
    digest: str


class ShockIdentity(BaseModel):
    """The scenario-to-factor mapping and shock conventions. These are arguments
    to `iter_batches`, NOT part of the scenario config, and each of them changes
    the result: proven by mutation (shift type, source factor, shock scale).
    Hashed as ordered vectors because position binds a required factor to its
    source factor."""

    model_config = _STRICT
    required_shift_type_ids_digest: str
    source_factor_ids_digest: str
    shock_scale_digest: str


class PnlIdentity(BaseModel):
    """Inputs to the P&L step downstream of the shocks."""

    model_config = _STRICT
    base_levels_digest: str
    sensitivities_digest: str


class ReportingIdentity(BaseModel):
    """Reporting-currency conversion. Scenario FX enters the P&L, so the FX
    inputs are identified by digest rather than restated."""

    model_config = _STRICT
    reporting_currency: str
    fx_digest: str


class RiskMeasureIdentity(BaseModel):
    """VaR/ES semantics. `confidence_level` lives outside the scenario config and
    changes both VaR and ES (proven by mutation)."""

    model_config = _STRICT
    confidence_level: float


class ExecutionIdentity(BaseModel):
    """Execution parameters that can perturb the final float reduction.

    `batch_size` does NOT change the scenario stream (D029, proven bitwise), but
    a downstream BLAS reduction over a batch-shaped array can differ in its last
    ulp with the batch width. Recording it makes the bitwise claim honest."""

    model_config = _STRICT
    batch_size: int = Field(gt=0)


class RunManifest(BaseModel):
    """
    Complete numerical identity of a risk run.

    `identity()` is a SHA-256 over a canonical serialization: keys sorted, no
    incidental whitespace, and every float written as its exact IEEE-754 hex
    representation, so the digest cannot drift with dict construction order or
    decimal formatting.
    """

    model_config = _STRICT

    manifest_schema_version: Literal["1"] = MANIFEST_SCHEMA_VERSION
    engine: EngineIdentity
    scenario_config: dict[str, Any]
    data: DataIdentity
    shocks: ShockIdentity
    pnl: PnlIdentity
    reporting: ReportingIdentity
    risk_measure: RiskMeasureIdentity
    execution: ExecutionIdentity

    def canonical_bytes(self) -> bytes:
        """
        Byte-stable IDENTITY form. Same semantic manifest -> same bytes.

        Not the persistence format: floats are rewritten as exact IEEE-754 hex
        so the digest cannot drift with decimal formatting. Use `to_json` to
        store a manifest and `identity` to address it.
        """
        return json.dumps(
            _canonicalize(self.model_dump(mode="python")),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def identity(self) -> str:
        """Content address of the run."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def to_json(self) -> str:
        """
        Persistence form: ordinary JSON, keys sorted, no incidental whitespace.

        Python's float repr round-trips exactly, so a manifest reloaded with
        `from_json` reproduces the same `identity()`.
        """
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> RunManifest:
        """Load and VALIDATE. A manifest is a production contract, not free JSON:
        unknown fields, missing sections and unknown schema versions are
        rejected."""
        return cls.model_validate(json.loads(payload))


def _canonicalize(value: object) -> object:
    """Normalise for byte-stable serialization.

    Floats become exact hex (`float.hex()`): decimal repr is lossy-looking and
    platform-sensitive, whereas hex is the exact binary value, so two floats one
    ulp apart serialize differently and never round together. Tuples become
    lists so a tuple/list distinction cannot alter the bytes.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return {"__f64__": value.hex()}
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    return value
