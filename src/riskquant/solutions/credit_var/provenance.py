# src/riskquant/solutions/credit_var/provenance.py
"""
Build identity, request identity and numerical environment of a credit-VaR run.

D-C15 makes provenance part of the release contract: a stored result must name
the scenario construction and the numerical environment that produced it, so
that a result of the pre-D-C15 generator cannot pass for a result of the
current one. This module owns every value of `RunProvenance` that does not come
from the request itself.

Contents
--------
SCENARIO_CONSTRUCTION_ID
    Name of the stochastic construction of `MixedCreditFactorGenerator`.
SCENARIO_CONSTRUCTION_SOURCE_SHA256
    SHA-256 of the LF-normalized text of the generator source. It lives beside
    the identifier so that a change of the generator source fails a test
    (WS_CREDIT_VAR_API C.12). The owner then keeps the identifier and records
    the new digest, or sets a new identifier. Nothing here reads the digest at
    run time; the guard is a test, not a run-time check.
NUMPY_VERSION, SCIPY_VERSION, NUMBA_VERSION
    The numerical environment of D-C15 and D041. NumPy and SciPy define the
    scenario stream; Numba defines the codegen of the JTD and tail kernels.
engine_version
    ``"<V>+src.<H>"`` (P-CREDIT_VAR_API-001 clause 5), computed once per
    process. There is no fallback string: an engine that cannot state its
    build raises `CreditVarEngineError`.
request_input_hash
    SHA-256 of the D040 canonical bytes of a validated request.
"""

from __future__ import annotations

__all__ = [
    "NUMBA_VERSION",
    "NUMPY_VERSION",
    "SCENARIO_CONSTRUCTION_ID",
    "SCENARIO_CONSTRUCTION_SOURCE",
    "SCENARIO_CONSTRUCTION_SOURCE_SHA256",
    "SCIPY_VERSION",
    "ScenarioConstructionId",
    "compute_engine_version",
    "engine_version",
    "request_input_hash",
    "source_tree_digest",
]

import hashlib
import json
import threading
from importlib import metadata
from pathlib import Path
from typing import Final, Literal

import numba
import numpy as np
import scipy
from pydantic import BaseModel

import riskquant
from riskquant.solutions.credit_var.errors import CreditVarEngineError

# ---------------------------------------------------------------------------
# Scenario construction (D-C15)
# ---------------------------------------------------------------------------

ScenarioConstructionId = Literal["MixedCreditFactorGenerator/D-C15"]

SCENARIO_CONSTRUCTION_ID: Final[ScenarioConstructionId] = (
    "MixedCreditFactorGenerator/D-C15"
)

#: Generator source, relative to the directory of the imported riskquant package.
SCENARIO_CONSTRUCTION_SOURCE: Final[str] = "credit/factors/simulate/mixed.py"

#: SHA-256 of `SCENARIO_CONSTRUCTION_SOURCE` with CRLF replaced by LF. A new
#: digest here without a new identifier states that the construction is
#: unchanged (for example a docstring edit); that is an owner decision.
SCENARIO_CONSTRUCTION_SOURCE_SHA256: Final[str] = (
    "7d8709bd761bbd5e1acbf9d81c932ca6b458dd61774af20d53296f15b6d84905"
)

# ---------------------------------------------------------------------------
# Numerical environment (D-C15, D041)
# ---------------------------------------------------------------------------

NUMPY_VERSION: Final[str] = str(np.__version__)
SCIPY_VERSION: Final[str] = str(scipy.__version__)
NUMBA_VERSION: Final[str] = str(numba.__version__)

# ---------------------------------------------------------------------------
# Build identity (P-CREDIT_VAR_API-001 clause 5)
# ---------------------------------------------------------------------------

_DISTRIBUTION: Final[str] = "riskquant"

# The package metadata version is static ("0.1.0" for every build), so it
# cannot tell two builds apart (F-9). The source digest can. One lock and one
# slot: the first successful computation is the value for the life of the
# process, and a failed computation stores nothing.
_engine_version_lock: Final[threading.Lock] = threading.Lock()
_engine_version_value: str | None = None


def source_tree_digest(root: Path) -> str:
    """SHA-256 over every ``*.py`` file below `root`.

    Files are taken in ascending order of their POSIX path relative to `root`
    (Python ``str`` order, i.e. Unicode code point order). Each file adds its
    UTF-8 relative path, one NUL byte, its bytes with CRLF replaced by LF, and
    one NUL byte. The LF normalization makes the digest independent of the
    git line-ending setting of the checkout. ``__pycache__`` directories and
    every non-``.py`` file (``.pyc``, Numba ``.nbi`` / ``.nbc`` caches) are
    outside the digest: they are build products, not source.

    Raises
    ------
    OSError
        A source file cannot be read.
    """
    entries = sorted(
        (path.relative_to(root).as_posix(), path)
        for path in root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.relative_to(root).parts
    )
    digest = hashlib.sha256()
    for relative, path in entries:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def compute_engine_version() -> str:
    """Compute ``"<V>+src.<H>"`` without the per-process cache.

    V is the distribution version from the installed package metadata. H is
    `source_tree_digest` of the directory of the imported ``riskquant``
    package; a sibling package such as ``riskquant_lab`` is outside it.

    Raises
    ------
    CreditVarEngineError
        The distribution metadata is missing, V carries a PEP 440 local segment
        (``+...``, which would make the result ambiguous), the package has no
        file location, or a source file cannot be read.
    """
    try:
        version = metadata.version(_DISTRIBUTION)
    except metadata.PackageNotFoundError as exc:
        raise CreditVarEngineError(
            f"engine_version: no installed metadata for distribution "
            f"'{_DISTRIBUTION}'. Install the package (for example 'pip install "
            "-e .'); a result without a build identity is not produced."
        ) from exc
    if not version or "+" in version:
        raise CreditVarEngineError(
            f"engine_version: distribution version {version!r} is empty or has a "
            "local segment ('+...'); the build identity appends its own local "
            "segment 'src.<sha256>' and needs a plain release version."
        )
    package_file = riskquant.__file__
    if package_file is None:
        raise CreditVarEngineError(
            "engine_version: the imported 'riskquant' package has no file location; "
            "its source tree cannot be digested."
        )
    try:
        tree = source_tree_digest(Path(package_file).parent)
    except OSError as exc:
        raise CreditVarEngineError(
            f"engine_version: a source file of the riskquant package cannot be "
            f"read: {exc}"
        ) from exc
    return f"{version}+src.{tree}"


def engine_version() -> str:
    """The build identity, computed at the first call and cached for the process.

    Raises
    ------
    CreditVarEngineError
        As `compute_engine_version`. A failure is not cached.
    """
    global _engine_version_value
    with _engine_version_lock:
        if _engine_version_value is None:
            _engine_version_value = compute_engine_version()
        return _engine_version_value


# ---------------------------------------------------------------------------
# Request identity (D040 canonical form, A_CONTRACT section 4.2)
# ---------------------------------------------------------------------------


def _canonicalize(value: object) -> object:
    """D040 canonical form: the rule of `run_identity.schemas._canonicalize`.

    Copied, not imported: that function is private to `run_identity`. A float
    becomes ``{"__f64__": float.hex()}``, so a one-ulp change and the sign of
    zero each change the bytes; a decimal repr would hide neither reliably.
    A tuple becomes a list, so the container type cannot alter the bytes. The
    ``bool`` test comes first because ``bool`` subclasses ``int``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return {"__f64__": value.hex()}
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value


def request_input_hash(request: BaseModel) -> str:
    """Lowercase SHA-256 hex (64 characters) of the canonical request bytes.

    Two requests that validate to field-wise bit-equal models give one hash:
    each float equal under `float.hex`, each int, bool, str and dict key equal
    under ``==``. Different hashes do not prove different results (D040
    asymmetry); equal hashes identify equal inputs.
    """
    payload = json.dumps(
        _canonicalize(request.model_dump(mode="python")),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
