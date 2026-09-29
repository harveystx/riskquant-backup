# src/riskquant/solutions/run_identity/hashing.py
"""
Content-addressed identity for the numerical inputs of a risk run.

Solution tier. Nothing here enters a generator, a kernel, or a Numba signature:
these helpers consume typed NumPy arrays and emit hex digests, and the digests
travel only inside the manifest (D001).

Identity is CONTENT identity, never object identity: two arrays with equal
dtype, shape and bytes hash equally regardless of their memory address, their
base object, or whether one is a copy. Layout is canonicalised (C order) before
hashing so an F-ordered and a C-ordered array holding the same logical matrix
share an identity, while a transpose (different logical matrix) does not.

Floats are hashed as their canonical IEEE-754 binary representation, never as
decimal text and never rounded: values one ULP apart have different identities,
because they can produce different results.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from riskquant.config.proxy_model.compiled import CompiledProxyModel

__all__ = [
    "digest_array",
    "digest_named_arrays",
    "digest_reporting_plan",
    "hash_compiled_book",
    "hash_compiled_portfolio",
    "hash_parametric_risk_data",
    "hash_proxy_model",
    "hash_risk_factor_matrix",
    "hash_universe_mapping",
]

# Cryptographic digest from the standard library; no new dependency. Collision
# resistance is assumed at the SHA-256 contract level, never proven here.
_HASH = "sha256"
_ENC = "utf-8"


def _canonical_bytes(array: npt.NDArray) -> bytes:
    """C-ordered, native-endianness-normalised raw buffer."""
    a = np.ascontiguousarray(array)
    if a.dtype.byteorder not in ("=", "|"):
        a = a.astype(a.dtype.newbyteorder("="), copy=False)
    return a.tobytes(order="C")


def digest_array(array: npt.NDArray) -> str:
    """
    Hex digest of one array's full numerical interpretation.

    dtype and shape are folded in, so the same bytes reinterpreted under a
    different dtype or reshaped to different dimensions do NOT collide.
    """
    h = hashlib.new(_HASH)
    h.update(str(array.dtype.str).encode(_ENC))
    h.update(b"|")
    h.update(str(array.shape).encode(_ENC))
    h.update(b"|")
    h.update(_canonical_bytes(array))
    return h.hexdigest()


def digest_named_arrays(**arrays: npt.NDArray) -> str:
    """
    Hex digest of an ordered, NAMED bundle of arrays.

    Names are folded in and the bundle is hashed in the caller's declaration
    order, so a matrix's (factor_ids, dates, values) triple cannot be confused
    with any other arrangement of the same buffers.
    """
    h = hashlib.new(_HASH)
    for name, array in arrays.items():
        h.update(name.encode(_ENC))
        h.update(b"=")
        h.update(digest_array(array).encode(_ENC))
        h.update(b";")
    return h.hexdigest()


def hash_risk_factor_matrix(matrix) -> str:
    """
    Identity of a historical scenario input.

    Hashing `values` alone is insufficient: the same buffer under a different
    factor-id vector, or a different column order, is a different numerical
    interpretation and a different risk result. factor_ids are hashed AS AN
    ORDERED VECTOR (never as a set) because column order is authoritative.
    """
    return digest_named_arrays(
        factor_ids=matrix.factor_ids, dates=matrix.dates, values=matrix.values
    )


def hash_parametric_risk_data(data) -> str:
    """
    Identity of the parametric construction data: the K-coupled arrays and the
    effective forecast period they represent (D090), and the driver chart of
    each column (D091). The period is hashed as int64 count and int8 unit code
    and the charts as the int8 ``chart_ids`` array, so the same numbers declared
    for a different period or a different chart are a different artifact.
    """
    return digest_named_arrays(
        factor_ids=data.factor_ids,
        cholesky_matrix=data.cholesky_matrix,
        volatilities=data.volatilities,
        effective_period_count=np.array([data.effective_period_count], dtype=np.int64),
        effective_period_unit=np.array([data.effective_period_unit], dtype=np.int8),
        chart_ids=data.chart_ids,
    )


def hash_compiled_book(book) -> str:
    """
    Identity of a compiled sensitivity book — the CSR arrays the Taylor kernel
    actually consumes, so the digest names exactly what priced the run rather
    than a source frame that might have been re-projected differently.
    """
    return digest_named_arrays(
        position_uids=book.position_uids,
        active_factor_ids=book.active_factor_ids,
        active_level_uix=book.active_level_uix,
        displacement_convention_ids=book.displacement_convention_ids,
        position_offsets=book.position_offsets,
        factor_uix=book.factor_uix,
        first_order=book.first_order,
        second_order=book.second_order,
        cross_offsets=book.cross_offsets,
        cross_factor_uix_i=book.cross_factor_uix_i,
        cross_factor_uix_j=book.cross_factor_uix_j,
        cross_second_order=book.cross_second_order,
    )


def digest_reporting_plan(
    *,
    reporting_ccy: str,
    settlement_ccys,
    fx_factor_names,
    fx_factor_rows: npt.NDArray,
) -> str:
    """
    Identity of the reporting-currency conversion inputs.

    The FX scenario levels themselves are NOT re-hashed: they are rows of the
    shocked levels, already fixed by the data + shock identities. What remains
    result-defining is which positions settle in which currency, which factor
    rows carry the FX rates, and their names (the quotation direction is derived
    from the names by the FX authority).
    """
    h = hashlib.new(_HASH)
    h.update(reporting_ccy.encode(_ENC))
    h.update(b"|")
    h.update(",".join(str(c) for c in settlement_ccys).encode(_ENC))
    h.update(b"|")
    h.update(",".join(str(n) for n in fx_factor_names).encode(_ENC))
    h.update(b"|")
    h.update(digest_array(np.asarray(fx_factor_rows, dtype=np.int32)).encode(_ENC))
    return h.hexdigest()


# Sentinel for an absent slice bound. Distinct from every real bound, so
# slice(None, 4) and slice(0, 4) never digest alike.
_NO_SLICE_BOUND = np.iinfo(np.int64).min


def _field_as_array(name: str, value: object) -> npt.NDArray:
    """
    Canonical array form of one dataclass field for content hashing.

    Arrays pass through `digest_array` unchanged (dtype and shape folded in).
    Slices, integer tuples, enums, strings and scalars get one fixed array
    encoding each. Any other type raises: an identity that silently skipped a
    field could not distinguish two objects that differ only in that field.
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, slice):
        return np.asarray(
            [
                _NO_SLICE_BOUND if b is None else int(b)
                for b in (value.start, value.stop, value.step)
            ],
            dtype=np.int64,
        )
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, str):
        return np.frombuffer(value.encode(_ENC), dtype=np.uint8)
    if isinstance(value, (bool, np.bool_)):
        return np.asarray([bool(value)], dtype=np.bool_)
    if isinstance(value, (int, np.integer)):
        return np.asarray([int(value)], dtype=np.int64)
    if isinstance(value, (float, np.floating)):
        return np.asarray([float(value)], dtype=np.float64)
    if isinstance(value, tuple) and all(
        isinstance(v, (int, np.integer)) for v in value
    ):
        return np.asarray([int(v) for v in value], dtype=np.int64)
    raise TypeError(
        f"field {name!r} of type {type(value).__name__} has no canonical identity "
        "encoding; refusing to hash an object while ignoring one of its fields."
    )


def _dataclass_arrays(obj: Any, *, prefix: str) -> dict[str, npt.NDArray]:
    """Every dataclass field of `obj`, in declaration order, as named arrays."""
    return {
        f"{prefix}{f.name}": _field_as_array(f.name, getattr(obj, f.name))
        for f in dataclasses.fields(obj)
    }


def hash_compiled_portfolio(portfolio) -> str:
    """
    Content digest of a `CompiledPortfolio`: the compiled valuation
    representation the FULL_REVAL route prices.

    Every block contributes every field (parent `trade_uids` and all SoA
    arrays) in block order, tagged by block type, so a changed quantity, a
    changed row order, a moved row or a re-stamped mapper all change it.
    """
    arrays: dict[str, npt.NDArray] = {
        "unique_dates_ord32": portfolio.unique_dates_ord32,
        "uix_fingerprint": _field_as_array(
            "uix_fingerprint", portfolio.uix_fingerprint
        ),
    }
    for index, block in enumerate(portfolio.blocks):
        prefix = f"block{index}:{type(block).__name__}:{int(block.BLOCK_TYPE_ID)}:"
        arrays.update(_dataclass_arrays(block, prefix=prefix))
    return digest_named_arrays(**arrays)


def hash_universe_mapping(universe_map) -> str:
    """
    Content digest of a `UniverseMapping`: the ordered required-factor axis,
    its shift types, the discounting mode and the full layout plan that maps
    flat level rows onto market-data slots.
    """
    return digest_named_arrays(**_dataclass_arrays(universe_map, prefix=""))


def hash_proxy_model(model: CompiledProxyModel) -> str:
    """
    Identity of the compiled, run-bound Proxy V1 operator (D088 §7).

    Hashes the canonical CSR arrays the run will evaluate, in this fixed
    declaration order: row targets, row offsets, term drivers, term driver
    charts, term loadings. The digest is therefore over the canonical resolved
    operator after totality closure, never over authoring syntax: equivalent
    authoring forms that canonicalise to identical binary content share an
    identity, and any change of canonical content (one ULP of a merged loading,
    a chart, a driver, a target, a row) changes it. The target chart kappa_A is
    not part of this digest; it enters the run identity through the static
    configuration and universe-map digests.
    """
    return digest_named_arrays(
        row_target_rf_ids=model.row_target_rf_ids,
        row_term_offsets=model.row_term_offsets,
        term_driver_rf_ids=model.term_driver_rf_ids,
        term_chart_ids=model.term_chart_ids,
        term_loadings=model.term_loadings,
    )
