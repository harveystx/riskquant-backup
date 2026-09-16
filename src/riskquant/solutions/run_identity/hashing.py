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

import hashlib

import numpy as np
import numpy.typing as npt

__all__ = [
    "digest_array",
    "digest_named_arrays",
    "digest_reporting_plan",
    "hash_compiled_book",
    "hash_parametric_risk_data",
    "hash_risk_factor_matrix",
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
    """Identity of the parametric construction data (K-coupled triple)."""
    return digest_named_arrays(
        factor_ids=data.factor_ids,
        cholesky_matrix=data.cholesky_matrix,
        volatilities=data.volatilities,
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
