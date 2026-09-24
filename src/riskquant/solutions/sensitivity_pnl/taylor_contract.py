# src/riskquant/solutions/sensitivity_pnl/taylor_contract.py
"""
Taylor representation contract (D081 §4-§10, P1-A).

A compiled sensitivity book is a bag of numbers. Whether those numbers are a
TRUSTWORTHY settlement-currency Taylor representation is a separate question
with five parts, each of which can be silently wrong while every shape and
dtype is right:

1. **Order.** `FIRST_ORDER`, `SECOND_ORDER_TRUNCATED` (diagonal only, a
   declared truncation) or `SECOND_ORDER_FULL` (diagonal + every cross pair).
2. **Dependency universe F(p).** The canonical axis factors the position's
   valuation function mathematically depends on at the bound base. It is
   DECLARED data, never inferred from which Greeks happen to be non-zero:
   ``V = x*y`` at ``y0 = 0`` has ``V_x = 0`` yet ``x in F(p)``.
3. **Completeness.** Every derivative the declared order requires over
   F(p) is present as an explicit value; an explicit ``0.0`` is a KNOWN zero,
   an absent entry is MISSING and fails. Sparse storage alone never proves
   completeness -- the declared F(p) and order do.
4. **Base binding.** The exact valuation point the Greeks and ``V_0`` were
   produced at: ``(as_of, base-levels digest, discounting_mode, static-config
   content digest, valuation-contract fingerprint, V_0 digest)``. Greeks from
   base A applied at base B fail; ``V_0`` from another base fails; ``q_0`` is
   read from the run's base levels, so it is covered by the levels digest.
5. **Explicit zero exposure.** A declared position with ``F(p) = {}`` must be
   listed as such. "No rows" alone is ambiguous between "zero exposure" and
   "Greeks never supplied" and is rejected.

This module validates the contract at the boundary; the compiler and kernels
remain unaware of it (kernels do dumb math). An UNCERTIFIED book (no
declaration) still compiles for engine-level use, but the production entry
point `solutions.risk_run.api.run_scenario_var_es` refuses it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import hashlib
import inspect
import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.solutions.run_identity.hashing import digest_named_arrays

_HASH = "sha256"


class TaylorOrder(enum.StrEnum):
    FIRST_ORDER = "FIRST_ORDER"
    SECOND_ORDER_TRUNCATED = "SECOND_ORDER_TRUNCATED"
    SECOND_ORDER_FULL = "SECOND_ORDER_FULL"


class TaylorContractError(ValueError):
    """A Taylor representation violates its declared contract."""


# =============================================================================
# Base binding
# =============================================================================
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TaylorBaseBinding:
    """
    The exact valuation point a Taylor representation describes.

    Every field is a content identity, not a label: two bindings are equal only
    if the Greeks and ``V_0`` they certify were produced at the same economic
    base under the same valuation semantics.
    """

    as_of_ord32: int
    base_levels_digest: str
    discounting_mode: str
    static_config_digest: str
    valuation_contract_fingerprint: str
    base_settlement_values_digest: str


def base_levels_digest(
    *, factor_ids: npt.NDArray[np.int32], base_levels: npt.NDArray[np.float64]
) -> str:
    """
    Digest of the base levels KEYED BY FACTOR IDENTITY.

    Canonicalised by ascending factor id, so reordering the axis while keeping
    each level with its factor gives the same digest, whereas reordering the
    levels without their ids (a positional misalignment) does not.
    """
    ids = np.asarray(factor_ids, dtype=np.int32)
    lv = np.asarray(base_levels, dtype=np.float64)
    if ids.ndim != 1 or lv.ndim != 1 or ids.shape[0] != lv.shape[0]:
        raise TaylorContractError(
            f"base levels {lv.shape} do not align with factor ids {ids.shape}."
        )
    if np.unique(ids).shape[0] != ids.shape[0]:
        raise TaylorContractError("base-level factor ids must be unique.")
    order = np.argsort(ids, kind="stable")
    return digest_named_arrays(
        factor_ids=np.ascontiguousarray(ids[order]),
        base_levels=np.ascontiguousarray(lv[order]),
    )


def base_settlement_values_digest(
    *, position_uids: Sequence[int] | npt.NDArray[Any], values: npt.NDArray[np.float64]
) -> str:
    """Digest of ``V_0`` keyed by position uid (order-canonical)."""
    uids = np.asarray(position_uids, dtype=np.int64)
    v0 = np.asarray(values, dtype=np.float64)
    if uids.ndim != 1 or v0.shape != uids.shape:
        raise TaylorContractError(
            f"V_0 {v0.shape} does not align with position uids {uids.shape}."
        )
    if not np.isfinite(v0).all():
        raise TaylorContractError("V_0 must be finite.")
    order = np.argsort(uids, kind="stable")
    return digest_named_arrays(
        position_uids=np.ascontiguousarray(uids[order]),
        base_settlement_values=np.ascontiguousarray(v0[order]),
    )


def static_config_content_digest(static_config: Any) -> str:
    """
    Semantic content digest of a compiled static configuration.

    Walks dataclasses (by field name), mappings (by sorted key), sequences (in
    order), numpy arrays (dtype, shape, bytes), enums (value), dates and
    primitives. Anything else is REJECTED rather than hashed by ``repr``: an
    incidental serialisation is not an identity.
    """
    h = hashlib.new(_HASH)
    _feed(h, static_config)
    return h.hexdigest()


def _feed(h: Any, obj: Any) -> None:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        h.update(b"D" + type(obj).__qualname__.encode())
        for f in sorted(dataclasses.fields(obj), key=lambda f: f.name):
            h.update(b"." + f.name.encode())
            _feed(h, getattr(obj, f.name))
        return
    if isinstance(obj, Mapping):
        h.update(b"M%d" % len(obj))
        for key in sorted(obj, key=repr):
            _feed(h, key)
            _feed(h, obj[key])
        return
    if isinstance(obj, np.ndarray):
        h.update(b"A" + obj.dtype.str.encode() + str(obj.shape).encode())
        h.update(np.ascontiguousarray(obj).tobytes())
        return
    if isinstance(obj, enum.Enum):
        h.update(b"E" + type(obj).__qualname__.encode())
        _feed(h, obj.value)
        return
    if isinstance(obj, bool | int | str | type(None)):
        h.update(b"P" + repr(obj).encode())
        return
    if isinstance(obj, float | np.floating):
        h.update(b"F" + float(obj).hex().encode())
        return
    if isinstance(obj, np.integer):
        h.update(b"P" + repr(int(obj)).encode())
        return
    if isinstance(obj, dt.date):
        h.update(b"T" + obj.isoformat().encode())
        return
    if isinstance(obj, frozenset | set):
        h.update(b"S%d" % len(obj))
        for item in sorted(obj, key=repr):
            _feed(h, item)
        return
    if isinstance(obj, list | tuple):
        h.update(b"L%d" % len(obj))
        for item in obj:
            _feed(h, item)
        return
    if isinstance(obj, pl.DataFrame):
        h.update(b"PL" + repr(obj.schema).encode())
        for name in obj.columns:
            _feed(h, obj[name].to_list())
        return
    raise TaylorContractError(
        f"static configuration contains an unhashable {type(obj).__qualname__}; "
        "extend the canonical digest explicitly rather than hashing its repr."
    )


def valuation_contract_fingerprint() -> str:
    """
    Fingerprint of the valuation SEMANTICS the Greeks and V_0 depend on.

    Folds in the convention constants (ACT/365F, ACT/365.25 TTE, TTE floor),
    the pricer and sensitizer registries by qualified name, and the source of
    the valuation, sensitivity and curve packages. Any change to a pricer,
    sensitizer, CDF choice or interpolation rule changes the fingerprint; the
    source hash is deliberately conservative (it errs toward invalidation).
    """
    from riskquant import constants
    from riskquant.sensitivities.sensitizer_registry import BLOCK_SENSITIZERS
    from riskquant.valuation.pricer_registry import BLOCK_PRICERS

    h = hashlib.new(_HASH)
    for name in ("DAYS_PER_YR_ACT_365_25", "DAYS_PER_YR_ACT_365F", "TTE_FLOOR"):
        h.update(name.encode() + b"=")
        _feed(h, getattr(constants, name))
    for label, registry in (
        ("pricers", BLOCK_PRICERS),
        ("sensitizers", BLOCK_SENSITIZERS),
    ):
        h.update(label.encode())
        for entry in registry:
            h.update(b"|" + _qualname(entry).encode())
    import riskquant.market_data.curves as curves_pkg
    import riskquant.sensitivities as sens_pkg
    import riskquant.valuation as val_pkg

    for pkg in (val_pkg, sens_pkg, curves_pkg):
        root = Path(inspect.getfile(pkg)).parent
        for path in sorted(root.rglob("*.py")):
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def _qualname(obj: Any) -> str:
    if obj is None:
        return "None"
    target = getattr(obj, "__wrapped__", obj)
    module = getattr(target, "__module__", type(target).__module__)
    name = getattr(target, "__qualname__", type(target).__qualname__)
    return f"{module}.{name}"


def verify_base_binding(
    *, representation: TaylorBaseBinding, expected: TaylorBaseBinding
) -> None:
    """Fail closed, naming every field that differs."""
    diffs = [
        f.name
        for f in dataclasses.fields(TaylorBaseBinding)
        if getattr(representation, f.name) != getattr(expected, f.name)
    ]
    if diffs:
        raise TaylorContractError(
            "Taylor representation is bound to a different base: mismatched "
            f"{diffs}. Greeks and V_0 are valid only at the exact base they "
            "were produced at (D081 §10)."
        )


# =============================================================================
# Declaration and completeness
# =============================================================================
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class TaylorDeclaration:
    """
    The contract a sensitivity book claims to satisfy.

    `dependencies` is F(p): one row per ``(position_id, factor_id)`` the
    position's valuation depends on. `zero_exposure_position_uids` lists the
    positions with ``F(p) = {}`` explicitly.
    """

    order: TaylorOrder
    dependencies: pl.DataFrame
    zero_exposure_position_uids: tuple[int, ...]
    base_binding: TaylorBaseBinding


def validate_taylor_completeness(
    *,
    declaration: TaylorDeclaration,
    sensitivities: pl.DataFrame,
    cross_sensitivities: pl.DataFrame | None,
    position_uids: Sequence[int] | npt.NDArray[Any] | None,
) -> None:
    """
    Prove the book is COMPLETE_FOR_STRUCTURE over the declared F(p).

    Fails closed on: an undeclared row set; a position with no F(p) that is not
    declared zero-exposure; a required derivative that is absent (missing is
    never zero); a derivative outside F(p) (wrong model); second-order content
    under FIRST_ORDER; any cross entry under FIRST_ORDER or
    SECOND_ORDER_TRUNCATED (the truncation is a declaration, not an accident);
    a missing, duplicated or doubly-oriented cross pair under SECOND_ORDER_FULL.
    """
    if not isinstance(declaration.order, TaylorOrder):
        raise TaylorContractError(f"undeclared Taylor order: {declaration.order!r}.")
    if position_uids is None:
        raise TaylorContractError(
            "a certified Taylor book must declare its position row set."
        )
    rows = {int(u) for u in np.asarray(position_uids).tolist()}
    deps = declaration.dependencies
    if set(deps.columns) != {"position_id", "factor_id"}:
        raise TaylorContractError(
            "F(p) must be a (position_id, factor_id) table; got columns "
            f"{sorted(deps.columns)}."
        )
    dep_keys = [
        (int(p), int(f)) for p, f in deps.select("position_id", "factor_id").iter_rows()
    ]
    if len(dep_keys) != len(set(dep_keys)):
        raise TaylorContractError("F(p) contains duplicate (position, factor) rows.")
    f_of: dict[int, set[int]] = {}
    for p, f in dep_keys:
        f_of.setdefault(p, set()).add(f)
    zero = {int(u) for u in declaration.zero_exposure_position_uids}
    if zero & set(f_of):
        raise TaylorContractError(
            f"positions {sorted(zero & set(f_of))} are declared zero-exposure but "
            "have a non-empty F(p)."
        )
    unresolved = rows - set(f_of) - zero
    extra = (set(f_of) | zero) - rows
    if unresolved:
        raise TaylorContractError(
            f"positions {sorted(unresolved)} have neither a declared F(p) nor an "
            "explicit zero-exposure declaration; absence is not zero exposure."
        )
    if extra:
        raise TaylorContractError(
            f"F(p) declares positions {sorted(extra)} outside the book's row set."
        )

    sens_keys = [
        (int(p), int(f))
        for p, f in sensitivities.select("position_id", "factor_id").iter_rows()
    ]
    present = set(sens_keys)
    required = set(dep_keys)
    missing = sorted(required - present)
    outside = sorted(present - required)
    if missing:
        raise TaylorContractError(
            f"required derivatives are MISSING for (position, factor) {missing[:10]}; "
            "an absent derivative is never an implicit zero -- supply 0.0 "
            "explicitly if it is a known zero."
        )
    if outside:
        raise TaylorContractError(
            f"derivatives supplied for (position, factor) {outside[:10]} outside "
            "the declared F(p): the representation describes a different model."
        )

    order = declaration.order
    has_cross = cross_sensitivities is not None and cross_sensitivities.height > 0
    if order is TaylorOrder.FIRST_ORDER:
        so = sensitivities["second_order"].to_numpy()
        if np.any(so != 0.0):
            raise TaylorContractError(
                "a FIRST_ORDER representation carries non-zero second-order terms."
            )
        if has_cross:
            raise TaylorContractError(
                "a FIRST_ORDER representation carries cross terms."
            )
        return
    if order is TaylorOrder.SECOND_ORDER_TRUNCATED:
        if has_cross:
            raise TaylorContractError(
                "SECOND_ORDER_TRUNCATED omits cross terms by declaration; a book "
                "that carries them must declare SECOND_ORDER_FULL."
            )
        return

    # SECOND_ORDER_FULL: every unordered pair of F(p) exactly once.
    got: list[tuple[int, int, int]] = []
    if has_cross:
        assert cross_sensitivities is not None
        for p, i, j in cross_sensitivities.select(
            "position_id", "factor_id_i", "factor_id_j"
        ).iter_rows():
            a, b = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
            got.append((int(p), a, b))
    if len(got) != len(set(got)):
        raise TaylorContractError(
            "SECOND_ORDER_FULL cross terms contain a duplicated pair (possibly as "
            "(i, j) and (j, i)); each unordered pair appears exactly once."
        )
    need = {
        (p, a, b)
        for p, fs in f_of.items()
        for a, b in itertools.combinations(sorted(fs), 2)
    }
    missing_x = sorted(need - set(got))
    outside_x = sorted(set(got) - need)
    if missing_x:
        raise TaylorContractError(
            f"SECOND_ORDER_FULL cross derivatives MISSING for {missing_x[:10]}; "
            "supply 0.0 explicitly for a known-zero cross term."
        )
    if outside_x:
        raise TaylorContractError(
            f"cross terms {outside_x[:10]} are outside the declared F(p)."
        )


def verify_declared_order_against_book(*, order: object, book: Any) -> None:
    """
    Bind a declared Taylor order to the COMPILED arrays the kernel evaluates.

    `validate_taylor_completeness` proves the order against the spec frames at
    compile. A declaration can afterwards be re-attached to another book
    (`dataclasses.replace`, direct construction); this check makes the pair
    coherent wherever a calculator is built. The kernel evaluates exactly
    `sum d1 dx + 1/2 sum d2 dx^2 + sum_{i<j} c dx_i dx_j` over the stored
    terms, so the order is coherent with a book iff the book stores exactly
    the terms the order admits (D081 §5):

    * FIRST_ORDER: no non-zero diagonal second order, no cross entry;
    * SECOND_ORDER_TRUNCATED: no cross entry;
    * SECOND_ORDER_FULL: every position carries each unordered pair of its
      factor rows exactly once, C(n_p, 2) entries (a compiled position's rows
      are its F(p), and a known-zero pair is stored explicitly).

    O(rows + cross entries); no numerics.
    """
    if not isinstance(order, TaylorOrder):
        raise TaylorContractError(f"undeclared Taylor order: {order!r}.")
    num_cross = int(book.cross_second_order.shape[0])
    if order is TaylorOrder.FIRST_ORDER:
        if bool(np.any(book.second_order != 0.0)):
            raise TaylorContractError(
                "a FIRST_ORDER declaration is attached to a book with non-zero "
                "second-order terms; the kernel would evaluate them."
            )
        if num_cross:
            raise TaylorContractError(
                f"a FIRST_ORDER declaration is attached to a book with {num_cross} "
                "cross term(s); the kernel would evaluate them."
            )
        return
    if order is TaylorOrder.SECOND_ORDER_TRUNCATED:
        if num_cross:
            raise TaylorContractError(
                "a SECOND_ORDER_TRUNCATED declaration omits cross terms, but the "
                f"attached book carries {num_cross}; declare SECOND_ORDER_FULL."
            )
        return
    rows = np.diff(np.asarray(book.position_offsets, dtype=np.int64))
    cross = np.diff(np.asarray(book.cross_offsets, dtype=np.int64))
    need = rows * (rows - 1) // 2
    if not np.array_equal(cross, need):
        bad = np.flatnonzero(cross != need)
        raise TaylorContractError(
            "a SECOND_ORDER_FULL declaration is attached to a book whose cross "
            "terms are not every unordered pair exactly once: positions "
            f"{np.asarray(book.position_uids)[bad][:10].tolist()} carry "
            f"{cross[bad][:10].tolist()} pair(s) where {need[bad][:10].tolist()} "
            "are required."
        )


def verify_coordinate_transform_base(
    *, transform_base_digest: str | None, declaration: TaylorDeclaration
) -> None:
    """
    A RiskQuant coordinate transformation must have been evaluated at the base
    the representation is bound to. Native-coordinate books (no transformation,
    hence no base-dependent operation) carry None and are unaffected.
    """
    if (
        transform_base_digest is not None
        and transform_base_digest != declaration.base_binding.base_levels_digest
    ):
        raise TaylorContractError(
            "coordinate transformation base differs from the bound Taylor base "
            "(base_levels_digest mismatch): the derivatives were transformed at "
            "another base than the one the representation claims."
        )
