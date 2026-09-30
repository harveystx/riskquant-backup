# tests/ingest_fixtures/config/risk_proxy_model/test_canonical_risk_proxy_model_fixture.py
"""
Canonical Proxy V1 authored-term fixture -> edge -> compiler -> digest.

Expected canonical arrays are written out by hand from D088 §6; the node rows
are also checked against an independent canonicaliser (bit-pattern ordering,
left-to-right float64 sum, exact-zero removal) that shares no code with the
compiler. The digest is a pinned golden: it fails on any change to the
canonical representation, the canonicalisation or the digest encoding.
"""

from __future__ import annotations

import itertools
import math
import random
import struct

import numpy as np
import polars as pl
import pytest

from riskquant.config.proxy_model.compile import compile_proxy_model
from riskquant.config.proxy_model.ingest import PROXY_MODEL_TERM_SCHEMA, proxy_terms_from_df
from riskquant.config.proxy_model.schema_objects import ContainerProxyTerm, NodeProxyTerm
from riskquant.config.static_config.compile import compile_risk_static_config_from_dict_of_dfs
from riskquant.exceptions import ConfigurationError
from riskquant.ingest_fixtures.as_of import CANONICAL_AS_OF_DATE
from riskquant.ingest_fixtures.config.risk_proxy_model.canonical_risk_proxy_model import (
    CANONICAL_RISK_PROXY_MODEL_DATA,
    CANONICAL_RISK_PROXY_MODEL_DF,
)
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
)
from riskquant.ingest_fixtures.schemas import RiskProxyModelTermData
from riskquant.market_data.risk_factors.shift_types import ShiftType
from riskquant.solutions.run_identity.hashing import hash_proxy_model

LOG, SIMPLE, DIFF = 1, 2, 3
AXIS_NAMES = ("USDEUR", "USDCHF", "USDGBP", "EURCHF", "AAPL_EQ", "WTI_CRUDE_FWD[M+0]", "WTI_CRUDE_FWD[M+1]")
#: Pinned golden (canonical static configuration, CANONICAL_AS_OF_DATE, AXIS_NAMES).
GOLDEN_DIGEST = "1df3ac636b64403dee58a164bd8eeaf8ae5d69669a4ba1e19efb7bf7284db75a"


@pytest.fixture(scope="module")
def cfg():
    return compile_risk_static_config_from_dict_of_dfs(dict(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS))


def _compile(cfg, df):
    axis = np.array([int(cfg.get_rf_id(n)) for n in AXIS_NAMES], dtype=np.int32)
    return compile_proxy_model(
        authored_terms=proxy_terms_from_df(df),
        static_config=cfg,
        as_of_date=CANONICAL_AS_OF_DATE,
        run_axis_rf_ids=axis,
    )


# ---- the fixture and the edge ----------------------------------------------------------------


def test_fixture_columns_match_the_typed_schema_and_the_edge_schema() -> None:
    assert list(CANONICAL_RISK_PROXY_MODEL_DATA) == list(RiskProxyModelTermData.__annotations__)
    assert dict(CANONICAL_RISK_PROXY_MODEL_DF.schema) == PROXY_MODEL_TERM_SCHEMA


def test_edge_maps_rows_to_terms_in_row_order_without_changing_them() -> None:
    assert proxy_terms_from_df(CANONICAL_RISK_PROXY_MODEL_DF) == (
        NodeProxyTerm(target_name="EURCHF", driver_name="USDCHF", driver_chart=ShiftType.LOG_RETURN, loading=1.0),
        NodeProxyTerm(target_name="EURCHF", driver_name="USDEUR", driver_chart=ShiftType.LOG_RETURN, loading=-1.0),
        NodeProxyTerm(target_name="AAPL_EQ", driver_name="USDEUR", driver_chart=ShiftType.DIFFERENCE, loading=0.3),
        NodeProxyTerm(target_name="AAPL_EQ", driver_name="USDEUR", driver_chart=ShiftType.DIFFERENCE, loading=0.2),
        NodeProxyTerm(target_name="AAPL_EQ", driver_name="USDGBP", driver_chart=ShiftType.SIMPLE_RETURN, loading=-0.0),
        ContainerProxyTerm(target_container_name="WTI_CRUDE_FWD", driver_container_name="BRENT_CRUDE_FWD",
                           driver_chart=ShiftType.LOG_RETURN, loading=0.95),
    )


# ---- canonical representation and golden digest ---------------------------------------------------


def test_canonical_arrays_are_the_hand_derived_rows(cfg) -> None:
    rf = {n: int(cfg.get_rf_id(n)) for n in (*AXIS_NAMES, "BRENT_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+1]")}
    m = _compile(cfg, CANONICAL_RISK_PROXY_MODEL_DF)
    expected = [
        (rf["USDEUR"], [(rf["USDEUR"], LOG, 1.0)]),  # identity row
        (rf["USDCHF"], [(rf["USDCHF"], LOG, 1.0)]),  # identity row
        (rf["USDGBP"], [(rf["USDGBP"], LOG, 1.0)]),  # identity row
        (rf["EURCHF"], sorted([(rf["USDCHF"], LOG, 1.0), (rf["USDEUR"], LOG, -1.0)])),
        (rf["AAPL_EQ"], [(rf["USDEUR"], DIFF, 0.5)]),  # 0.3 + 0.2 merged; -0.0 dropped; kappa LOG != tau DIFF
        (rf["WTI_CRUDE_FWD[M+0]"], [(rf["BRENT_CRUDE_FWD[M+0]"], LOG, 0.95)]),  # EXACT at the as-of
        (rf["WTI_CRUDE_FWD[M+1]"], [(rf["BRENT_CRUDE_FWD[M+1]"], LOG, 0.95)]),
    ]
    got = []
    for r, target in enumerate(m.row_target_rf_ids.tolist()):
        lo, hi = m.row_term_offsets[r], m.row_term_offsets[r + 1]
        got.append((target, list(zip(m.term_driver_rf_ids[lo:hi].tolist(), m.term_chart_ids[lo:hi].tolist(),
                                     m.term_loadings[lo:hi].tolist(), strict=True))))
    assert got == expected


def test_golden_digest(cfg) -> None:
    assert hash_proxy_model(_compile(cfg, CANONICAL_RISK_PROXY_MODEL_DF)) == GOLDEN_DIGEST


def test_authoring_row_order_does_not_change_the_digest(cfg) -> None:
    rows = list(range(CANONICAL_RISK_PROXY_MODEL_DF.height))
    perms = list(itertools.permutations(rows))
    for perm in random.Random(7).sample(perms, 40):
        assert hash_proxy_model(_compile(cfg, CANONICAL_RISK_PROXY_MODEL_DF[list(perm)])) == GOLDEN_DIGEST


# ---- independent canonicaliser ------------------------------------------------------------------


def _bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _oracle_row(terms: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """D088 §6 from first principles: drop zeros, group, order by bit pattern,
    left-to-right float64 sum, drop exact zero, sort by (driver, chart)."""
    groups: dict[tuple[int, int], list[float]] = {}
    for d, c, b in terms:
        if b != 0.0:
            groups.setdefault((d, c), []).append(b)
    out = []
    for key, coeffs in groups.items():
        total = 0.0
        for b in sorted(coeffs, key=_bits):
            total = total + b
        if total != 0.0:
            out.append((key[0], key[1], total))
    return sorted(out, key=lambda t: (t[0], t[1]))


@pytest.mark.parametrize(
    "loadings",
    [
        [0.3, 0.2],
        [0.1, 0.2, 0.3],
        [0.3, 0.1, 0.2],
        [1e16, 1.0, -1e16],
        [1e16, -1e16, 1.0],  # authored order sums to 1.0; bit-pattern order sums to 0.0
        [0.3, 0.2, 0.1],  # authored order 0.6; bit-pattern order 0.6000000000000001
        [0.7, -0.7],  # exact cancellation of the (USDEUR, DIFF) group
        [0.5, -0.0, +0.0],
    ],
)
def test_compiler_matches_the_independent_canonicaliser_bit_for_bit(cfg, loadings) -> None:
    rows = {
        "term_kind": ["NODE"] * (len(loadings) + 1),
        "target_name": ["AAPL_EQ"] * (len(loadings) + 1),
        "driver_name": ["USDEUR"] * len(loadings) + ["USDGBP"],
        "driver_chart": ["DIFFERENCE"] * len(loadings) + ["SIMPLE_RETURN"],
        "loading": [*loadings, 0.25],
    }
    m = _compile(cfg, pl.DataFrame(rows, schema=PROXY_MODEL_TERM_SCHEMA))
    usdeur, usdgbp, aapl = (int(cfg.get_rf_id(n)) for n in ("USDEUR", "USDGBP", "AAPL_EQ"))
    r = int(np.flatnonzero(m.row_target_rf_ids == aapl)[0])
    lo, hi = m.row_term_offsets[r], m.row_term_offsets[r + 1]
    got = list(zip(m.term_driver_rf_ids[lo:hi].tolist(), m.term_chart_ids[lo:hi].tolist(),
                   m.term_loadings[lo:hi].tolist(), strict=True))
    expected = _oracle_row([(usdeur, DIFF, b) for b in loadings] + [(usdgbp, SIMPLE, 0.25)])
    assert [(d, c, _bits(b)) for d, c, b in got] == [(d, c, _bits(b)) for d, c, b in expected]


def test_a_row_that_cancels_to_nothing_is_rejected(cfg) -> None:
    df = pl.DataFrame({"term_kind": ["NODE", "NODE"], "target_name": ["AAPL_EQ", "AAPL_EQ"],
                       "driver_name": ["USDEUR", "USDEUR"], "driver_chart": ["DIFFERENCE", "DIFFERENCE"],
                       "loading": [0.7, -0.7]}, schema=PROXY_MODEL_TERM_SCHEMA)
    with pytest.raises(ConfigurationError):
        _compile(cfg, df)


# ---- edge rejections --------------------------------------------------------------------------


def _mutated(**changes) -> pl.DataFrame:
    return CANONICAL_RISK_PROXY_MODEL_DF.with_columns(**changes)


@pytest.mark.parametrize(
    "df",
    [
        CANONICAL_RISK_PROXY_MODEL_DF.drop("loading"),
        CANONICAL_RISK_PROXY_MODEL_DF.with_columns(target_chart=pl.lit("LOG_RETURN")),
        CANONICAL_RISK_PROXY_MODEL_DF.with_columns(pl.col("loading").cast(pl.Float32)),
        _mutated(target_name=pl.when(pl.int_range(pl.len()) == 2).then(None).otherwise(pl.col("target_name"))),
        _mutated(driver_name=pl.when(pl.int_range(pl.len()) == 1).then(pl.lit("  ")).otherwise(pl.col("driver_name"))),
        _mutated(term_kind=pl.when(pl.int_range(pl.len()) == 0).then(pl.lit("RISK_FACTOR")).otherwise(pl.col("term_kind"))),
        _mutated(driver_chart=pl.when(pl.int_range(pl.len()) == 0).then(pl.lit("ABSOLUTE_BASE_RETURN")).otherwise(pl.col("driver_chart"))),
        _mutated(driver_chart=pl.when(pl.int_range(pl.len()) == 0).then(pl.lit("NO_SHIFT")).otherwise(pl.col("driver_chart"))),
        _mutated(driver_chart=pl.when(pl.int_range(pl.len()) == 0).then(pl.lit("log_return")).otherwise(pl.col("driver_chart"))),
        _mutated(loading=pl.when(pl.int_range(pl.len()) == 0).then(pl.lit(math.nan)).otherwise(pl.col("loading"))),
        _mutated(loading=pl.when(pl.int_range(pl.len()) == 0).then(pl.lit(math.inf)).otherwise(pl.col("loading"))),
    ],
    ids=["missing_column", "extra_target_chart_column", "float32_loading", "null", "blank_name",
         "d013_term_kind", "abs_chart", "no_shift_chart", "lowercase_chart", "nan_loading", "inf_loading"],
)
def test_the_edge_rejects_malformed_frames(df) -> None:
    with pytest.raises(ConfigurationError):
        proxy_terms_from_df(df)


# ---- independent digest derivation (no production hashing code) ----------------------------


def _independent_digest(named: list[tuple[str, np.ndarray]]) -> str:
    """D088 §7 encoding re-derived: per array sha256(dtype.str | shape | LE
    C-order bytes), bundled as name=hex; in declaration order."""
    import hashlib

    bundle = hashlib.sha256()
    for name, arr in named:
        a = np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("<"), copy=False))
        h = hashlib.sha256()
        h.update(a.dtype.str.encode() + b"|" + str(a.shape).encode() + b"|" + a.tobytes())
        bundle.update(name.encode() + b"=" + h.hexdigest().encode() + b";")
    return bundle.hexdigest()


def test_golden_digest_is_reproduced_from_the_hand_derived_rows(cfg) -> None:
    rf = {n: int(cfg.get_rf_id(n)) for n in (*AXIS_NAMES, "BRENT_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+1]")}
    rows = [
        (rf["USDEUR"], [(rf["USDEUR"], LOG, 1.0)]),
        (rf["USDCHF"], [(rf["USDCHF"], LOG, 1.0)]),
        (rf["USDGBP"], [(rf["USDGBP"], LOG, 1.0)]),
        (rf["EURCHF"], sorted([(rf["USDCHF"], LOG, 1.0), (rf["USDEUR"], LOG, -1.0)])),
        (rf["AAPL_EQ"], [(rf["USDEUR"], DIFF, 0.5)]),
        (rf["WTI_CRUDE_FWD[M+0]"], [(rf["BRENT_CRUDE_FWD[M+0]"], LOG, 0.95)]),
        (rf["WTI_CRUDE_FWD[M+1]"], [(rf["BRENT_CRUDE_FWD[M+1]"], LOG, 0.95)]),
    ]
    terms = [t for _, ts in rows for t in ts]
    offsets = np.cumsum([0] + [len(ts) for _, ts in rows])
    named = [
        ("row_target_rf_ids", np.array([r for r, _ in rows], np.int32)),
        ("row_term_offsets", np.array(offsets, np.int32)),
        ("term_driver_rf_ids", np.array([t[0] for t in terms], np.int32)),
        ("term_chart_ids", np.array([t[1] for t in terms], np.int8)),
        ("term_loadings", np.array([t[2] for t in terms], np.float64)),
    ]
    assert _independent_digest(named) == GOLDEN_DIGEST


# ---- the edge contract (D093): exact dtypes, verbatim values, full preservation -------------


def _frame(rows, **schema_over) -> pl.DataFrame:
    schema = {**PROXY_MODEL_TERM_SCHEMA, **schema_over}
    return pl.DataFrame({k: [r[i] for r in rows] for i, k in enumerate(PROXY_MODEL_TERM_SCHEMA)}, schema=schema)


_ROWS = [("NODE", "AAPL_EQ", "USDEUR", "DIFFERENCE", 0.25)]


@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame({"term_kind": ["NODE"], "target_name": ["AAPL_EQ"], "driver_name": ["USDEUR"],
                      "driver_chart": ["DIFFERENCE"], "loading": [1]}),  # Int64 loading
        pl.DataFrame({"term_kind": ["NODE"], "target_name": ["AAPL_EQ"], "driver_name": ["USDEUR"],
                      "driver_chart": ["DIFFERENCE"], "loading": ["0.25"]}),  # String loading
        _frame(_ROWS).with_columns(pl.col("loading").cast(pl.Decimal(10, 4))),
        _frame(_ROWS).with_columns(pl.col("driver_chart").cast(pl.Categorical)),
        _frame(_ROWS).with_columns(pl.col("term_kind").cast(pl.Enum(["NODE", "CONTAINER"]))),
        _frame(_ROWS).with_columns(pl.col("target_name").cast(pl.Categorical)),
    ],
    ids=["int_loading", "string_loading", "decimal_loading", "categorical_chart", "enum_kind", "categorical_name"],
)
def test_the_edge_does_not_coerce_dtypes(df) -> None:
    with pytest.raises(ConfigurationError, match="dtype"):
        proxy_terms_from_df(df)


@pytest.mark.parametrize("column", list(PROXY_MODEL_TERM_SCHEMA))
def test_a_null_in_any_column_is_rejected(column) -> None:
    df = _frame(_ROWS * 2).with_columns(
        pl.when(pl.int_range(pl.len()) == 1).then(None).otherwise(pl.col(column)).alias(column)
    )
    with pytest.raises(ConfigurationError, match=f"row 1, column '{column}'"):
        proxy_terms_from_df(df)


@pytest.mark.parametrize("name", ["", " ", "\t", "  \n "])
def test_empty_and_whitespace_only_names_are_rejected(name) -> None:
    with pytest.raises(ConfigurationError, match="blank name"):
        proxy_terms_from_df(_frame([("NODE", name, "USDEUR", "DIFFERENCE", 0.25)]))


def test_names_are_passed_verbatim_and_resolved_only_by_the_compiler(cfg) -> None:
    """The edge does not trim, resolve or check existence; the compiler refuses
    a name that does not resolve at the as-of."""
    (term,) = proxy_terms_from_df(_frame([("NODE", " AAPL_EQ", "NOT_A_FACTOR", "DIFFERENCE", 0.25)]))
    assert (term.target_name, term.driver_name) == (" AAPL_EQ", "NOT_A_FACTOR")
    with pytest.raises(ConfigurationError):
        _compile(cfg, _frame([("NODE", "AAPL_EQ", "NOT_A_FACTOR", "DIFFERENCE", 0.25)]))


def test_the_edge_preserves_every_row_exactly() -> None:
    """Row count, order, duplicates, +0.0 and -0.0, negative loadings, mixed
    kinds and several charts of one driver: nothing is merged, dropped,
    reordered or expanded."""
    rows = [
        ("NODE", "AAPL_EQ", "USDEUR", "LOG_RETURN", 0.2),
        ("CONTAINER", "WTI_CRUDE_FWD", "BRENT_CRUDE_FWD", "LOG_RETURN", 0.95),
        ("NODE", "AAPL_EQ", "USDEUR", "LOG_RETURN", 0.3),
        ("NODE", "AAPL_EQ", "USDEUR", "SIMPLE_RETURN", -1.5),
        ("NODE", "AAPL_EQ", "USDEUR", "DIFFERENCE", 0.0),
        ("NODE", "AAPL_EQ", "USDEUR", "DIFFERENCE", -0.0),
        ("NODE", "AAPL_EQ", "USDEUR", "LOG_RETURN", 0.2),
    ]
    terms = proxy_terms_from_df(_frame(rows))
    assert len(terms) == len(rows)
    for (kind, target, driver, chart, beta), t in zip(rows, terms, strict=True):
        assert isinstance(t, NodeProxyTerm if kind == "NODE" else ContainerProxyTerm)
        names = (t.target_name, t.driver_name) if kind == "NODE" else (t.target_container_name, t.driver_container_name)
        assert names == (target, driver) and t.driver_chart.name == chart
        assert _bits(t.loading) == _bits(beta)  # the sign of zero is preserved


def test_physical_column_order_is_not_semantic() -> None:
    reordered = CANONICAL_RISK_PROXY_MODEL_DF.select(list(reversed(CANONICAL_RISK_PROXY_MODEL_DF.columns)))
    assert proxy_terms_from_df(reordered) == proxy_terms_from_df(CANONICAL_RISK_PROXY_MODEL_DF)


def test_an_empty_frame_authors_no_terms_and_compiles_to_identity_rows(cfg) -> None:
    empty = CANONICAL_RISK_PROXY_MODEL_DF.clear()
    assert proxy_terms_from_df(empty) == ()
    m = _compile(cfg, empty)
    assert m.term_loadings.tolist() == [1.0] * len(AXIS_NAMES)
    assert m.term_driver_rf_ids.tolist() == m.row_target_rf_ids.tolist()


def test_a_container_row_equals_the_typed_container_term(cfg) -> None:
    """The edge does not expand containers; frame and typed authoring compile
    to the same canonical arrays and digest."""
    frame = _frame([("CONTAINER", "WTI_CRUDE_FWD", "BRENT_CRUDE_FWD", "LOG_RETURN", 0.95)])
    typed = (ContainerProxyTerm(target_container_name="WTI_CRUDE_FWD", driver_container_name="BRENT_CRUDE_FWD",
                                driver_chart=ShiftType.LOG_RETURN, loading=0.95),)
    axis = np.array([int(cfg.get_rf_id(n)) for n in AXIS_NAMES], dtype=np.int32)
    a = _compile(cfg, frame)
    b = compile_proxy_model(authored_terms=typed, static_config=cfg, as_of_date=CANONICAL_AS_OF_DATE, run_axis_rf_ids=axis)
    assert hash_proxy_model(a) == hash_proxy_model(b)
    for f in ("row_target_rf_ids", "row_term_offsets", "term_driver_rf_ids", "term_chart_ids", "term_loadings"):
        assert getattr(a, f).tobytes() == getattr(b, f).tobytes()
