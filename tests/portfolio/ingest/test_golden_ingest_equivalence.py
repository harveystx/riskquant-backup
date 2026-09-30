# tests/portfolio/ingest/test_golden_ingest_equivalence.py
"""
Golden equivalence test: Polars ingest pipeline vs. pandas-era oracle.

Oracle: 6 SPOT rows, hostile static config (sparse interleaved RF ids, mixed
CCY, negative-level prices). Captured by running the pandas pipeline on the
same fixture before migration. See ``_pandas_era_oracle.json``.

Decision D5: Polars nullable int (netting_set_id all non-null in this fixture).
Decision D6: Error row indices now 0-based positional integers.
Decision D7: Date columns now ``pl.Date``; ordinal encoding unchanged.

Pre-existing defect noted:
    The pandas 3.0 pipeline raises ``IntCastingNaNError`` on
    ``_materialize_nullable_int_col`` when ``netting_set_id`` contains null
    values. This defect is fixed naturally by the Polars migration (pl.Int64
    preserves nulls natively). The oracle was captured from a fixture with no
    null netting_set_ids to avoid the pre-existing defect.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_DIR = Path(__file__).parent
_ORACLE_PATH = _DIR / "_pandas_era_oracle.json"

# hostile_fixture lives under tests/market_data/remediation/
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "market_data" / "remediation")
)

import hostile_fixture as HF  # noqa: E402  (sys.path manipulation above)

from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch


@pytest.fixture(scope="module")
def oracle() -> dict:
    return json.loads(_ORACLE_PATH.read_text())


@pytest.fixture(scope="module")
def static_config():
    return compile_risk_static_config_from_dict_of_dfs(HF.static_config_dfs())


@pytest.fixture(scope="module")
def polars_batch(static_config):
    """Run the Polars pipeline on the same SPOT_BOOK used for the oracle."""
    spot_pl = pl.from_pandas(HF.SPOT_BOOK)
    return compile_raw_df_into_ingest_batch(raw_df=spot_pl, static_config=static_config)


class TestGoldenIngestEquivalence:
    """Byte-for-byte field comparison against the pandas-era oracle."""

    def _check_field(self, polars_batch, oracle, field: str) -> None:
        arr = getattr(polars_batch, field)
        got = hashlib.sha256(arr.tobytes()).hexdigest()
        exp = oracle["fields"][field]["sha256"]
        assert got == exp, (
            f"Field '{field}' sha256 mismatch.\n"
            f"  expected : {exp}\n"
            f"  got      : {got}\n"
            f"  expected values : {oracle['fields'][field]['values']}\n"
            f"  got values      : {arr.tolist()}"
        )

    def test_batch_size(self, polars_batch, oracle):
        assert polars_batch.size == oracle["size"]

    def test_as_of_ord32(self, polars_batch, oracle):
        assert polars_batch.as_of_ord32 == oracle["as_of_ord32"]

    def test_instrument_ids(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "instrument_ids")

    def test_uids(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "uids")

    def test_qty(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "qty")

    def test_strike_price(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "strike_price")

    def test_cp_mult(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "cp_mult")

    def test_settle_ord(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "settle_ord")

    def test_delivery_ord(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "delivery_ord")

    def test_expiry_ord(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "expiry_ord")

    def test_underlying_ids(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "underlying_ids")

    def test_volatility_ids(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "volatility_ids")

    def test_fx_u2s_ids(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "fx_u2s_ids")

    def test_fx_u2s_inv(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "fx_u2s_inv")

    def test_fx_k2s_ids(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "fx_k2s_ids")

    def test_fx_k2s_inv(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "fx_k2s_inv")

    def test_under_discount_curve_ids(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "under_discount_curve_ids")

    def test_strike_discount_curve_ids(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "strike_discount_curve_ids")

    def test_vol_surface_y_axis_type_ids(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "vol_surface_y_axis_type_ids")

    def test_netting_set_id(self, polars_batch, oracle):
        self._check_field(polars_batch, oracle, "netting_set_id")


class TestNullableNettingSetId:
    """
    Verify the Polars pipeline handles nullable netting_set_id correctly.

    Pre-existing pandas 3.0 defect: ``_materialize_nullable_int_col`` would
    raise ``IntCastingNaNError`` on null netting_set_ids. This case is now
    handled correctly by pl.Int64 native null semantics.
    """

    def test_null_netting_set_id_becomes_sentinel(self, static_config):
        from riskquant.constants import NO_ID

        spot_with_null_netting = pl.from_dict(
            {
                "uid": [101, 102],
                "instrument": ["SPOT", "SPOT"],
                "quantity": [100.0, -250.0],
                "underlying": ["EQ_USD", "EQ_GBP"],
                "settlement_ccy": ["EUR", "EUR"],
                "netting_set_id": [None, None],  # all null
            }
        )

        batch = compile_raw_df_into_ingest_batch(
            raw_df=spot_with_null_netting,
            static_config=static_config,
        )
        assert batch.netting_set_id.dtype == np.int64
        assert np.all(batch.netting_set_id == int(NO_ID))

    def test_mixed_null_netting_set_id(self, static_config):
        from riskquant.constants import NO_ID

        spot_mixed = pl.from_dict(
            {
                "uid": [101, 102],
                "instrument": ["SPOT", "SPOT"],
                "quantity": [100.0, -250.0],
                "underlying": ["EQ_USD", "EQ_GBP"],
                "settlement_ccy": ["EUR", "EUR"],
                "netting_set_id": [42, None],
            }
        )

        batch = compile_raw_df_into_ingest_batch(
            raw_df=spot_mixed,
            static_config=static_config,
        )
        assert batch.netting_set_id[0] == 42
        assert batch.netting_set_id[1] == int(NO_ID)
