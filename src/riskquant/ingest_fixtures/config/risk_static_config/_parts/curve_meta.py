# src/riskquant/ingest_fixtures/config/risk_static_config/_parts/curve_meta.py

from __future__ import annotations

import numpy as np
import polars as pl

from riskquant.ingest_fixtures.schemas import RiskStaticConfigCurveMeta
from riskquant.ingest_fixtures.utils import validate_column_lengths

# container_id allocation. Curve and surface ids are DISJOINT id-spaces in
# the GlobalIdNamespace ranges (validate._validate_global_id_namespace):
#   curves    : CURVES range   — 250_000_000 + k*1_000   (this table)
#   surfaces  : SURFACES range — 500_000_000 + k*1_000
# The compiled RiskStaticConfig stores per-container attributes SLOT-indexed
# (parallel to a sorted container_ids axis), so these large ids never size a
# dense array.
#
# Discount-curve invariants (enforced by validate.py):
#   - Every ccy that holds any discount curve has exactly one row with
#     is_default_discount_for_ccy=True.
#   - The default's basis must be the firm-standard (OIS).
#   - For each (quote_ccy, discount_basis) pair there is at most one row.
#   - asset_class == "IR" iff discount_basis != "NOT_A_DISCOUNT_CURVE".
#
# Non-discount curves (commodities) carry discount_basis="NOT_A_DISCOUNT_CURVE"
# and is_default_discount_for_ccy=False.

CANONICAL_CURVE_NAMES = [
    "USD_SOFR_OIS",
    "USD_LIBOR_3M",
    "USD_CSA_EUR",
    "EUR_ESTR_OIS",
    "EUR_EURIBOR_6M",
    "GBP_SONIA_OIS",
    "CHF_SARON_OIS",
    "AUD_AONIA_OIS",
    "BRENT_CRUDE_FWD",
    "WTI_CRUDE_FWD",
    "HENRY_HUB_GAS_FWD",
    "CO2_EUA_ANNUAL_FWD",
    "POWER_CH_BASELOAD_HY",
]

_NUM_CANONICAL_CURVES = len(CANONICAL_CURVE_NAMES)

_CANONICAL_CURVE_IDS = np.arange(
    250_000_000,
    250_000_000 + _NUM_CANONICAL_CURVES * 1_000,
    step=1_000,
    dtype=int,
).tolist()


def _get_data() -> RiskStaticConfigCurveMeta:
    return {
        "curve_id": _CANONICAL_CURVE_IDS,
        "curve_name": CANONICAL_CURVE_NAMES,
        "quote_ccy": [
            "USD",
            "USD",
            "USD",
            "EUR",
            "EUR",
            "GBP",
            "CHF",
            "AUD",
            "USD",
            "USD",
            "USD",
            "EUR",
            "CHF",
        ],
        "asset_class": [
            "IR",
            "IR",
            "IR",
            "IR",
            "IR",
            "IR",
            "IR",
            "IR",
            "COMMODITY",
            "COMMODITY",
            "COMMODITY",
            "COMMODITY",
            "COMMODITY",
        ],
        "asset_subclass": [
            "IR_OIS",
            "IR_IBOR",
            "IR_OIS",
            "IR_OIS",
            "IR_IBOR",
            "IR_OIS",
            "IR_OIS",
            "IR_OIS",
            "CM_OIL",
            "CM_OIL",
            "CM_NATGAS",
            "CM_EUA",
            "CM_POWER",
        ],
        "quote_type": [
            "RATE",
            "RATE",
            "RATE",
            "RATE",
            "RATE",
            "RATE",
            "RATE",
            "RATE",
            "PRICE",
            "PRICE",
            "PRICE",
            "PRICE",
            "PRICE",
        ],
        "unit_scale": [
            "PERCENT",
            "PERCENT",
            "PERCENT",
            "PERCENT",
            "PERCENT",
            "PERCENT",
            "PERCENT",
            "PERCENT",
            "DECIMAL",
            "DECIMAL",
            "DECIMAL",
            "DECIMAL",
            "DECIMAL",
        ],
        "shift_type": [
            "DIFFERENCE",
            "DIFFERENCE",
            "DIFFERENCE",
            "DIFFERENCE",
            "DIFFERENCE",
            "DIFFERENCE",
            "DIFFERENCE",
            "DIFFERENCE",
            "LOG_RETURN",
            "LOG_RETURN",
            "LOG_RETURN",
            "LOG_RETURN",
            "LOG_RETURN",
        ],
        "x_axis_kind": [
            "LABEL",
            "LABEL",
            "LABEL",
            "LABEL",
            "LABEL",
            "LABEL",
            "LABEL",
            "LABEL",
            "OFFSET_M",
            "OFFSET_M",
            "OFFSET_M",
            "OFFSET_Y",
            "OFFSET_HY",
        ],
        "x_interp_method": [
            "LINEAR",
            "LINEAR",
            "LINEAR",
            "LINEAR",
            "LINEAR",
            "LINEAR",
            "LINEAR",
            "LINEAR",
            "LINEAR",
            "LINEAR",
            "LINEAR",
            "LINEAR",
            "LINEAR",
        ],
        "x_extrap_method": [
            "FLAT",
            "FLAT",
            "FLAT",
            "FLAT",
            "FLAT",
            "FLAT",
            "FLAT",
            "FLAT",
            "FLAT",
            "FLAT",
            "FLAT",
            "FLAT",
            "FLAT",
        ],
        # Each curve owns its tenor-resolution convention directly. Only the
        # method's required parameter is populated; every other roll field is
        # None (validate._validate_roll_params rejects meaningless params).
        #   LABEL (0-7)                -> all None (no as-of dependence)
        #   OFFSET_M  CALENDAR_PERIOD  -> anchor_offset (months) only
        #   OFFSET_Y  ROLL_ON_MMDD     -> roll_month + roll_day only
        #   OFFSET_HY HALF_YEAR_BY_LABEL -> fh_bh_mmdd only
        "roll_method": [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "CALENDAR_PERIOD",
            "CALENDAR_PERIOD",
            "CALENDAR_PERIOD",
            "ROLL_ON_MMDD",
            "HALF_YEAR_BY_LABEL",
        ],
        "anchor_offset": [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            # Monthly commodities: spot is next month (offset in months).
            1,
            1,
            1,
            None,
            None,
        ],
        "roll_month": [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            # Annual carbon rolls on 1 April.
            4,
            None,
        ],
        "roll_day": [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            None,
        ],
        "fh_bh_mmdd": [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            # Half-year power: first-half 1 Apr, back-half 1 Oct.
            "0401|1001",
        ],
        "discount_basis": [
            # IR discount curves carry a real basis.
            "OIS",
            "IBOR_3M",
            "CSA_EUR",
            "OIS",
            "IBOR_6M",
            "OIS",
            "OIS",
            "OIS",
            # Commodity curves are non-discount; carry the sentinel.
            "NOT_A_DISCOUNT_CURVE",
            "NOT_A_DISCOUNT_CURVE",
            "NOT_A_DISCOUNT_CURVE",
            "NOT_A_DISCOUNT_CURVE",
            "NOT_A_DISCOUNT_CURVE",
        ],
        "is_default_discount_for_ccy": [
            # Exactly one True per ccy that holds discount curves: the OIS row.
            True,
            False,
            False,
            True,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ],
    }


CANONICAL_CURVE_META_DATA: RiskStaticConfigCurveMeta = _get_data()

validate_column_lengths(CANONICAL_CURVE_META_DATA)

_CANONICAL_CURVE_META_DF = pl.DataFrame(CANONICAL_CURVE_META_DATA)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print_dict(CANONICAL_CURVE_META_DATA, title="Canonical Curve Meta Dictionary")

    print("Canonical Curve Meta DataFrame (Transposed):")
    print(_CANONICAL_CURVE_META_DF.transpose(include_header=True))

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
