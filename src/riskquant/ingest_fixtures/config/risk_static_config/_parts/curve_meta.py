# src/riskquant/ingest_fixtures/config/risk_static_config/_parts/curve_meta.py

from __future__ import annotations

import numpy as np
import pandas as pd

from riskquant.ingest_fixtures.schemas import RiskStaticConfigCurveMeta
from riskquant.ingest_fixtures.utils import validate_column_lengths

# container_id namespace partition (shared across curves and surfaces):
#   discount curves       : 100..199
#   non-discount curves   : 200..299
#   surfaces              : 300..399
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
    100_000_000,
    100_000_000 + _NUM_CANONICAL_CURVES,
    dtype=int,
)


def _get_data() -> RiskStaticConfigCurveMeta:
    return {
        "curve_id": list(_CANONICAL_CURVE_IDS),
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
        "unit_type": [
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
        "spot_roll_rule_name": [
            # LABEL curves have no roll rule.
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            # Monthly commodities use NEXT_MONTH.
            "NEXT_MONTH",
            "NEXT_MONTH",
            "NEXT_MONTH",
            # Annual carbon uses April-begin annual roll.
            "ANNUAL_ROLL_APRIL_BGN",
            # Half-year power uses the label-form half-year roll.
            "HALF_YEAR_LABEL_APR_OCT",
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

_CANONICAL_CURVE_META_DF = pd.DataFrame(CANONICAL_CURVE_META_DATA)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print_dict(CANONICAL_CURVE_META_DATA, title="Canonical Curve Meta Dictionary")

    print_df(
        _CANONICAL_CURVE_META_DF.T,
        index=True,
        title="Canonical Curve Meta DataFrame (Transposed)",
    )

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
