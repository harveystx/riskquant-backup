# src/riskquant/ingest_fixtures/config/risk_static_config/_parts/curve_point_meta.py

from __future__ import annotations

import pandas as pd

from riskquant.ingest_fixtures.schemas import RiskStaticConfigCurvePointMeta
from riskquant.ingest_fixtures.utils import validate_column_lengths

# rf_id allocation per curve. Each curve gets a contiguous block; the
# first rf_id of the block is `_rf_id_base`, and points are emitted in
# the order their tenor_tokens appear in `_tenor_tokens`.
#
# rf_id partition (must not collide with node_meta or surface_point_meta):
#   nodes          : 0 ... 99_999
#   curve_points   : 100_000 ... 199_999
#   surface_points : 200_000 ... 299_999
#
# Tenor labels follow the IR conventional schedule: [ON]..[10Y].
# Monthly commodities use OFFSET_M tokens [M+k]; annual carbon uses
# OFFSET_Y tokens [Y+k]; half-year power uses OFFSET_HY tokens [HY+k].

_LABEL_TENORS: tuple[str, ...] = (
    "[ON]",
    "[1W]",
    "[1M]",
    "[3M]",
    "[6M]",
    "[1Y]",
    "[2Y]",
    "[5Y]",
    "[10Y]",
)

_MONTHLY_OFFSETS: tuple[str, ...] = (
    "[M+0]",
    "[M+1]",
    "[M+2]",
    "[M+3]",
    "[M+6]",
    "[M+12]",
)

_ANNUAL_OFFSETS: tuple[str, ...] = (
    "[Y+0]",
    "[Y+1]",
    "[Y+2]",
    "[Y+3]",
)

_HALF_YEAR_OFFSETS: tuple[str, ...] = (
    "[HY+0]",
    "[HY+1]",
    "[HY+2]",
    "[HY+3]",
)


# (curve_id, rf_id_base, tenor_tokens)
_CURVE_BLOCKS: tuple[tuple[int, int, tuple[str, ...]], ...] = (
    # --- USD discount curves ---
    (100_000_000, 1000, _LABEL_TENORS),  # USD_SOFR_OIS
    (100_000_001, 1010, _LABEL_TENORS),  # USD_LIBOR_3M
    (100_000_002, 1020, _LABEL_TENORS),  # USD_CSA_EUR
    # --- EUR discount curves ---
    (100_000_003, 1100, _LABEL_TENORS),  # EUR_ESTR_OIS
    (100_000_004, 1110, _LABEL_TENORS),  # EUR_EURIBOR_6M
    # --- GBP discount curves ---
    (100_000_005, 1200, _LABEL_TENORS),  # GBP_SONIA_OIS
    # --- CHF discount curves ---
    (100_000_006, 1300, _LABEL_TENORS),  # CHF_SARON_OIS
    # --- AUD discount curves ---
    (100_000_007, 1400, _LABEL_TENORS),  # AUD_AONIA_OIS
    # --- Commodity forward curves ---
    (100_000_008, 2000, _MONTHLY_OFFSETS),  # BRENT_CRUDE_FWD
    (100_000_009, 2010, _MONTHLY_OFFSETS),  # WTI_CRUDE_FWD
    (100_000_010, 2020, _MONTHLY_OFFSETS),  # HENRY_HUB_GAS_FWD
    (100_000_011, 2100, _ANNUAL_OFFSETS),  # CO2_EUA_ANNUAL_FWD
    (100_000_012, 2200, _HALF_YEAR_OFFSETS),  # POWER_CH_BASELOAD_HY
)


def _get_data() -> RiskStaticConfigCurvePointMeta:
    rf_ids: list[int] = []
    curve_ids: list[int] = []
    tenor_tokens: list[str] = []

    for curve_id, rf_id_base, tokens in _CURVE_BLOCKS:
        for offset, token in enumerate(tokens):
            rf_ids.append(rf_id_base + offset)
            curve_ids.append(curve_id)
            tenor_tokens.append(token)

    return {
        "rf_id": rf_ids,
        "curve_id": curve_ids,
        "tenor_token": tenor_tokens,
    }


CANONICAL_CURVE_POINT_META_DATA: RiskStaticConfigCurvePointMeta = _get_data()

validate_column_lengths(CANONICAL_CURVE_POINT_META_DATA)

_CANONICAL_CURVE_POINT_META_DF = pd.DataFrame(CANONICAL_CURVE_POINT_META_DATA)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print(
        f"Canonical Curve Point Meta Data ({len(CANONICAL_CURVE_POINT_META_DATA['rf_id'])} rows):"
    )
    print_dict(CANONICAL_CURVE_POINT_META_DATA)

    print("As DataFrame:")
    print_df(_CANONICAL_CURVE_POINT_META_DF, index=False)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
