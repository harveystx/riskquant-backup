# src/riskquant/ingest_fixtures/config/risk_static_config/_parts/surface_meta.py

from __future__ import annotations

import numpy as np
import polars as pl

from riskquant.ingest_fixtures.schemas import RiskStaticConfigSurfaceMeta
from riskquant.ingest_fixtures.utils import validate_column_lengths

# Three vol surfaces, one per asset class that supports vanilla options.
# Each exercises a distinct (x_axis_kind, y_axis_kind) combination:
#
#   BRENT_CRUDE_VOL_SURF       : (OFFSET_M w/ NEXT_MONTH, STRIKE)
#                          Listed-commodity option vol; absolute strike.
#   USDEUR_VOL_SURF      : (LABEL, DELTA_CALL)
#                          OTC FX vol convention; delta-quoted smile.
#   AAPL_VOL_SURF        : (LABEL, LOG_MONEYNESS_FWD)
#                          Listed-equity option vol; log-moneyness in
#                          forward terms.


CANONICAL_SURFACE_NAMES = [
    "BRENT_CRUDE_VOL_SURF",
    "USDEUR_VOL_SURF",
    "AAPL_VOL_SURF",
]


_NUM_CANONICAL_SURFACES = len(CANONICAL_SURFACE_NAMES)

_CANONICAL_SURFACE_IDS = np.arange(
    500_000_000,
    500_000_000 + _NUM_CANONICAL_SURFACES * 1000,
    step=1000,
    dtype=int,
).tolist()


def _get_data() -> RiskStaticConfigSurfaceMeta:
    return {
        "surface_id": _CANONICAL_SURFACE_IDS,
        "surface_name": CANONICAL_SURFACE_NAMES,
        "quote_ccy": [
            "USD",
            "USD",  # FX vol surface in standard pair quote convention
            "USD",
        ],
        "asset_class": [
            "COMMODITY",
            "FX",
            "EQUITY",
        ],
        "asset_subclass": [
            "CM_OIL",
            "FX_G10",
            "EQ_SINGLE_NAME",
        ],
        "quote_type": [
            "VOL",
            "VOL",
            "VOL",
        ],
        "unit_scale": [
            "DECIMAL",
            "DECIMAL",
            "DECIMAL",
        ],
        "shift_type": [
            "LOG_RETURN",
            "LOG_RETURN",
            "LOG_RETURN",
        ],
        "x_axis_kind": [
            "OFFSET_M",
            "LABEL",
            "LABEL",
        ],
        "x_interp_method": [
            "LINEAR",
            "LINEAR",
            "LINEAR",
        ],
        "x_extrap_method": [
            "FLAT",
            "FLAT",
            "FLAT",
        ],
        # Each surface owns its X-axis tenor-resolution convention directly,
        # exactly as curve_meta does. Only the method's required parameter is
        # populated; the rest are None.
        "roll_method": [
            "CALENDAR_PERIOD",  # OFFSET_M surface: monthly calendar roll
            None,  # LABEL FX vol surface
            None,  # LABEL equity vol surface
        ],
        "anchor_offset": [
            1,  # spot is next month
            None,
            None,
        ],
        "roll_month": [
            None,
            None,
            None,
        ],
        "roll_day": [
            None,
            None,
            None,
        ],
        "fh_bh_mmdd": [
            None,
            None,
            None,
        ],
        "y_axis_kind": [
            "STRIKE",
            "DELTA_CALL",
            "LOG_MONEYNESS_FWD",
        ],
        "y_interp_method": [
            "LINEAR",
            "LINEAR",
            "LINEAR",
        ],
        "y_extrap_method": [
            "FLAT",
            "FLAT",
            "FLAT",
        ],
    }


CANONICAL_SURFACE_META_DATA: RiskStaticConfigSurfaceMeta = _get_data()

validate_column_lengths(CANONICAL_SURFACE_META_DATA)

_CANONICAL_SURFACE_META_DF = pl.DataFrame(CANONICAL_SURFACE_META_DATA)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print_dict(CANONICAL_SURFACE_META_DATA, title="Canonical Surface Meta Dictionary")

    print("Canonical Surface Meta DataFrame (Transposed):")
    print(_CANONICAL_SURFACE_META_DF.transpose(include_header=True))

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
