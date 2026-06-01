# src/riskquant/ingest_fixtures/config/risk_static_config/_parts/node_meta.py

from __future__ import annotations

import pandas as pd

from riskquant.ingest_fixtures.schemas import RiskStaticConfigNodeMeta
from riskquant.ingest_fixtures.utils import validate_column_lengths

# rf_id namespace partition (shared across nodes, curve_points, surface_points):
#   nodes          : 1..99
#   curve_points   : 100..9_999
#   surface_points : 10_000..99_999
#
# FX nodes: node_name MUST equal fx_base_ccy + quote_ccy. The validator
# enforces this. The fx_spot_lookup table is keyed by (base_ccy_id,
# quote_ccy_id) and populated from these rows — no string heuristics.

CANONICAL_NODE_NAMES = [
    "USDEUR",
    "USDCHF",
    "USDGBP",
    "USDAUD",
    "EURGBP",
    "EURCHF",
    "AAPL_EQ",
]

CANONICAL_NODE_IDS = list(range(0, len(CANONICAL_NODE_NAMES)))


def _get_data() -> RiskStaticConfigNodeMeta:
    return {
        "node_id": CANONICAL_NODE_IDS,
        "node_name": CANONICAL_NODE_NAMES,
        "quote_ccy": [
            "EUR",
            "CHF",
            "GBP",
            "AUD",
            "GBP",
            "CHF",
            "USD",
        ],
        "asset_class": [
            "FX",
            "FX",
            "FX",
            "FX",
            "FX",
            "FX",
            "EQUITY",
        ],
        "asset_subclass": [
            "FX_G10",
            "FX_G10",
            "FX_G10",
            "FX_G10",
            "FX_G10",
            "FX_G10",
            "EQ_SINGLE_NAME",
        ],
        "unit_type": [
            "FX_RATE",
            "FX_RATE",
            "FX_RATE",
            "FX_RATE",
            "FX_RATE",
            "FX_RATE",
            "PRICE",
        ],
        "unit_scale": [
            "DECIMAL",
            "DECIMAL",
            "DECIMAL",
            "DECIMAL",
            "DECIMAL",
            "DECIMAL",
            "DECIMAL",
        ],
        "shift_type": [
            "LOG_RETURN",
            "LOG_RETURN",
            "LOG_RETURN",
            "LOG_RETURN",
            "LOG_RETURN",
            "LOG_RETURN",
            "LOG_RETURN",
        ],
        "fx_base_ccy": [
            "USD",
            "USD",
            "USD",
            "USD",
            "EUR",
            "EUR",
            None,
        ],
    }


CANONICAL_NODE_META_DATA: RiskStaticConfigNodeMeta = _get_data()

validate_column_lengths(CANONICAL_NODE_META_DATA)

_CANONICAL_NODE_META_DF = pd.DataFrame(CANONICAL_NODE_META_DATA)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical Node Meta Data:")
    print_dict(CANONICAL_NODE_META_DATA)

    print("As DataFrame:")
    print_df(_CANONICAL_NODE_META_DF, index=False)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
