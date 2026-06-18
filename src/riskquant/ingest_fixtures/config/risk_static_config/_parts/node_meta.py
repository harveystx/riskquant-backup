# src/riskquant/ingest_fixtures/config/risk_static_config/_parts/node_meta.py

from __future__ import annotations

import pandas as pd

from riskquant.ingest_fixtures.schemas import RiskStaticConfigNodeMeta
from riskquant.ingest_fixtures.utils import validate_column_lengths

# rf_id allocation (one shared id-space across nodes, curve_points and
# surface_points; all in the GlobalIdNamespace NODES range and globally
# unique — enforced by validate._validate_rf_id_uniqueness):
#   nodes          : 0 .. n_nodes-1                 (this table)
#   curve_points   : per-curve 1_000-wide blocks from 1_000
#   surface_points : per-surface 100_000-wide blocks from 100_000
#
# FX nodes: node_name MUST equal fx_base_ccy + quote_ccy. The validator
# enforces this. The fx_spot_lookup table is keyed by (base_ccy_id,
# quote_ccy_id) and populated from these rows — no string heuristics.
#
# VOL-NODE convention
# -------------------
# Per-underlying ATM volatility expressed as a scalar risk-factor node.
# The option-NODE kernel reads `vol_nodes[v_uix]` to get one sigma per
# scenario; treating the ATM vol as a node-typed observable matches that
# access pattern exactly. Surfaces stay for the option-CURVE kernel,
# which interpolates pillar-wise.
#
# Naming: `{UNDERLYING}_ATM_VOL`. The trailing `_ATM_VOL` token
# distinguishes a vol-node from a vol-surface (`{UNDERLYING}_VOL_SURF`)
# and from a price-node. Pricer-side resolution is purely
# `static_config.container_name_to_id`; no string parsing.

CANONICAL_NODE_NAMES = [
    # FX nodes
    "USDEUR",
    "USDCHF",
    "USDGBP",
    "USDAUD",
    "EURGBP",
    "EURCHF",
    # Equity nodes
    "AAPL_EQ",
    # ATM-vol nodes (one per option-node underlying that needs a scalar
    # implied vol — see module docstring).
    "AAPL_ATM_VOL",
    "USDEUR_ATM_VOL",
    "BRENT_CRUDE_ATM_VOL",
]

CANONICAL_NODE_IDS = list(range(0, len(CANONICAL_NODE_NAMES)))


def _get_data() -> RiskStaticConfigNodeMeta:
    return {
        "node_id": CANONICAL_NODE_IDS,
        "node_name": CANONICAL_NODE_NAMES,
        "quote_ccy": [
            # FX nodes — quote currency of the rate
            "EUR",
            "CHF",
            "GBP",
            "AUD",
            "GBP",
            "CHF",
            # Equity
            "USD",
            # Vol nodes — vols are dimensionless but the schema requires
            # a non-null currency. Use the underlying's quote ccy for
            # bookkeeping consistency (it does not affect the kernel).
            "USD",  # AAPL_ATM_VOL — underlying is USD-priced
            "EUR",  # USDEUR_ATM_VOL — underlying USDEUR is EUR-quoted
            "USD",  # BRENT_CRUDE_ATM_VOL — underlying is USD-priced
        ],
        "asset_class": [
            "FX",
            "FX",
            "FX",
            "FX",
            "FX",
            "FX",
            "EQUITY",
            # Vol nodes — classified under the underlying's asset class.
            # The vol-vs-price distinction is carried by `quote_type=VOL`.
            "EQUITY",
            "FX",
            "COMMODITY",
        ],
        "asset_subclass": [
            "FX_G10",
            "FX_G10",
            "FX_G10",
            "FX_G10",
            "FX_G10",
            "FX_G10",
            "EQ_SINGLE_NAME",
            # Vol nodes
            "EQ_SINGLE_NAME",
            "FX_G10",
            "CM_OIL",
        ],
        "quote_type": [
            "FX_RATE",
            "FX_RATE",
            "FX_RATE",
            "FX_RATE",
            "FX_RATE",
            "FX_RATE",
            "PRICE",
            # Vol nodes — `UnitType.VOL` is the canonical enum value.
            "VOL",
            "VOL",
            "VOL",
        ],
        "unit_scale": [
            "DECIMAL",
            "DECIMAL",
            "DECIMAL",
            "DECIMAL",
            "DECIMAL",
            "DECIMAL",
            "DECIMAL",
            # Vol nodes — 0.20 = 20% vol.
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
            # Vol nodes — log-shifts on vol keep sigma positive under
            # scenario stress, which matches the standard quant
            # convention for lognormal vol modelling.
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
            # Vol nodes — not FX.
            None,
            None,
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
