# src/riskquant/ingest_fixtures/config/risk_static_config/_parts/surface_point_meta.py

from __future__ import annotations

import pandas as pd

from riskquant.ingest_fixtures.schemas import RiskStaticConfigSurfacePointMeta
from riskquant.ingest_fixtures.utils import validate_column_lengths

# rf_id allocation (one shared id-space; globally unique vs node_meta and
# curve_point_meta — enforced by validate._validate_rf_id_uniqueness).
# Each surface gets a contiguous 100_000-wide block from `rf_id_base`:
#   nodes          : 0 .. n_nodes-1
#   curve_points   : 1_000-wide blocks from 1_000
#   surface_points : 100_000-wide blocks from 100_000    (this table)
#
# Each surface is a 4-expiry x 5-y-label grid (20 points per surface).
# Total surface points: 3 surfaces x 20 = 60.
#
# y_label is stored verbatim (case-preserving) in surface_point_meta;
# the validator does not uppercase y labels since strike values are
# numeric-typed strings ("50.00", "75.50") that the surface interpolator
# parses at runtime.

# (surface_id, rf_id_base, x_tokens, y_labels)
_SURFACE_BLOCKS: tuple[tuple[int, int, tuple[str, ...], tuple[str, ...]], ...] = (
    # --- BRENT_VOL_SURF: 4 monthly expiries x 5 absolute strikes ---
    (
        500_000_000,
        100_000,
        ("[M+0]", "[M+1]", "[M+3]", "[M+6]"),
        ("50.00", "65.00", "75.00", "85.00", "100.00"),
    ),
    # --- USDEUR_VOL_SURF: 4 LABEL expiries x 5 delta points ---
    (
        500_001_000,
        200_000,
        ("[1W]", "[1M]", "[3M]", "[1Y]"),
        ("0.10", "0.25", "0.50", "0.75", "0.90"),
    ),
    # --- AAPL_VOL_SURF: 4 LABEL expiries x 5 log-moneyness levels ---
    (
        500_002_000,
        300_000,
        ("[1M]", "[3M]", "[6M]", "[1Y]"),
        ("-0.20", "-0.10", "0.00", "0.10", "0.20"),
    ),
)


def _get_data() -> RiskStaticConfigSurfacePointMeta:
    rf_ids: list[int] = []
    surface_ids: list[int] = []
    x_tokens: list[str] = []
    y_labels: list[str] = []

    for surface_id, rf_id_base, xs, ys in _SURFACE_BLOCKS:
        idx = 0
        for x_token in xs:
            for y_label in ys:
                rf_ids.append(rf_id_base + idx)
                surface_ids.append(surface_id)
                x_tokens.append(x_token)
                y_labels.append(y_label)
                idx += 1

    return {
        "rf_id": rf_ids,
        "surface_id": surface_ids,
        "x_tenor_token": x_tokens,
        "y_label": y_labels,
    }


CANONICAL_SURFACE_POINT_META_DATA: RiskStaticConfigSurfacePointMeta = _get_data()

validate_column_lengths(CANONICAL_SURFACE_POINT_META_DATA)

_CANONICAL_SURFACE_POINT_META_DF = pd.DataFrame(CANONICAL_SURFACE_POINT_META_DATA)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print(
        f"Canonical Surface Point Meta Data "
        f"({len(CANONICAL_SURFACE_POINT_META_DATA['rf_id'])} rows):"
    )
    print_dict(CANONICAL_SURFACE_POINT_META_DATA)

    print("As DataFrame:")
    print_df(_CANONICAL_SURFACE_POINT_META_DF, index=False)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
