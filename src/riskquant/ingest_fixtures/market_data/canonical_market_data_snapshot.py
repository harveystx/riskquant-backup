# src/riskquant/ingest_fixtures/market_data/canonical_market_data_snapshot.py
"""
Canonical as-of-date market-data snapshot.

Ground-truth set of risk-factor levels at the canonical as-of date. Every
key is a **relative risk-factor name** that resolves through
`RiskStaticConfig.rel_name_to_rf_id` — the fixture stays in verbatim
alignment with the static-config canonical fixture so the two can be
compiled together and round-trip through the typed snapshot without any
string-parsing layer.

Layout
------
The primary contract is a single flat `dict[rel_name, level]`:

    "AAPL_EQ"                            -> node level
    "USD_SOFR_OIS[1Y]"                   -> curve point level
    "BRENT_CRUDE_VOL_SURF[M+0][50.00]"   -> surface point level

Coverage
--------
- 7 nodes        (all canonical FX + equity)
- 98 curve points  (13 curves: 8 IR × 9 pillars + 3 OFFSET_M × 6 + 1 OFFSET_Y × 4 + 1 OFFSET_HY × 4)
- 60 surface points  (3 surfaces × 4 expiries × 5 y-labels)
Total: 165 levels.

Unit conventions
----------------
The fixture stores levels at the unit_scale declared by the static
config (so a user reading the file sees `5.30` for a 5.30 % rate, not
`0.0530`):

    FX / equity nodes  : DECIMAL  e.g. USDEUR = 0.9250, AAPL_EQ = 185.50
    IR curve points    : PERCENT  e.g. USD_SOFR_OIS[1Y] = 4.80
    Commodity curves   : DECIMAL  e.g. BRENT_CRUDE_FWD[M+0] = 82.50
    Vol surfaces       : DECIMAL  e.g. AAPL_VOL_SURF[1M][0.00] = 0.22

The snapshot compiler
(`compile_risk_factor_level_snapshot_from_levels_dict`) applies the
per-rf `unit_scale_factor` so the compiled `factor_levels` array is
always in base DECIMAL.

DELTA-quoted surfaces (USDEUR_VOL_SURF)
---------------------------------------
The runtime `build_market_data_matrix` REJECTS delta-quoted surfaces
because no DELTA → STRIKE calibration is in place. The snapshot still
emits the levels — it is a pure-data fixture with no opinion about
which surfaces are pricer-runnable. Downstream filtering is the caller's
job.

Validation
----------
A module-import-time check rebuilds the expected relative-name keyset
from the static-config canonical fixture and fails loud on any drift.
"""

from __future__ import annotations

from typing import Final

from riskquant.ingest_fixtures.config.risk_static_config._parts.curve_meta import (
    CANONICAL_CURVE_NAMES,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.curve_point_meta import (
    CANONICAL_CURVE_POINT_META_DATA,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.node_meta import (
    CANONICAL_NODE_NAMES,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.surface_meta import (
    CANONICAL_SURFACE_NAMES,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.surface_point_meta import (
    CANONICAL_SURFACE_POINT_META_DATA,
)

# ============================================================================
# 1. NODES  (7 scalars, all DECIMAL per node_meta.unit_scale)
# ============================================================================
# FX convention: <base><quote>, value = 1 unit of base in quote ccy.
_NODE_LEVELS: Final[dict[str, float]] = {
    "USDEUR": 0.9250,
    "USDCHF": 0.8800,
    "USDGBP": 0.7850,
    "USDAUD": 1.5200,
    "EURGBP": 0.8480,
    "EURCHF": 0.9510,
    "AAPL_EQ": 185.50,
}


# ============================================================================
# 2. CURVES  (117 pillar levels across 13 curves)
# ============================================================================
# Keys are `f"{curve_name}{tenor_token}"` — the exact rel-name form the
# static config emits. IR curves are PERCENT; commodities DECIMAL.
_CURVE_LEVELS: Final[dict[str, float]] = {
    # ---- USD_SOFR_OIS  (LABEL, PERCENT) ----
    "USD_SOFR_OIS[ON]": 5.30,
    "USD_SOFR_OIS[1W]": 5.31,
    "USD_SOFR_OIS[1M]": 5.32,
    "USD_SOFR_OIS[3M]": 5.25,
    "USD_SOFR_OIS[6M]": 5.10,
    "USD_SOFR_OIS[1Y]": 4.80,
    "USD_SOFR_OIS[2Y]": 4.40,
    "USD_SOFR_OIS[5Y]": 4.10,
    "USD_SOFR_OIS[10Y]": 4.00,
    # ---- USD_LIBOR_3M  (LABEL, PERCENT) ----
    "USD_LIBOR_3M[ON]": 5.45,
    "USD_LIBOR_3M[1W]": 5.46,
    "USD_LIBOR_3M[1M]": 5.48,
    "USD_LIBOR_3M[3M]": 5.40,
    "USD_LIBOR_3M[6M]": 5.25,
    "USD_LIBOR_3M[1Y]": 4.95,
    "USD_LIBOR_3M[2Y]": 4.55,
    "USD_LIBOR_3M[5Y]": 4.25,
    "USD_LIBOR_3M[10Y]": 4.15,
    # ---- USD_CSA_EUR  (LABEL, PERCENT) ----
    "USD_CSA_EUR[ON]": 5.35,
    "USD_CSA_EUR[1W]": 5.36,
    "USD_CSA_EUR[1M]": 5.37,
    "USD_CSA_EUR[3M]": 5.30,
    "USD_CSA_EUR[6M]": 5.15,
    "USD_CSA_EUR[1Y]": 4.85,
    "USD_CSA_EUR[2Y]": 4.45,
    "USD_CSA_EUR[5Y]": 4.15,
    "USD_CSA_EUR[10Y]": 4.05,
    # ---- EUR_ESTR_OIS  (LABEL, PERCENT) ----
    "EUR_ESTR_OIS[ON]": 3.90,
    "EUR_ESTR_OIS[1W]": 3.90,
    "EUR_ESTR_OIS[1M]": 3.91,
    "EUR_ESTR_OIS[3M]": 3.85,
    "EUR_ESTR_OIS[6M]": 3.70,
    "EUR_ESTR_OIS[1Y]": 3.40,
    "EUR_ESTR_OIS[2Y]": 3.00,
    "EUR_ESTR_OIS[5Y]": 2.70,
    "EUR_ESTR_OIS[10Y]": 2.60,
    # ---- EUR_EURIBOR_6M  (LABEL, PERCENT) ----
    "EUR_EURIBOR_6M[ON]": 4.05,
    "EUR_EURIBOR_6M[1W]": 4.05,
    "EUR_EURIBOR_6M[1M]": 4.06,
    "EUR_EURIBOR_6M[3M]": 4.00,
    "EUR_EURIBOR_6M[6M]": 3.85,
    "EUR_EURIBOR_6M[1Y]": 3.55,
    "EUR_EURIBOR_6M[2Y]": 3.15,
    "EUR_EURIBOR_6M[5Y]": 2.85,
    "EUR_EURIBOR_6M[10Y]": 2.75,
    # ---- GBP_SONIA_OIS  (LABEL, PERCENT) ----
    "GBP_SONIA_OIS[ON]": 5.15,
    "GBP_SONIA_OIS[1W]": 5.15,
    "GBP_SONIA_OIS[1M]": 5.16,
    "GBP_SONIA_OIS[3M]": 5.10,
    "GBP_SONIA_OIS[6M]": 4.95,
    "GBP_SONIA_OIS[1Y]": 4.60,
    "GBP_SONIA_OIS[2Y]": 4.20,
    "GBP_SONIA_OIS[5Y]": 3.90,
    "GBP_SONIA_OIS[10Y]": 3.80,
    # ---- CHF_SARON_OIS  (LABEL, PERCENT) ----
    "CHF_SARON_OIS[ON]": 1.70,
    "CHF_SARON_OIS[1W]": 1.70,
    "CHF_SARON_OIS[1M]": 1.71,
    "CHF_SARON_OIS[3M]": 1.65,
    "CHF_SARON_OIS[6M]": 1.50,
    "CHF_SARON_OIS[1Y]": 1.25,
    "CHF_SARON_OIS[2Y]": 1.00,
    "CHF_SARON_OIS[5Y]": 0.85,
    "CHF_SARON_OIS[10Y]": 0.80,
    # ---- AUD_AONIA_OIS  (LABEL, PERCENT) ----
    "AUD_AONIA_OIS[ON]": 4.25,
    "AUD_AONIA_OIS[1W]": 4.25,
    "AUD_AONIA_OIS[1M]": 4.26,
    "AUD_AONIA_OIS[3M]": 4.20,
    "AUD_AONIA_OIS[6M]": 4.05,
    "AUD_AONIA_OIS[1Y]": 3.80,
    "AUD_AONIA_OIS[2Y]": 3.50,
    "AUD_AONIA_OIS[5Y]": 3.30,
    "AUD_AONIA_OIS[10Y]": 3.20,
    # ---- BRENT_CRUDE_FWD  (OFFSET_M, DECIMAL) ----
    "BRENT_CRUDE_FWD[M+0]": 82.50,
    "BRENT_CRUDE_FWD[M+1]": 82.00,
    "BRENT_CRUDE_FWD[M+2]": 81.60,
    "BRENT_CRUDE_FWD[M+3]": 81.10,
    "BRENT_CRUDE_FWD[M+6]": 79.50,
    "BRENT_CRUDE_FWD[M+12]": 76.00,
    # ---- WTI_CRUDE_FWD  (OFFSET_M, DECIMAL) ----
    "WTI_CRUDE_FWD[M+0]": 78.00,
    "WTI_CRUDE_FWD[M+1]": 77.50,
    "WTI_CRUDE_FWD[M+2]": 77.10,
    "WTI_CRUDE_FWD[M+3]": 76.60,
    "WTI_CRUDE_FWD[M+6]": 75.00,
    "WTI_CRUDE_FWD[M+12]": 71.50,
    # ---- HENRY_HUB_GAS_FWD  (OFFSET_M, DECIMAL) ----
    "HENRY_HUB_GAS_FWD[M+0]": 2.50,
    "HENRY_HUB_GAS_FWD[M+1]": 2.65,
    "HENRY_HUB_GAS_FWD[M+2]": 2.75,
    "HENRY_HUB_GAS_FWD[M+3]": 2.80,
    "HENRY_HUB_GAS_FWD[M+6]": 3.10,
    "HENRY_HUB_GAS_FWD[M+12]": 3.30,
    # ---- CO2_EUA_ANNUAL_FWD  (OFFSET_Y, DECIMAL) ----
    "CO2_EUA_ANNUAL_FWD[Y+0]": 65.50,
    "CO2_EUA_ANNUAL_FWD[Y+1]": 68.00,
    "CO2_EUA_ANNUAL_FWD[Y+2]": 71.20,
    "CO2_EUA_ANNUAL_FWD[Y+3]": 74.50,
    # ---- POWER_CH_BASELOAD_HY  (OFFSET_HY, DECIMAL) ----
    "POWER_CH_BASELOAD_HY[HY+0]": 95.00,
    "POWER_CH_BASELOAD_HY[HY+1]": 98.50,
    "POWER_CH_BASELOAD_HY[HY+2]": 102.00,
    "POWER_CH_BASELOAD_HY[HY+3]": 105.50,
}


# ============================================================================
# 3. SURFACES  (60 pillar levels: 3 surfaces × 4 expiries × 5 y-labels)
# ============================================================================
# Keys are `f"{surface_name}{x_tenor_token}[{y_label}]"`. y_label is the
# bare numeric stored in surface_point_meta — no `STRIKE=` /
# `MONEYNESS=` prefix. All vol values are DECIMAL (0.30 = 30% vol).
#
# BRENT_CRUDE_VOL_SURF — (OFFSET_M, STRIKE). Strikes are absolute price
# levels in USD/bbl. Smile centred near front-month Brent (~82.50).
_BRENT_VOL_LEVELS: Final[dict[str, float]] = {
    # [M+0]
    "BRENT_CRUDE_VOL_SURF[M+0][50.00]": 0.42,
    "BRENT_CRUDE_VOL_SURF[M+0][65.00]": 0.36,
    "BRENT_CRUDE_VOL_SURF[M+0][75.00]": 0.32,
    "BRENT_CRUDE_VOL_SURF[M+0][85.00]": 0.30,
    "BRENT_CRUDE_VOL_SURF[M+0][100.00]": 0.34,
    # [M+1]
    "BRENT_CRUDE_VOL_SURF[M+1][50.00]": 0.40,
    "BRENT_CRUDE_VOL_SURF[M+1][65.00]": 0.34,
    "BRENT_CRUDE_VOL_SURF[M+1][75.00]": 0.31,
    "BRENT_CRUDE_VOL_SURF[M+1][85.00]": 0.29,
    "BRENT_CRUDE_VOL_SURF[M+1][100.00]": 0.33,
    # [M+3]
    "BRENT_CRUDE_VOL_SURF[M+3][50.00]": 0.38,
    "BRENT_CRUDE_VOL_SURF[M+3][65.00]": 0.33,
    "BRENT_CRUDE_VOL_SURF[M+3][75.00]": 0.30,
    "BRENT_CRUDE_VOL_SURF[M+3][85.00]": 0.28,
    "BRENT_CRUDE_VOL_SURF[M+3][100.00]": 0.32,
    # [M+6]
    "BRENT_CRUDE_VOL_SURF[M+6][50.00]": 0.36,
    "BRENT_CRUDE_VOL_SURF[M+6][65.00]": 0.31,
    "BRENT_CRUDE_VOL_SURF[M+6][75.00]": 0.28,
    "BRENT_CRUDE_VOL_SURF[M+6][85.00]": 0.27,
    "BRENT_CRUDE_VOL_SURF[M+6][100.00]": 0.30,
}

# USDEUR_VOL_SURF — (LABEL, DELTA_CALL). Y-axis is call delta:
# 0.10 = deep OTM call, 0.50 = ATM, 0.90 = deep ITM call.
# REJECTED by the runtime builder today (calibration not implemented).
_USDEUR_VOL_LEVELS: Final[dict[str, float]] = {
    # [1W]
    "USDEUR_VOL_SURF[1W][0.10]": 0.085,
    "USDEUR_VOL_SURF[1W][0.25]": 0.080,
    "USDEUR_VOL_SURF[1W][0.50]": 0.076,
    "USDEUR_VOL_SURF[1W][0.75]": 0.079,
    "USDEUR_VOL_SURF[1W][0.90]": 0.084,
    # [1M]
    "USDEUR_VOL_SURF[1M][0.10]": 0.090,
    "USDEUR_VOL_SURF[1M][0.25]": 0.084,
    "USDEUR_VOL_SURF[1M][0.50]": 0.080,
    "USDEUR_VOL_SURF[1M][0.75]": 0.083,
    "USDEUR_VOL_SURF[1M][0.90]": 0.088,
    # [3M]
    "USDEUR_VOL_SURF[3M][0.10]": 0.094,
    "USDEUR_VOL_SURF[3M][0.25]": 0.088,
    "USDEUR_VOL_SURF[3M][0.50]": 0.084,
    "USDEUR_VOL_SURF[3M][0.75]": 0.087,
    "USDEUR_VOL_SURF[3M][0.90]": 0.092,
    # [1Y]
    "USDEUR_VOL_SURF[1Y][0.10]": 0.100,
    "USDEUR_VOL_SURF[1Y][0.25]": 0.094,
    "USDEUR_VOL_SURF[1Y][0.50]": 0.090,
    "USDEUR_VOL_SURF[1Y][0.75]": 0.093,
    "USDEUR_VOL_SURF[1Y][0.90]": 0.098,
}

# AAPL_VOL_SURF — (LABEL, LOG_MONEYNESS_FWD). Y-axis is ln(K/F).
# Negative = OTM put; positive = OTM call.
_AAPL_VOL_LEVELS: Final[dict[str, float]] = {
    # [1M]
    "AAPL_VOL_SURF[1M][-0.20]": 0.28,
    "AAPL_VOL_SURF[1M][-0.10]": 0.24,
    "AAPL_VOL_SURF[1M][0.00]": 0.22,
    "AAPL_VOL_SURF[1M][0.10]": 0.23,
    "AAPL_VOL_SURF[1M][0.20]": 0.26,
    # [3M]
    "AAPL_VOL_SURF[3M][-0.20]": 0.29,
    "AAPL_VOL_SURF[3M][-0.10]": 0.25,
    "AAPL_VOL_SURF[3M][0.00]": 0.23,
    "AAPL_VOL_SURF[3M][0.10]": 0.24,
    "AAPL_VOL_SURF[3M][0.20]": 0.27,
    # [6M]
    "AAPL_VOL_SURF[6M][-0.20]": 0.30,
    "AAPL_VOL_SURF[6M][-0.10]": 0.26,
    "AAPL_VOL_SURF[6M][0.00]": 0.24,
    "AAPL_VOL_SURF[6M][0.10]": 0.25,
    "AAPL_VOL_SURF[6M][0.20]": 0.28,
    # [1Y]
    "AAPL_VOL_SURF[1Y][-0.20]": 0.31,
    "AAPL_VOL_SURF[1Y][-0.10]": 0.27,
    "AAPL_VOL_SURF[1Y][0.00]": 0.25,
    "AAPL_VOL_SURF[1Y][0.10]": 0.26,
    "AAPL_VOL_SURF[1Y][0.20]": 0.29,
}

_SURFACE_LEVELS: Final[dict[str, float]] = {
    **_BRENT_VOL_LEVELS,
    **_USDEUR_VOL_LEVELS,
    **_AAPL_VOL_LEVELS,
}


# ============================================================================
# Public exports
# ============================================================================
# Primary contract: flat `(rel_name, level)` dict. The snapshot compiler
# consumes this directly. Production loaders should emit the same shape.
CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT: Final[dict[str, float]] = {
    **_NODE_LEVELS,
    **_CURVE_LEVELS,
    **_SURFACE_LEVELS,
}

# Secondary view: nested by topology. For human inspection / pprint only;
# downstream code MUST depend on the flat dict above.
CANONICAL_MARKET_DATA_SNAPSHOT_NESTED_DICT: Final[dict[str, dict[str, float]]] = {
    "nodes": _NODE_LEVELS,
    "curves": _CURVE_LEVELS,
    "surfaces": _SURFACE_LEVELS,
}


# ============================================================================
# Cross-validation against the static-config canonical fixture
# ============================================================================


def _build_expected_curve_rel_names() -> set[str]:
    """Reconstruct the curve-point rel-name keyset from the static config."""
    curve_id_to_name = dict(
        zip(
            range(100_000_000, 100_000_000 + len(CANONICAL_CURVE_NAMES)),
            CANONICAL_CURVE_NAMES,
            strict=True,
        )
    )
    return {
        f"{curve_id_to_name[int(cid)]}{token}"
        for cid, token in zip(
            CANONICAL_CURVE_POINT_META_DATA["curve_id"],
            CANONICAL_CURVE_POINT_META_DATA["tenor_token"],
            strict=True,
        )
    }


def _build_expected_surface_rel_names() -> set[str]:
    """Reconstruct the surface-point rel-name keyset from the static config."""
    surface_id_to_name = dict(
        zip(
            range(200_000_000, 200_000_000 + len(CANONICAL_SURFACE_NAMES)),
            CANONICAL_SURFACE_NAMES,
            strict=True,
        )
    )
    return {
        f"{surface_id_to_name[int(sid)]}{x_tok}[{y_lab}]"
        for sid, x_tok, y_lab in zip(
            CANONICAL_SURFACE_POINT_META_DATA["surface_id"],
            CANONICAL_SURFACE_POINT_META_DATA["x_tenor_token"],
            CANONICAL_SURFACE_POINT_META_DATA["y_label"],
            strict=True,
        )
    }


def _validate_against_static_config() -> None:
    # Nodes.
    fixture_nodes = set(_NODE_LEVELS)
    config_nodes = set(CANONICAL_NODE_NAMES)
    if fixture_nodes != config_nodes:
        raise ValueError(
            "canonical_market_data_snapshot.nodes drift: "
            f"missing={sorted(config_nodes - fixture_nodes)}, "
            f"extra={sorted(fixture_nodes - config_nodes)}"
        )

    # Curves.
    fixture_curves = set(_CURVE_LEVELS)
    expected_curves = _build_expected_curve_rel_names()
    if fixture_curves != expected_curves:
        raise ValueError(
            "canonical_market_data_snapshot.curves drift: "
            f"missing={sorted(expected_curves - fixture_curves)}, "
            f"extra={sorted(fixture_curves - expected_curves)}"
        )

    # Surfaces.
    fixture_surfaces = set(_SURFACE_LEVELS)
    expected_surfaces = _build_expected_surface_rel_names()
    if fixture_surfaces != expected_surfaces:
        raise ValueError(
            "canonical_market_data_snapshot.surfaces drift: "
            f"missing={sorted(expected_surfaces - fixture_surfaces)}, "
            f"extra={sorted(fixture_surfaces - expected_surfaces)}"
        )


_validate_against_static_config()


# ============================================================================
# Diagnostics
# ============================================================================


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_dict,
        print_subheader,
        print_success_message,
        print_text,
    )

    print_calling_file()

    print_text(
        f"Canonical market-data snapshot: "
        f"{len(_NODE_LEVELS)} node + {len(_CURVE_LEVELS)} curve-point + "
        f"{len(_SURFACE_LEVELS)} surface-point levels "
        f"({len(CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT)} total)."
    )

    print_subheader(title="NODES")
    print_dict(_NODE_LEVELS)

    print_subheader(title="CURVES")
    print_dict(_CURVE_LEVELS)

    print_subheader(title="SURFACES")
    print_dict(_SURFACE_LEVELS)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
