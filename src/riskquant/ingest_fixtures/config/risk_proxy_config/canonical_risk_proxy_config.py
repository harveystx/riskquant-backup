# src/riskquant/ingest_fixtures/config/risk_proxy_config/canonical_risk_proxy_config.py

from __future__ import annotations

import polars as pl

from riskquant.ingest_fixtures.schemas import RiskProxyConfigData
from riskquant.ingest_fixtures.utils import validate_column_lengths

# All names referenced below exist in the canonical static_config fixture
# (see ingest_fixtures/config/risk_static_config/_parts/*).
#
# Single-hop policy (enforced by validate_proxy_maps):
#   - No proxy_name appears as a required_name anywhere in this table.
#   - No required_name proxies itself.
#   - No container in container-level rows has a child rf in rf-level rows.
#
# Container-level rows (3):
#   WTI_CRUDE_FWD          -> BRENT_CRUDE_FWD     (oil-on-oil, scale 0.95)
#   HENRY_HUB_GAS_FWD      -> BRENT_CRUDE_FWD     (gas-on-oil, scale 1.40, illustrative)
#   POWER_CH_BASELOAD_HY   -> CO2_EUA_ANNUAL_FWD  (illustrative, scale 1.20)
#
# RF-level rows (3):
#   USDGBP                 -> USDEUR              (G10 cross via parent, scale 0.85)
#   EURCHF                 -> USDEUR              (low-vol cross via parent, scale 0.60)
#   USD_LIBOR_3M[10Y]      -> USD_SOFR_OIS[10Y]   (tenor-basis substitution, scale 1.00)
#
# Cross-level check: WTI_CRUDE_FWD, HENRY_HUB_GAS_FWD, POWER_CH_BASELOAD_HY
# are container-proxied. None of their child rf names appear as
# rf-level required_names below — no conflict.
#
# Coverage note (mirrors the static_config fixture's documented
# roll-method gap): all 3 container-level rows proxy CURVE containers;
# no SURFACE container is proxied here, and no surface-point rf_id
# appears as an rf-level required_name. Both are genuinely supported by
# the resolver (see audit.py's kind_label mapping, which includes
# SURFACE and SURFACE_POINT) but are not exercised by this fixture. See
# demos/config/proxy_config/demo_walkthrough.py for standalone examples
# of both, and
# tests/ingest_fixtures/config/risk_proxy_config/test_canonical_fixture.py
# ::test_documented_proxy_coverage_gap for the tracked, executable pin.


def _get_data() -> RiskProxyConfigData:
    return {
        "proxying_level": [
            "CONTAINER",
            "CONTAINER",
            "CONTAINER",
            "RISK_FACTOR",
            "RISK_FACTOR",
            "RISK_FACTOR",
        ],
        "required_name": [
            "WTI_CRUDE_FWD",
            "HENRY_HUB_GAS_FWD",
            "POWER_CH_BASELOAD_HY",
            "USDGBP",
            "EURCHF",
            "USD_LIBOR_3M[10Y]",
        ],
        "proxy_name": [
            "BRENT_CRUDE_FWD",
            "BRENT_CRUDE_FWD",
            "CO2_EUA_ANNUAL_FWD",
            "USDEUR",
            "USDEUR",
            "USD_SOFR_OIS[10Y]",
        ],
        "vol_scaling_factor": [
            0.95,
            1.40,
            1.20,
            0.85,
            0.60,
            1.00,
        ],
    }


CANONICAL_RISK_PROXY_CONFIG_DATA: RiskProxyConfigData = _get_data()


validate_column_lengths(CANONICAL_RISK_PROXY_CONFIG_DATA)


# Polars, matching riskquant.config.proxy_config's compile pipeline
# (riskquant.config.proxy_config.compile.compile_proxy_topology_from_df
# takes a pl.DataFrame, not pandas -- see RQ-AUD-018).
CANONICAL_RISK_PROXY_CONFIG_DF: pl.DataFrame = pl.DataFrame(
    CANONICAL_RISK_PROXY_CONFIG_DATA
)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical Risk Proxy Config Data:")
    print_dict(CANONICAL_RISK_PROXY_CONFIG_DATA)

    print("\nAs DataFrame:")
    print(CANONICAL_RISK_PROXY_CONFIG_DF)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    from pathlib import Path

    import xlsxwriter

    _debug_print()

    # --- SAVE TO EXCEL FOR MANUAL INSPECTION ---

    # Path to data folder is relative to the root of the repository.
    fp = Path(
        Path(__file__).resolve().parent.parent.parent.parent.parent.parent,
        "data",
        "canonical_examples",
        "canonical_proxy_config.xlsx",
    )
    fp.parent.mkdir(parents=True, exist_ok=True)

    print(fp)
    # Save to a single-sheet Excel workbook for manual inspection, if
    # desired, via Polars' write_excel (see
    # ingest_fixtures/config/risk_static_config/canonical_risk_static_config.py
    # for the equivalent multi-sheet pattern, and RQ-AUD-018 for why this
    # is Polars rather than pandas.ExcelWriter).
    workbook = xlsxwriter.Workbook(str(fp))
    CANONICAL_RISK_PROXY_CONFIG_DF.write_excel(
        workbook=workbook, worksheet="proxy_config"
    )
    workbook.close()
