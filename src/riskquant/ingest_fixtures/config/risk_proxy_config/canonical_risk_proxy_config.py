# src/riskquant/ingest_fixtures/config/risk_proxy_config/canonical_risk_proxy_config.py

from __future__ import annotations

import pandas as pd

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


CANONICAL_RISK_PROXY_CONFIG_DF = pd.DataFrame(CANONICAL_RISK_PROXY_CONFIG_DATA)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical Risk Proxy Config Data:")
    print_dict(CANONICAL_RISK_PROXY_CONFIG_DATA)

    print("\nAs DataFrame:")
    print_df(CANONICAL_RISK_PROXY_CONFIG_DF, index=False, float_format="{:.2f}")

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
