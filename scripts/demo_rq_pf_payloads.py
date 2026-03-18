import numpy as np

from riskquant.constants import NO_FX_RF_REQUIRED
from riskquant.portfolio.payloads.forward import (
    ForwardCurvePayload,
    ForwardNodePayload,
)
from riskquant.portfolio.payloads.spot import SpotPayload
from riskquant.portfolio.payloads.vanilla_european_option import (
    VanillaEuropeanOptionCurvePayload,
    VanillaEuropeanOptionNodePayload,
)
from riskquant.utils.printer import print_success_message


if __name__ == "__main__":

    # --- SPOT PAYLOAD EXAMPLE ---

    # Small usage example showing construction and printing of a SpotPayload.
    # Example values chosen for illustration only.
    spot_payload = SpotPayload(
        price_rf_id=np.int32(101),
        quantity=np.float64(1_000.0),
        fx_rf_id=NO_FX_RF_REQUIRED,
    )
    print("\nExample SpotPayload:")
    print(spot_payload)

    # --- FORWARD PAYLOAD EXAMPLE ---

    # Example instantiation of ForwardNodePayload for testing and demonstration
    fwd_node_payload = ForwardNodePayload(
        quantity=np.float64(1_000.0),
        strike=np.float64(1.2345),
        settlement_date_ordinal32=np.int32(19_100),
        price_rf_id=np.int32(101),
        fx_under_to_settle_rf_id=NO_FX_RF_REQUIRED,
        fx_strike_to_settle_rf_id=NO_FX_RF_REQUIRED,
    )
    print("\nExample ForwardNodePayload:")
    print(fwd_node_payload)

    # Example instantiation of ForwardCurvePayload for testing and demonstration
    fwd_curve_payload = ForwardCurvePayload(
        quantity=np.float64(1_000.0),
        strike=np.float64(1.2345),
        delivery_date_ordinal32=np.int32(19_150),
        settlement_date_ordinal32=np.int32(19_100),
        price_curve_id=np.int32(202),
        fx_under_to_settle_rf_id=NO_FX_RF_REQUIRED,
        fx_strike_to_settle_rf_id=NO_FX_RF_REQUIRED,
    )
    print("\nExample ForwardCurvePayload:")
    print(fwd_curve_payload)

    # --- VANILLA OPTION PAYLOAD EXAMPLE ---

    # Example instantiation of VanillaEuropeanOptionNodePayload for testing and demonstration
    vanilla_option_payload = VanillaEuropeanOptionNodePayload(
        quantity=np.float64(1_000.0),
        strike=np.float64(1.2345),
        is_call=np.int8(1),
        expiry_date_ordinal32=np.int32(19_200),
        payoff_settlement_date_ordinal32=np.int32(19_300),
        underlying_price_rf_id=np.int32(101),
        vol_rf_id=np.int32(102),
        fx_under_to_settle_rf_id=NO_FX_RF_REQUIRED,
        fx_strike_to_settle_rf_id=NO_FX_RF_REQUIRED,
    )
    print("\nExample VanillaEuropeanOptionNodePayload:")
    print(vanilla_option_payload)

    # Example instantiation of VanillaEuropeanOptionCurvePayload for testing and demonstration
    vanilla_option_curve_payload = VanillaEuropeanOptionCurvePayload(
        quantity=np.float64(1_000.0),
        strike=np.float64(1.2345),
        is_call=np.int8(1),
        expiry_date_ordinal32=np.int32(19_200),
        payoff_settlement_date_ordinal32=np.int32(19_300),
        delivery_date_ordinal32=np.int32(19_500),
        underlying_price_curve_rf_id=np.int32(201),
        vol_surface_rf_id=np.int32(202),
        fx_under_to_settle_rf_id=NO_FX_RF_REQUIRED,
        fx_strike_to_settle_rf_id=NO_FX_RF_REQUIRED,
    )
    print("\nExample VanillaEuropeanOptionCurvePayload:")
    print(vanilla_option_curve_payload)

    # --- SUCCESSFUL RUN INDICATOR ---

    # Print
    print_success_message()
