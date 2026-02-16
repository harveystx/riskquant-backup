# src/riskquant/portfolio/constants.py

"""
Common portfolio sentinel constants.

"""

import numpy as np

# --- SENTINELS FOR "MISSING" IDS (payload fields stay numeric) ---
NO_ID: np.int32 = np.int32(-1)  # missing ID


# --- SENTINEL FOR "NO UIX" IN BLOCKS ---
NO_UIX: np.int32 = np.int32(-1)


# FX sentinel: indicates that no FX risk factor is required for a conversion
# (i.e., underlying_ccy == settlement_ccy or no FX conversion applies).
NO_FX_RF_REQUIRED: np.int32 = np.int32(-1)

# Sentinel for missing delivery date encoded as an int32 ordinal (days since DATE_EPOCH). Delivery dates are used in the forward block for economic exposure timing but are not required for discounting, which is anchored to settlement dates. This sentinel indicates that no delivery date is specified or relevant for the position. Sometimes price_rf_id may refere to a specific risk factor where the price is not dependent on a delivery date, such as a spot price or a forward curve point with no specific delivery date. In other cases, the price_rf_id that is itself an entire curve/surface/cube and the delivery date must be used to identify the specific point on the curve/surface/cube that defines the economic exposure. In these cases, the delivery date is required and must be specified as an ordinal32. The NO_DELIVERY_DATE_ORDINAL32 sentinel is used to indicate that no delivery date is specified or relevant for the position, which can occur when price_rf_id references a risk factor where the price is not dependent on a delivery date.
NO_DELIVERY_DATE_ORDINAL32: np.int32 = np.int32(-1)

# Sentinel for missing settlement/payment date encoded as an int32 ordinal
# (days since DATE_EPOCH). Used by ForwardPayload as `settlement_date_ordinal32`.
NO_SETTLEMENT_DATE_ORDINAL32: np.int32 = np.int32(-1)

# Sentinel for missing option expiry date encoded as an int32 ordinal.
NO_EXPIRY_DATE_ORDINAL32: np.int32 = np.int32(-1)
