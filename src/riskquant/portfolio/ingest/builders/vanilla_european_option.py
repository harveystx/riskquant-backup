"""
Module for building vanilla European option trade records from ingested batch data.

This module provides functionality to construct TradeRecord objects for
vanilla European option transactions, extracting position data from an
IngestBatch and validating all required fields (strike price, call/put flag,
expiry date, strike currency, and volatility risk factor) before packaging
the data into standardized trade record format.
"""

from __future__ import annotations

import numpy as np

# Import static configuration containing risk factor ID mappings
from riskquant.config.deprecated_configs import RiskStaticConfig

# Import custom exception for pretty error formatting
from riskquant.exceptions import PrettyKeyError

# Import sentinel constant for detecting missing expiry dates
from riskquant.portfolio.constants import NO_EXPIRY_DATE_ORDINAL32

# Import the VanillaEuropeanOptionPayload data structure for option trade metadata
from riskquant.portfolio.payloads.vanilla_european_option import (
    VanillaEuropeanOptionPayload,
)

# Import the TradeRecord class for creating standardized trade objects
from riskquant.portfolio.trades import TradeRecord

# Import IngestBatch for type hinting and batch data structure
from riskquant.portfolio.ingest.precomputer import IngestBatch

# Import helper function to check for missing/NA values in batch data
from riskquant.portfolio.ingest.parser import _is_na_helper as is_na


def build_vanilla_european_option(
    *,
    ix: int,
    batch: IngestBatch,
    cfg: RiskStaticConfig,
) -> TradeRecord:
    """
    Build a vanilla European option trade record from batch ingestion data with comprehensive validation.

    Constructs a TradeRecord object for a vanilla European option transaction
    by extracting position data from a precompiled IngestBatch at a specific
    index. Performs extensive validation of all required fields including
    strike price, call/put flag, expiry date, strike currency, and volatility
    risk factor (via static config lookup) before wrapping the data in a
    VanillaEuropeanOptionPayload.

    Args:
        ix: Index into the batch arrays to identify the specific position
           within the batch to process.
        b: IngestBatch object containing precomputed arrays of trade data,
           including UIDs, prices, quantities, strikes, expiry dates, call/put
           flags, strike currencies, and FX conversion rates.
        cfg: RiskStaticConfig object containing the mapping of risk factor
             names to their IDs. Used to look up the implied volatility
             risk factor ID using the naming convention vol_<underlying_name>.

    Returns:
        TradeRecord: A complete trade record for the vanilla European option
                     position containing the unique identifier, instrument ID,
                     and payload with underlying price, quantity, strike,
                     call/put flag, implied volatility, expiry date, and FX
                     conversion rate information.

    Raises:
        PrettyKeyError: If any required field is missing or invalid:
                       - Strike price is missing or NA
                       - is_call flag is missing or NA
                       - Expiry date is missing or set to NO_EXPIRY_DATE
                       - Strike currency is missing or empty
                       - Implied volatility risk factor not found in config
                         (uses naming convention vol_<underlying_risk_factor_name>)
    """
    # Extract the unique identifier (uid) for this position by indexing into the batch UIDs array
    uid = int(batch.uids[ix])

    # Extract the strike price from the batch by row index using .iat[] for positional access
    K = batch.strike_price.iat[ix]

    # Validate that the strike price is present; raise error if missing or NA
    if is_na(K):
        raise PrettyKeyError(
            "VANILLA_EUROPEAN_OPTION row missing strike_price:\n"
            f"- uid={uid}\n"
            f"- underlying={batch.rf_names.iat[ix]!r}\n"
        )

    # Extract the is_call flag from the batch indicating whether option is a call (True) or put (False)
    ic = batch.is_call.iat[ix]

    # Validate that the is_call flag is present; raise error if missing or NA
    if is_na(ic):
        raise PrettyKeyError(
            "VANILLA_EUROPEAN_OPTION row missing is_call:\n"
            f"- uid={uid}\n"
            f"- underlying={batch.rf_names.iat[ix]!r}\n"
        )

    # Convert is_call boolean to numpy int8 for compact storage (1 for call, 0 for put)
    ic8 = np.int8(int(ic))

    # Validate that the expiry date is set (not the sentinel NO_EXPIRY_DATE value)
    if int(batch.expiry_ord[ix]) == int(NO_EXPIRY_DATE_ORDINAL32):
        raise PrettyKeyError(
            "VANILLA_EUROPEAN_OPTION row missing expiry_date:\n"
            f"- uid={uid}\n"
            f"- underlying={batch.rf_names.iat[ix]!r}\n"
        )

    # Extract the strike currency (in effective terms), trim whitespace, and convert to uppercase
    vsc = str(batch.strike_ccy_eff[ix]).strip().upper()

    # Validate that the strike currency is present (non-empty string)
    if not vsc:
        raise PrettyKeyError(
            "VANILLA_EUROPEAN_OPTION row missing strike_ccy:\n"
            f"- uid={uid}\n"
            f"- underlying={batch.rf_names.iat[ix]!r}\n"
        )

    # Extract the underlying risk factor name from the batch
    under_name = str(batch.rf_names.iat[ix])

    # Construct the expected implied volatility risk factor name using the convention vol_<underlying_name>
    vol_name = f"vol_{under_name}"

    # Look up the implied volatility risk factor ID from the static config using the constructed name
    try:
        vol_rf_id = np.int32(cfg.rf_name_to_id[vol_name])
    except KeyError as e:
        # Raise a descriptive error if the volatility risk factor is not found in config
        raise PrettyKeyError(
            "Missing implied volatility risk factor for vanilla option:\n"
            f"- uid={uid}\n"
            f"- underlying={under_name!r}\n"
            f"- expected_vol_rf_name={vol_name!r}\n"
            "Convention: vol_<underlying_risk_factor_name>\n"
        ) from e

    # Construct and return a TradeRecord with all required vanilla European option trade information
    return TradeRecord(
        # Set the unique trade identifier from the extracted uid
        trade_uid=uid,
        # Set the instrument identifier from the batch's opt_id (common for all options in batch)
        instrument_id=batch.opt_id,
        # Create a VanillaEuropeanOptionPayload containing all option-specific data
        payload=VanillaEuropeanOptionPayload(
            # Store the underlying asset price risk factor ID as a numpy int32 from the batch array
            underlying_price_rf_id=np.int32(batch.price_rf_ids[ix]),
            # Store the option quantity (contract count) as a numpy float64 from the batch array
            quantity=np.float64(batch.qty[ix]),
            # Store the validated strike price as a numpy float64
            strike=np.float64(K),
            # Store the validated option type (1 for call, 0 for put) as a numpy int8
            is_call=ic8,
            # Store the implied volatility risk factor ID looked up from the static config
            vol_rf_id=vol_rf_id,
            # Store the expiry date as a numpy int32 ordinal (days since epoch)
            expiry_date_ordinal32=np.int32(batch.expiry_ord[ix]),
            # Store the FX underlying-to-settlement risk factor ID as a numpy int32 from the batch array
            fx_under_to_settle_rf_id=np.int32(batch.fx_under_to_settle[ix]),
            # Store the FX strike-to-settlement risk factor ID as a numpy int32 from the batch array
            fx_strike_to_settle_rf_id=np.int32(batch.fx_strike_to_settle[ix]),
        ),
    )
