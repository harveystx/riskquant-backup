# src\riskquant\portfolio\ingest\fx_resolver.py
import numpy as np

from riskquant.exceptions import PrettyKeyError
from riskquant.portfolio.constants import NO_FX_RF_REQUIRED
from riskquant.portfolio.fx import get_fx_pair_for_under_to_settle_conversion


def resolve_fx_rf_ids_inplace(
    *,
    base_ccy: np.ndarray,  # (n,) object/str
    settlement_ccy: np.ndarray,  # (n,) object/str
    out_fx_rf_ids: np.ndarray,  # (n,) int32
    rf_name_to_rf_id: dict[str, int],
) -> None:
    """
    Resolve FX RF IDs for rows where base_ccy != settlement_ccy.

    out_fx_rf_ids is filled with FX_NONE then overwritten for quanto rows.
    """
    # Initialize all FX RF IDs to NO_FX_RF_REQUIRED, which indicates no FX conversion needed.
    out_fx_rf_ids.fill(np.int32(NO_FX_RF_REQUIRED))

    # Get a boolean mask for rows where base_ccy != settlement_ccy, since those are the only rows that require FX conversion.
    mask = base_ccy != settlement_ccy

    # Early exit if there are no rows that require FX conversion.
    if not np.any(mask):
        return

    # Get the indices of the rows that require FX conversion.
    qidx = np.nonzero(mask)[0]

    # Local bindings for efficiency in the loop
    fx_pair_fn = get_fx_pair_for_under_to_settle_conversion
    rf_name_to_rf_id_map = rf_name_to_rf_id

    # Loop over the rows that require FX conversion and resolve the FX RF ID for each one.
    for i in qidx:
        # Get the FX pair for this row based on the base_ccy and settlement_ccy.
        fx_pair = fx_pair_fn(
            underlying_ccy=str(base_ccy[i]),
            settlement_ccy=str(settlement_ccy[i]),
        )

        # If the FX pair is an empty string, it means that no FX conversion is needed, so we can leave the FX RF ID as NO_FX_RF_REQUIRED.
        if fx_pair == "":
            out_fx_rf_ids[i] = np.int32(NO_FX_RF_REQUIRED)
            continue

        # Now look up the FX RF ID for this FX pair in the rf_name_to_rf_id mapping.
        try:
            out_fx_rf_ids[i] = np.int32(rf_name_to_rf_id_map[fx_pair])
        except KeyError as e:
            raise PrettyKeyError(
                "Missing FX spot risk factor for currency conversion:\n"
                f"- fx_pair={fx_pair!r}\n"
                f"- base_ccy={base_ccy[i]!r}\n"
                f"- settlement_ccy={settlement_ccy[i]!r}\n"
            ) from e
