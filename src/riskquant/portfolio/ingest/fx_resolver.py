# src\riskquant\portfolio\ingest\fx_resolver.py
import numpy as np

from riskquant.exceptions import PrettyKeyError
from riskquant.constants import NO_FX_RF_REQUIRED
from riskquant.portfolio.fx import get_fx_pair_for_under_to_settle_conversion


def resolve_fx_rf_ids_inplace(
    *,
    base_ccy: np.ndarray,  # (n,) object/str
    settlement_ccy: np.ndarray,  # (n,) object/str
    out_fx_rf_ids: np.ndarray,  # (n,) int32
    out_fx_invert: np.ndarray,  # (n,) int32
    rf_name_to_rf_id: dict[str, int],
) -> None:
    """
    Resolve FX RF IDs for rows where base_ccy != settlement_ccy.

    out_fx_rf_ids is filled with NO_FX_RF_REQUIRED then overwritten for quanto rows.
    out_fx_invert is filled with 0, and set to 1 if the inverted FX pair is used.
    """
    # Initialize default states (No FX required, not inverted)
    out_fx_rf_ids.fill(np.int32(NO_FX_RF_REQUIRED))
    out_fx_invert.fill(0)

    # Get a boolean mask for rows where base_ccy != settlement_ccy
    mask = base_ccy != settlement_ccy

    # Early exit if there are no rows that require FX conversion.
    if not np.any(mask):
        return

    # Get the indices of the rows that require FX conversion.
    qidx = np.nonzero(mask)[0]

    # Local bindings for efficiency in the loop
    fx_pair_fn = get_fx_pair_for_under_to_settle_conversion
    rf_name_to_rf_id_map = rf_name_to_rf_id

    # Loop over the rows that require FX conversion and resolve the FX RF ID
    for i in qidx:
        b_ccy_str = str(base_ccy[i])
        s_ccy_str = str(settlement_ccy[i])

        # Get the FX pair for this row
        fx_pair_std = fx_pair_fn(
            underlying_ccy=b_ccy_str,
            settlement_ccy=s_ccy_str,
        )

        # If the FX pair is an empty string, it means no FX conversion is needed.
        if fx_pair_std == "":
            continue

        # 1. Try the standard FX pair first
        if fx_pair_std in rf_name_to_rf_id_map:
            out_fx_rf_ids[i] = np.int32(rf_name_to_rf_id_map[fx_pair_std])
            continue

        # 2. Try the inverted FX pair
        # Assuming your standard nomenclature is BaseQuote (e.g., EURUSD)
        fx_pair_inv = f"{s_ccy_str}{b_ccy_str}"
        if fx_pair_inv in rf_name_to_rf_id_map:
            out_fx_rf_ids[i] = np.int32(rf_name_to_rf_id_map[fx_pair_inv])
            out_fx_invert[i] = (
                1  # Flag this row for mathematical inversion later
            )
            continue

        # 3. Neither pair exists in the static config
        raise PrettyKeyError(
            "Missing FX spot risk factor for currency conversion:\n"
            f"- Searched for standard: {fx_pair_std!r}\n"
            f"- Searched for inverted: {fx_pair_inv!r}\n"
            f"- base_ccy={b_ccy_str!r}\n"
            f"- settlement_ccy={s_ccy_str!r}\n"
        )
