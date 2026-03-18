# src/riskquant/portfolio/ingest/fx_resolver.py
import numpy as np
import pandas as pd

from riskquant.exceptions import PrettyKeyError
from riskquant.constants import NO_FX_RF_REQUIRED


def resolve_fx_rf_ids_inplace(
    *,
    base_ccy: np.ndarray,  # (n,) object/str
    settlement_ccy: np.ndarray,  # (n,) object/str
    out_fx_rf_ids: np.ndarray,  # (n,) int32
    out_fx_invert: np.ndarray,  # (n,) int32
    rf_name_to_rf_id: dict[str, int],
) -> None:
    """
    Vectorized resolution of FX RF IDs for rows where base_ccy != settlement_ccy.
    Completely eliminates Python for-loops and regex string matching.
    """
    # Initialize default states (No FX required, not inverted)
    out_fx_rf_ids.fill(np.int32(NO_FX_RF_REQUIRED))
    out_fx_invert.fill(0)

    # Get a boolean mask for rows where base_ccy != settlement_ccy
    mask = base_ccy != settlement_ccy

    # Early exit if there are no rows that require FX conversion.
    if not np.any(mask):
        return

    # Extract strings for mismatched rows
    b_str = base_ccy[mask].astype(str)
    s_str = settlement_ccy[mask].astype(str)

    # Fast vectorized string concatenation for currency pairs
    std_pairs = np.char.add(b_str, s_str)
    inv_pairs = np.char.add(s_str, b_str)

    # Map directly via Pandas (C-level hash lookup)
    std_mapped = pd.Series(std_pairs).map(rf_name_to_rf_id).to_numpy()
    inv_mapped = pd.Series(inv_pairs).map(rf_name_to_rf_id).to_numpy()

    # Identify which mapping succeeded
    std_valid = ~np.isnan(std_mapped)
    inv_valid = ~np.isnan(inv_mapped)

    # Find failures
    missing_mask = ~(std_valid | inv_valid)
    if np.any(missing_mask):
        bad_b = b_str[missing_mask][0]
        bad_s = s_str[missing_mask][0]
        raise PrettyKeyError(
            "Missing FX spot risk factor for currency conversion:\n"
            f"- Searched for standard: '{bad_b}{bad_s}'\n"
            f"- Searched for inverted: '{bad_s}{bad_b}'\n"
            f"- base_ccy='{bad_b}'\n"
            f"- settlement_ccy='{bad_s}'\n"
        )

    # Use standard where valid, else fallback to inverted
    final_ids = np.where(std_valid, std_mapped, inv_mapped).astype(np.int32)
    final_inverts = np.where(std_valid, 0, 1).astype(np.int32)

    # Slot back into output arrays
    out_fx_rf_ids[mask] = final_ids
    out_fx_invert[mask] = final_inverts
