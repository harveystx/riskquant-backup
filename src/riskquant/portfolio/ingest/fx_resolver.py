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
    # Initialize all FX RF IDs to FX_NONE, which indicates no FX conversion needed. We will overwrite this for any rows that require FX conversion (i.e. where base_ccy != settlement_ccy). This ensures that any rows that do not require FX conversion will have the correct FX RF ID of FX_NONE, and we only need to focus on resolving the FX RF IDs for the rows that do require conversion.This also means that if there are any errors in resolving the FX RF IDs for the rows that dorequire conversion, we will still have a valid FX RF ID of FX_NONE for the rows that donot require conversion, which is a safe default that indicates no FX risk factor should be applied.
    out_fx_rf_ids.fill(np.int32(NO_FX_RF_REQUIRED))

    # Get a boolean mask for rows where base_ccy != settlement_ccy, since those are the only rows that require FX conversion. We can skip all the rows where the base and settlement currency are the same, since they do not require any FX conversion and should have an FX RF ID of FX_NONE. This allows us to focus our efforts on resolving the FX RF IDs for the rows that actually require conversion, which is more efficient than trying to resolve FX RF IDs for all rows regardless of whether they need it or not. We can compute this mask using a simple elementwise comparison of the base_ccy and settlement_ccyarrays. This will give us a boolean array where True indicates rows that require FX conversion and False indicatesrows that do not. We can then use this mask to index into the arrays and only process the
    mask = base_ccy != settlement_ccy

    # Early exit if there are no rows that require FX conversion, since the rest of the code assumes at least one row to process.
    if not np.any(mask):
        return

    # Get the indices of the rows that require FX conversion, since we will need to iterate over these rows to resolve the FX RF IDs. We can use np.nonzero(mask)[0] to get the indices of the True values in the mask, which correspond to the rows that require FX conversion. This will give us an array of integer indices that we can use to index into the base_ccy and settlement_ccy arrays to get the relevant currency pairs for each row that requires FX conversion. We will then use these currency pairs to look up the corresponding FX RF IDs in the rf_name_to_rf_id mapping, and fill in the out_fx_rf_ids array with the resolved FX RF IDs for these rows.
    qidx = np.nonzero(mask)[0]

    # Local bindings for efficiency in the loop
    fx_pair_fn = get_fx_pair_for_under_to_settle_conversion
    rf_name_to_rf_id_map = rf_name_to_rf_id

    # Loop over the rows that require FX conversion and resolve the FX RF ID for each one. For each row that requires FX conversion, we will determine the relevant FX pair based on the base_ccy and settlement_ccy for that row, and then look up the corresponding FX RF ID in the rf_name_to_rf_id mapping. If we cannot find a valid FX pair or if the FX pair is not present in the rf_name_to_rf_id mapping, we will raise an error with a helpful message indicating the row index and the offending currency pair. If we successfully resolve the FX RF ID for a row, we will fill it into the out_fx_rf_ids array at the corresponding index.
    for i in qidx:
        # Get the FX pair for this row based on the base_ccy and settlement_ccy. We will use the get_fx_pair_for_under_to_settle_conversion function to determine the correct FX pair for converting from the base currency to the settlement currency. This function will return a string representing the FX pair in a standardized format (e.g. "EURUSD" for converting from EUR to USD). If the base and settlement currencies are the same, this function will return an empty string, but we have already filtered those out with our mask, so we should not encounter that case here. If we do encounter an empty string, it would indicate a bug in our logic, so we will treat it as an error case and raise an error with a helpful message indicating the row index and the offending currency pair.
        fx_pair = fx_pair_fn(
            underlying_ccy=str(base_ccy[i]),
            settlement_ccy=str(settlement_ccy[i]),
        )

        # fx_pair_fn returns "" when equal, but mask excludes equals; still safe:
        if not fx_pair:
            out_fx_rf_ids[i] = np.int32(NO_FX_RF_REQUIRED)
            continue

        # Now look up the FX RF ID for this FX pair in the rf_name_to_rf_id mapping. If we cannot find a valid FX RF ID for this FX pair, it means we are missing a necessary FX spot risk factor for performing the currency conversion, which is an error since we need this to properly account for the FX risk in our portfolio. In this case, we will raise an error with a helpful message indicating the row index and the offending currency pair, as well as the base and settlement currencies for additional context. If we do find a valid FX RF ID, we will fill it into the out_fx_rf_ids array at the corresponding index for this row.
        try:
            out_fx_rf_ids[i] = np.int32(rf_name_to_rf_id_map[fx_pair])
        except KeyError as e:
            raise PrettyKeyError(
                "Missing FX spot risk factor for currency conversion:\n"
                f"- fx_pair={fx_pair!r}\n"
                f"- base_ccy={base_ccy[i]!r}\n"
                f"- settlement_ccy={settlement_ccy[i]!r}\n"
            ) from e
