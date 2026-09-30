# src/riskquant/portfolio/ingest/utils.py
import numpy as np


def is_na_helper(x: object) -> bool:
    """
    Check if a scalar value is missing or invalid (None, NaN, ±inf).

    ``pd.NA`` is intentionally absent: ingest now operates on Polars frames
    which use numpy-extracted float NaN rather than the pandas NA singleton.
    """
    if x is None:
        return True
    if isinstance(x, (float, np.floating)):
        return not np.isfinite(x)
    return False
