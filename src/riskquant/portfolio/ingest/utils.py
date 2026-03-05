# src/riskquant/portfolio/ingest/utils.py

import numpy as np
import pandas as pd


def is_na_helper(x: object) -> bool:
    """Check if a scalar value is missing or invalid (None, NaN, pd.NA, ±inf)."""
    if x is None:
        # None is considered missing.
        return True

    if x is pd.NA:
        # pd.NA is considered missing.
        return True

    if isinstance(x, (float, np.floating)):
        # Check for NaN or infinite values.
        # np.isfinite returns False for NaN and ±inf, so we return True for those cases.
        # NaNs and infinite values are considered missing/invalid.
        return not np.isfinite(x)

    # For all other types (e.g., int, str), we consider them valid and not missing.
    return False
