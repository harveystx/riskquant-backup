# src/riskquant/market_data/risk_factors/builder_utils.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _validate_mapping_dict(m: Mapping[str, Any]) -> None:
    """Validate a small static mapping of risk factor name -> integer id.

    Checks performed:
      - keys must be non-empty strings
      - values must be integer-like (reject bool, float, str)
      - no two keys map to the same integer id

    This helper is optimized for configuration-time validation (small dicts),
    not for per-row lookups in tight loops.
    """
    # Empty/blank key detection (single pass)
    for k in m.keys():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"Mapping contains invalid/blank key: {k!r}")

    # Ensure values are integer-like and detect duplicates by tracking seen ids.
    seen: dict[int, str] = {}
    for k, v in m.items():
        # `bool` is a subclass of int in Python; explicitly reject it here because
        # booleans are not valid factor ids.
        if isinstance(v, (bool,)):
            raise ValueError(
                f"Mapping value for '{k}' must be an int id, got bool={v!r}"
            )

        # Accept Python ints and NumPy integer scalars; reject other types.
        if not isinstance(v, (int, np.integer)):
            raise ValueError(
                f"Mapping value for '{k}' must be an int id, got {type(v).__name__}={v!r}"
            )

        iv = int(v)
        prev = seen.get(iv)
        if prev is not None:
            # Clear, actionable error message when multiple names map to same id.
            raise ValueError(
                "Mapping contains duplicate ID assignment:\n"
                f"  '{prev}' and '{k}' both map to ID={iv}\n"
                "Each risk factor must have a unique risk_factor_name_id."
            )
        seen[iv] = k


def _require_mapping_key(
    m: Mapping[str, Any],
    key: str,
    *,
    preview_n: int = 25,
) -> Any:
    """Lookup `key` in mapping `m` and raise a detailed error if missing.

    This helper provides user-friendly diagnostics on configuration errors by
    showing a short preview of known keys and a suggested fix. Designed to be
    inexpensive on the success path (single hash lookup); the preview list is
    materialized only on error.

    Parameters
    ----------
    m : Mapping[str, Any]
        Mapping from risk factor name to id.
    key : str
        Key to lookup (column label converted to string prior to lookup).
    preview_n : int
        Maximum number of known keys to show in the error message.
    """
    try:
        return m[key]
    except KeyError:
        # Materialize a small preview of keys only when diagnosing a failure --
        # prevents huge error messages for very large config maps.
        keys = list(m.keys())
        n = len(keys)
        preview = keys[:preview_n]
        preview_str = ", ".join(preview) + (" ..." if n > preview_n else "")

        raise ValueError(
            "Risk factor ID mapping lookup failed.\n"
            f"Missing key: '{key}'\n\n"
            "Fix:\n"
            "  Add this risk factor to the STATIC config table that defines risk_factor_name_id.\n"
            "  Example row to add:\n"
            f"    risk_factor_name = {key}\n"
            "    risk_factor_name_id = <new unique integer>\n\n"
            f"Known keys preview ({min(preview_n, n)} of {n}):\n"
            f"  {preview_str}\n"
        ) from None
