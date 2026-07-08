# src/riskquant/credit/exposure/add_on/ingest.py
"""
Supervisory add-on ingest (the edge).

`compile_add_on_factors` turns a config-authored ``subclass -> factor`` table into
a validated, UIX-sorted `CompiledAddOnFactors`. Fail-fast on anything that would
corrupt the lookup or the exposure: missing columns, non-finite or negative
factors, duplicate subclasses.
"""

from __future__ import annotations

__all__ = [
    "compile_add_on_factors",
]

import numpy as np
import pandas as pd

from riskquant.credit.exposure.add_on.compiled import CompiledAddOnFactors


def compile_add_on_factors(
    df: pd.DataFrame,
    *,
    subclass_col: str = "subclass_uix",
    factor_col: str = "factor",
) -> CompiledAddOnFactors:
    """Compile a subclass + factor table into a validated `CompiledAddOnFactors`.

    Rows are sorted by subclass here, so callers need not pre-sort.
    """
    for col in (subclass_col, factor_col):
        if col not in df.columns:
            raise ValueError(
                f"compile_add_on_factors: missing required column '{col}'."
            )

    subclass = np.ascontiguousarray(df[subclass_col].to_numpy(dtype=np.int32))
    factor = df[factor_col].to_numpy(dtype=np.float64)

    if not np.isfinite(factor).all():
        raise ValueError(
            f"compile_add_on_factors: non-finite value in factor column '{factor_col}'."
        )
    if (factor < 0.0).any():
        raise ValueError(
            f"compile_add_on_factors: negative add-on factor in '{factor_col}' "
            "(supervisory factors must be >= 0)."
        )
    if subclass.shape[0] != np.unique(subclass).shape[0]:
        raise ValueError("compile_add_on_factors: duplicate subclass_uix.")

    order = np.argsort(subclass, kind="stable")
    return CompiledAddOnFactors(
        subclass_uix=np.ascontiguousarray(subclass[order]),
        factor=np.ascontiguousarray(factor[order]),
    )
