# src/riskquant/credit/jtd/ingest/builders/marshall_olkin.py
from __future__ import annotations

import numpy as np
import pandas as pd

from riskquant.credit.jtd.blocks.marshall_olkin import MarshallOlkinJtdBlock
from riskquant.credit.jtd.ingest._shared import (
    _f64,
    _i64,
    _join_pd_lgd_from_master,
    _resolve_factor_uix,
)
from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit

_PREVIEW = 10


def build_marshall_olkin_block(
    df: pd.DataFrame,
    obligor_master: CompiledObligorCredit,
    factor_name_to_uix: dict[str, int],
) -> MarshallOlkinJtdBlock:

    obligor_uids = _i64(df, "uid")

    # Grab the baseline total PD and LGD from the single source of truth
    pd_total, lgd = _join_pd_lgd_from_master(
        obligor_uids=obligor_uids,
        obligor_master=obligor_master,
    )

    systemic_pd = _f64(df, col="systemic_pd")

    # -------------------------------------------------------------------------
    # PD Decoupling Algebra
    # -------------------------------------------------------------------------
    # The Marshall-Olkin model triggers default if EITHER the idiosyncratic
    # OR the systemic shock occurs.
    # Survival requires surviving both independent risks:
    # (1 - PD_total) = (1 - PD_idio) * (1 - PD_sys)

    # 1. Algebraic Guardrail
    # A firm cannot have a total PD lower than the probability of the systemic
    # shock it is exposed to.
    bad = pd_total < systemic_pd
    if bad.any():
        offending = obligor_uids[bad][:_PREVIEW].tolist()
        raise ValueError(
            f"MARSHALL_OLKIN: Total PD from obligor master is strictly less than "
            f"systemic_pd for obligor(s) {offending}. Survival decoupling requires "
            f"pd_total >= systemic_pd."
        )

    # 2. Decompose the total PD into its idiosyncratic component, which is what the
    # model will use to trigger default in the absence of a systemic shock.
    pd_idio_raw = 1.0 - ((1.0 - pd_total) / (1.0 - systemic_pd))
    pd_idio = np.ascontiguousarray(pd_idio_raw, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Factor Resolution
    # -------------------------------------------------------------------------
    uniform_uix = _resolve_factor_uix(
        df["uniform_factor"].to_numpy(copy=False),
        factor_name_to_uix,
        ctx="MARSHALL_OLKIN.uniform_factor",
    )

    systemic_uix = _resolve_factor_uix(
        df["systemic_uniform_factor"].to_numpy(copy=False),
        factor_name_to_uix,
        ctx="MARSHALL_OLKIN.systemic_uniform_factor",
    )

    # -------------------------------------------------------------------------
    # Build the Marshall-Olkin block
    # -------------------------------------------------------------------------

    return MarshallOlkinJtdBlock(
        obligor_uids=obligor_uids,
        pd_idio=pd_idio,
        pd_sys=systemic_pd,
        lgd=lgd,
        uniform_uix=uniform_uix,
        systemic_uix=systemic_uix,
    )
