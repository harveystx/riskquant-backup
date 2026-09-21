# src/riskquant/credit/jtd/ingest/bundle.py
"""
The mixed JTD ingest master.

`JtdIngestBundle` is the input-side analogue of `CompiledJtdPortfolio`: one
optional per-model authoring frame per JTD model, plus the shared long-format
multi-factor loadings relation. A "mixed portfolio" is this collection of
tightly-typed per-model masters — never a single wide union frame. The per-model
field order matches `CompiledJtdPortfolio`'s block band order so input and output
stay aligned.
"""

from __future__ import annotations

__all__ = [
    "JtdIngestBundle",
]

from dataclasses import dataclass, fields
from typing import Final

import pandas as pd

from riskquant.credit.jtd.models import JtdModel
from riskquant.credit.jtd.registry import model_field_order

# Canonical (model -> bundle attribute) pairing, in compiled-portfolio band
# order. Single source = the model registry; the self-validation guards below
# confirm the derived attrs exist on this dataclass.
_MODEL_FIELD_ORDER: Final[tuple[tuple[JtdModel, str], ...]] = model_field_order()


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class JtdIngestBundle:
    """Mixed JTD ingest master: optional per-model frames + shared loadings."""

    forced_default_df: pd.DataFrame | None = None
    independent_bernoulli_df: pd.DataFrame | None = None
    beta_binomial_df: pd.DataFrame | None = None
    stress_regime_bernoulli_df: pd.DataFrame | None = None
    sf_gaussian_copula_df: pd.DataFrame | None = None
    sf_students_t_copula_df: pd.DataFrame | None = None
    mf_gaussian_copula_df: pd.DataFrame | None = None
    mf_students_t_copula_df: pd.DataFrame | None = None
    marshall_olkin_df: pd.DataFrame | None = None

    # Shared long-format relation (obligor_uid, factor_name, factor_loading),
    # consumed by every multi-factor model. NOT a per-model frame.
    mf_loadings_df: pd.DataFrame | None = None

    def frames_by_model(self) -> dict[JtdModel, pd.DataFrame]:
        """Populated per-model frames keyed by model, in band order.

        Excludes `mf_loadings_df` (a relation, not a per-model frame).
        """
        out: dict[JtdModel, pd.DataFrame] = {}
        for model, attr in _MODEL_FIELD_ORDER:
            frame = getattr(self, attr)
            if frame is not None:
                out[model] = frame
        return out

    def is_empty(self) -> bool:
        """True if no per-model frame is populated (loadings alone is empty)."""
        return not self.frames_by_model()

    @classmethod
    def from_instrument_frame(
        cls,
        frame: pd.DataFrame,
        *,
        instrument_col: str = "instrument",
        mf_loadings_df: pd.DataFrame | None = None,
    ) -> JtdIngestBundle:
        """Authoring convenience: split a denormalized frame that carries an
        ``instrument`` column (a ``JtdModel`` member name per row) into the
        per-model slots, dropping the now-redundant ``instrument`` column.

        This is NOT a compile path — ``compile_jtd_portfolio`` only accepts a
        bundle. It exists for callers (demos, tests, programmatically generated
        mixed portfolios) that find it natural to emit one frame and normalize
        it here. The canonical authoring form is still the per-model slots.
        Unknown instrument names fail fast.
        """
        if instrument_col not in frame.columns:
            raise ValueError(
                f"from_instrument_frame: frame has no '{instrument_col}' column."
            )
        name_to_attr = {model.name: attr for model, attr in _MODEL_FIELD_ORDER}
        present = set(frame[instrument_col].astype(str).unique())
        unknown = present - set(name_to_attr)
        if unknown:
            raise ValueError(
                f"from_instrument_frame: unknown instrument(s) {sorted(unknown)}."
            )
        kwargs: dict[str, pd.DataFrame] = {}
        for name, attr in name_to_attr.items():
            sub = frame[frame[instrument_col] == name]
            if not sub.empty:
                kwargs[attr] = sub.drop(columns=[instrument_col]).reset_index(drop=True)
        return cls(mf_loadings_df=mf_loadings_df, **kwargs)


# Guard: every JTD model must have a bundle slot (mirrors the contract guard).
_covered = {m for m, _ in _MODEL_FIELD_ORDER}
_missing = [m.name for m in JtdModel if m not in _covered]
if _missing:
    raise RuntimeError(
        f"JtdIngestBundle has no frame slot for JTD model(s): {_missing}. "
        "Add them to _MODEL_FIELD_ORDER in riskqaunt/credit/jtd/ingest/bundle.py."
    )

# Guard: the per-model attribute names actually exist on the dataclass.
_attrs = {f.name for f in fields(JtdIngestBundle)}
_bad = [attr for _, attr in _MODEL_FIELD_ORDER if attr not in _attrs]
if _bad:
    raise RuntimeError(
        f"_MODEL_FIELD_ORDER references unknown JtdIngestBundle field(s): {_bad}."
    )
