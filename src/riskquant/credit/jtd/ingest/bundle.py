# src/riskquant/credit/jtd/ingest/bundle.py
"""
The mixed JTD ingest master.

`JtdIngestBundle` is the input-side analogue of `CompiledJtdPortfolio`: one
optional per-model authoring frame per JTD model, plus the shared long-format
multi-factor loadings relation. A "mixed portfolio" is this collection of
tightly-typed per-model masters — never a single wide union frame. The per-model
field order matches `CompiledJtdPortfolio`'s block band order so input and output
stay aligned.

Every frame is a ``pl.DataFrame`` (WS_CREDIT_POLARS C.7, D094 form): a slot that
holds any other object raises TypeError at construction, before any column is
read. Polars frames are immutable, so the bundle never changes a caller's frame
(D095 form).
"""

from __future__ import annotations

__all__ = [
    "JtdIngestBundle",
]

from dataclasses import dataclass, fields
from typing import Final

import polars as pl

from riskquant.credit.jtd.models import JtdModel
from riskquant.credit.jtd.registry import model_field_order

# Canonical (model -> bundle attribute) pairing, in compiled-portfolio band
# order. Single source = the model registry; the self-validation guards below
# confirm the derived attrs exist on this dataclass.
_MODEL_FIELD_ORDER: Final[tuple[tuple[JtdModel, str], ...]] = model_field_order()


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class JtdIngestBundle:
    """Mixed JTD ingest master: optional per-model frames + shared loadings."""

    forced_default_df: pl.DataFrame | None = None
    independent_bernoulli_df: pl.DataFrame | None = None
    beta_binomial_df: pl.DataFrame | None = None
    stress_regime_bernoulli_df: pl.DataFrame | None = None
    sf_gaussian_copula_df: pl.DataFrame | None = None
    sf_students_t_copula_df: pl.DataFrame | None = None
    mf_gaussian_copula_df: pl.DataFrame | None = None
    mf_students_t_copula_df: pl.DataFrame | None = None
    marshall_olkin_df: pl.DataFrame | None = None

    # Shared long-format relation (obligor_uid, factor_name, factor_loading),
    # consumed by every multi-factor model. NOT a per-model frame.
    mf_loadings_df: pl.DataFrame | None = None

    def __post_init__(self) -> None:
        # D094 form: refuse a non-Polars slot at construction, so no validator
        # or builder ever reads a pandas frame or a LazyFrame.
        for f in fields(self):
            _require_frame_or_none(
                getattr(self, f.name), where=f"JtdIngestBundle.{f.name}"
            )

    def frames_by_model(self) -> dict[JtdModel, pl.DataFrame]:
        """Populated per-model frames keyed by model, in band order.

        Excludes `mf_loadings_df` (a relation, not a per-model frame).
        """
        out: dict[JtdModel, pl.DataFrame] = {}
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
        frame: pl.DataFrame,
        *,
        instrument_col: str = "instrument",
        mf_loadings_df: pl.DataFrame | None = None,
    ) -> JtdIngestBundle:
        """Authoring convenience: split a denormalized frame that carries an
        ``instrument`` column (a ``JtdModel`` member name per row) into the
        per-model slots, dropping the now-redundant ``instrument`` column.

        This is NOT a compile path — ``compile_jtd_portfolio`` only accepts a
        bundle. It exists for callers (demos, tests, programmatically generated
        mixed portfolios) that find it natural to emit one frame and normalize
        it here. The canonical authoring form is still the per-model slots.
        Unknown instrument names fail fast. Each slot keeps the row order of
        ``frame``.

        Raises
        ------
        TypeError
            ``frame`` is not a ``pl.DataFrame``, or ``mf_loadings_df`` is
            neither None nor a ``pl.DataFrame``; raised before any column is
            read.
        ValueError
            ``frame`` has no ``instrument_col``; an unknown instrument name.
        """
        _require_frame_or_none(frame, where="from_instrument_frame: frame")
        if frame is None:
            raise TypeError(
                "from_instrument_frame: frame must be a pl.DataFrame (polars."
                "DataFrame); got None."
            )
        _require_frame_or_none(
            mf_loadings_df, where="from_instrument_frame: mf_loadings_df"
        )
        if instrument_col not in frame.columns:
            raise ValueError(
                f"from_instrument_frame: frame has no '{instrument_col}' column."
            )
        name_to_attr = {model.name: attr for model, attr in _MODEL_FIELD_ORDER}
        # str() of each distinct value, as the pandas astype(str) path did: a
        # null becomes "None" and is reported as an unknown instrument.
        present = {str(v) for v in frame.get_column(instrument_col).unique().to_list()}
        unknown = present - set(name_to_attr)
        if unknown:
            raise ValueError(
                f"from_instrument_frame: unknown instrument(s) {sorted(unknown)}."
            )
        kwargs: dict[str, pl.DataFrame] = {}
        for name, attr in name_to_attr.items():
            if name not in present:
                continue
            # filter keeps the row order of `frame` (A_INVENTORY section 4).
            sub = frame.filter(pl.col(instrument_col).cast(pl.String) == name)
            kwargs[attr] = sub.drop(instrument_col)
        return cls(mf_loadings_df=mf_loadings_df, **kwargs)


def _require_frame_or_none(obj: object, *, where: str) -> None:
    """Raise TypeError unless ``obj`` is None or a ``pl.DataFrame`` (D094 form)."""
    if obj is not None and not isinstance(obj, pl.DataFrame):
        kind = f"{type(obj).__module__}.{type(obj).__qualname__}"
        raise TypeError(
            f"{where} must be a pl.DataFrame (polars.DataFrame) or None; got "
            f"{kind}. Convert at the call boundary (pl.from_pandas, pl.from_dict); "
            "collect a pl.LazyFrame first."
        )


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
