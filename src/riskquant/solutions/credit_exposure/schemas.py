# src/riskquant/solutions/credit_exposure/schemas.py
"""
Credit-exposure request/result contract (Pydantic, the human edge).

Locks the two boundaries the pipeline hinges on, as types:

  CONTRACT A (input)   trades + netting sets + add-on table + method choice
  CONTRACT B (output)  per-obligor EAD (obligor_uid -> EAD) that feeds the JTD credit
                       layer, plus the netting-set detail behind it.

Validation mirrors ``solutions.credit_var``: a bad field raises a ``ValidationError``
whose ``loc`` points into ``trades`` / ``netting_sets`` / ``add_on_factors`` so a
front-end can map it straight to a cell. Method-specific requirements (CEM needs
the add-on table + priced-notional inputs) are enforced against the catalog.
"""

from __future__ import annotations

from typing import Any, Literal, cast

__all__ = [
    "AddOnFactorRow",
    "CreditExposureRequest",
    "CreditExposureResult",
    "ExposureProvenance",
    "HedgingSetRhoRow",
    "NettingSetExposureRow",
    "NettingSetRow",
    "ObligorExposure",
    "SubclassHedgingSetRow",
    "TradeRow",
]

import math
from collections.abc import Mapping

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from riskquant.solutions.credit_exposure.catalog import (
    EXPOSURE_MODEL_CATALOG,
    exposure_model_entry,
)

_STRICT = ConfigDict(extra="forbid", protected_namespaces=())


class TradeRow(BaseModel):
    """One trade. ``mtm`` is a valuation output injected here (not priced)."""

    model_config = _STRICT

    trade_uid: int
    netting_set_uid: int
    mtm: float = Field(description="Current mark-to-market value (valuation output).")
    # Priced-notional inputs — required only for methods with `requires_add_on`.
    volume: float | None = Field(default=None, ge=0.0)
    current_price: float | None = Field(default=None, ge=0.0)
    asset_subclass_uix: int | None = Field(default=None)
    # SA-CCR structural trade-level inputs (P2-11A/B). Optional on the shared
    # schema so CEM / replacement cost are unaffected; the SA-CCR boundary
    # requires them per-trade and fails fast when absent (see _validate_method).
    direction: Literal["long", "short"] | None = Field(
        default=None,
        description="SA-CCR trade direction; supervisory delta = +1 long / -1 short.",
    )
    remaining_maturity_years: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "SA-CCR remaining maturity M in years (> 0). The unmargined CRE52 "
            "maturity factor is sqrt(clamp(M, 10/250, 1))."
        ),
    )


class NettingSetRow(BaseModel):
    """One netting set: its obligor and its net collateral (variation margin)."""

    model_config = _STRICT

    netting_set_uid: int
    obligor_uid: int
    net_collateral: float = Field(default=0.0, ge=0.0)


class AddOnFactorRow(BaseModel):
    """One supervisory add-on factor, keyed by asset-subclass UIX (house-defined)."""

    model_config = _STRICT

    subclass_uix: int
    factor: float = Field(ge=0.0)


class SubclassHedgingSetRow(BaseModel):
    """Maps an asset subclass to its SA-CCR hedging set (house-defined)."""

    model_config = _STRICT

    subclass_uix: int
    hedging_set_uix: int


class HedgingSetRhoRow(BaseModel):
    """Per-hedging-set supervisory correlation (house-calibrated)."""

    model_config = _STRICT

    hedging_set_uix: int
    rho: float = Field(ge=0.0, le=1.0)


class CreditExposureRequest(BaseModel):
    """A single credit-exposure run, as a JSON payload (Contract A)."""

    model_config = _STRICT

    model: str = Field(
        description="Exposure method key, e.g. 'current_exposure_method'."
    )
    trades: list[TradeRow] = Field(min_length=1)
    netting_sets: list[NettingSetRow] = Field(min_length=1)
    add_on_factors: list[AddOnFactorRow] | None = Field(
        default=None,
        description="Supervisory add-on table for CEM; omit for replacement cost.",
    )
    pfe_floor: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="CEM NGR-compression floor; default 0.4 when omitted.",
    )
    subclass_hedging_sets: list[SubclassHedgingSetRow] | None = Field(
        default=None,
        description="SA-CCR subclass -> hedging-set map; required for 'saccr'.",
    )
    hedging_set_correlations: list[HedgingSetRhoRow] | None = Field(
        default=None,
        description="SA-CCR per-hedging-set correlation rho; required for 'saccr'.",
    )
    saccr_alpha: float | None = Field(
        default=None,
        gt=0.0,
        description="SA-CCR alpha in EAD = alpha*(RC+PFE); default 1.4 when omitted.",
    )
    saccr_multiplier_floor: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="SA-CCR PFE-multiplier floor; default 0.05 when omitted.",
    )
    pfe_coefficient: float | None = Field(
        default=None,
        ge=0.0,
        description="CEM NGR-compression slope; default 0.6 when omitted.",
    )

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @model_validator(mode="after")
    def _validate_method(self) -> CreditExposureRequest:
        if self.model not in EXPOSURE_MODEL_CATALOG:
            raise ValueError(
                f"unknown exposure model '{self.model}'. "
                f"Supported: {sorted(EXPOSURE_MODEL_CATALOG)}."
            )
        entry = exposure_model_entry(self.model)
        if entry.requires_add_on:
            if not self.add_on_factors:
                raise ValueError(
                    f"model '{self.model}' requires 'add_on_factors' (the supervisory "
                    "add-on table)."
                )
            for i, t in enumerate(self.trades):
                for field in ("volume", "current_price", "asset_subclass_uix"):
                    if getattr(t, field) is None:
                        raise ValueError(
                            f"trade {i} (trade_uid={t.trade_uid}) missing '{field}', required by "
                            f"model '{self.model}'."
                        )
        if entry.requires_saccr_config:
            if not self.subclass_hedging_sets or not self.hedging_set_correlations:
                raise ValueError(
                    f"model '{self.model}' requires 'subclass_hedging_sets' and "
                    "'hedging_set_correlations' (the SA-CCR hedging-set config)."
                )
            # P2-11A/B: direction and remaining maturity are structural SA-CCR
            # inputs, required per trade. No silent default to long / MF=1.
            for i, t in enumerate(self.trades):
                for field in ("direction", "remaining_maturity_years"):
                    if getattr(t, field) is None:
                        raise ValueError(
                            f"trade {i} (trade_uid={t.trade_uid}) missing "
                            f"'{field}', required by model '{self.model}' "
                            "(SA-CCR supervisory delta / maturity factor)."
                        )
        return self

    @classmethod
    def from_dataframe(
        cls,
        trades: pl.DataFrame,
        netting_sets: pl.DataFrame,
        *,
        model: str,
        add_on_factors: pl.DataFrame | None = None,
        pfe_floor: float | None = None,
        pfe_coefficient: float | None = None,
        subclass_hedging_sets: pl.DataFrame | None = None,
        hedging_set_correlations: pl.DataFrame | None = None,
        saccr_alpha: float | None = None,
        saccr_multiplier_floor: float | None = None,
    ) -> CreditExposureRequest:
        """Build a request from flat ``pl.DataFrame`` tables (notebook / desk ergonomics).

        ``trades`` needs ``trade_uid, netting_set_uid, mtm`` (plus ``volume,
        current_price, asset_subclass_uix`` for CEM, and additionally
        ``direction`` in {"long","short"} and ``remaining_maturity_years`` > 0
        for SA-CCR); ``netting_sets`` needs ``netting_set_uid, obligor_uid,
        net_collateral``; ``add_on_factors`` needs ``subclass_uix, factor``;
        ``subclass_hedging_sets`` needs ``subclass_uix, hedging_set_uix``;
        ``hedging_set_correlations`` needs ``hedging_set_uix, rho``. Extra
        columns are ignored.

        Column admission by dtype (WS_CREDIT_POLARS C.12, the C.11 rule of each
        field type). An int field admits an integer dtype, or a float dtype: an
        integral float cell (``7.0``) becomes its int; a fractional, NaN or
        infinite cell reaches the int field unchanged and raises
        ``pydantic.ValidationError`` (``int(7.9)`` would give a uid that the
        author did not write). A float field admits an integer or a float
        dtype. ``direction`` admits ``pl.Utf8``. An optional field also admits
        ``pl.Null``, and its null cell (in a float column: null or NaN) means
        absent. A null cell in a required field reaches the row model as
        ``None`` and raises ``pydantic.ValidationError`` at that field. Any
        other dtype (``pl.Boolean``, ``pl.Decimal`` included) raises TypeError
        that names the column.

        Raises
        ------
        TypeError
            A table is not a ``pl.DataFrame`` (all tables are checked before
            any read), or a read column has a dtype that its field rule does not
            admit.
        ValueError
            A required column is missing.
        pydantic.ValidationError
            A cell or the request fails the request rules.
        """
        tables: dict[str, object] = {
            "trades": trades,
            "netting_sets": netting_sets,
            "add_on_factors": add_on_factors,
            "subclass_hedging_sets": subclass_hedging_sets,
            "hedging_set_correlations": hedging_set_correlations,
        }
        for name, table in tables.items():
            optional = name not in ("trades", "netting_sets")
            if table is None and optional:
                continue
            if not isinstance(table, pl.DataFrame):
                kind = f"{type(table).__module__}.{type(table).__qualname__}"
                raise TypeError(
                    f"from_dataframe: '{name}' must be a pl.DataFrame "
                    f"(polars.DataFrame); got {kind}. Convert at the call "
                    "boundary (pl.from_pandas, pl.from_dict)."
                )

        trade_cols = _read(
            trades,
            "trades",
            required={"trade_uid": int, "netting_set_uid": int, "mtm": float},
            optional={
                "volume": float,
                "current_price": float,
                "asset_subclass_uix": int,
                "direction": str,
                "remaining_maturity_years": float,
            },
        )
        trade_rows = [
            TradeRow(**cast("dict[str, Any]", cells)) for cells in _records(trade_cols)
        ]
        ns_cols = _read(
            netting_sets,
            "netting_sets",
            required={
                "netting_set_uid": int,
                "obligor_uid": int,
                "net_collateral": float,
            },
        )
        ns_rows = [
            NettingSetRow(**cast("dict[str, Any]", cells))
            for cells in _records(ns_cols)
        ]
        addon_rows: list[AddOnFactorRow] | None = None
        if add_on_factors is not None:
            cols = _read(
                add_on_factors,
                "add_on_factors",
                required={"subclass_uix": int, "factor": float},
            )
            addon_rows = [
                AddOnFactorRow(**cast("dict[str, Any]", cells))
                for cells in _records(cols)
            ]
        hedging_rows: list[SubclassHedgingSetRow] | None = None
        if subclass_hedging_sets is not None:
            cols = _read(
                subclass_hedging_sets,
                "subclass_hedging_sets",
                required={"subclass_uix": int, "hedging_set_uix": int},
            )
            hedging_rows = [
                SubclassHedgingSetRow(**cast("dict[str, Any]", cells))
                for cells in _records(cols)
            ]
        rho_rows: list[HedgingSetRhoRow] | None = None
        if hedging_set_correlations is not None:
            cols = _read(
                hedging_set_correlations,
                "hedging_set_correlations",
                required={"hedging_set_uix": int, "rho": float},
            )
            rho_rows = [
                HedgingSetRhoRow(**cast("dict[str, Any]", cells))
                for cells in _records(cols)
            ]
        return cls(
            model=model,
            trades=trade_rows,
            netting_sets=ns_rows,
            add_on_factors=addon_rows,
            pfe_floor=pfe_floor,
            pfe_coefficient=pfe_coefficient,
            subclass_hedging_sets=hedging_rows,
            hedging_set_correlations=rho_rows,
            saccr_alpha=saccr_alpha,
            saccr_multiplier_floor=saccr_multiplier_floor,
        )


# -------------------------------------------------------------------------- output
class ObligorExposure(BaseModel):
    """One obligor's EAD — the row that feeds the credit-VaR ``ead`` field."""

    model_config = _STRICT

    obligor_uid: int
    ead: float


class NettingSetExposureRow(BaseModel):
    """One netting set's EAD and its obligor (the detail behind the obligor roll-up)."""

    model_config = _STRICT

    netting_set_uid: int
    obligor_uid: int
    ead: float


class ExposureProvenance(BaseModel):
    """Audit stamp: enough to reproduce a number."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    engine_version: str
    model: str
    n_trades: int
    n_netting_sets: int
    n_obligors: int


class CreditExposureResult(BaseModel):
    """Per-obligor EAD + netting-set detail + provenance (Contract B)."""

    model_config = _STRICT

    obligor_ead: list[ObligorExposure]
    netting_set_ead: list[NettingSetExposureRow]
    provenance: ExposureProvenance

    def ead_by_obligor_uid(self) -> dict[int, float]:
        """``{obligor_uid: EAD}`` — the join map for the credit-VaR handoff."""
        return {o.obligor_uid: o.ead for o in self.obligor_ead}

    # --- OBLIGOR EAD DATAFRAME
    @property
    def obligor_ead_df(self) -> pl.DataFrame:
        return self.obligor_ead_dataframe()

    def obligor_ead_dataframe(self) -> pl.DataFrame:
        """Return the obligor-EAD list as a DataFrame (for notebook / desk ergonomics)."""
        return pl.DataFrame({
            "obligor_uid": [o.obligor_uid for o in self.obligor_ead],
            "ead": [o.ead for o in self.obligor_ead],
        })

    # --- NETTING-SET EAD DATAFRAME
    @property
    def netting_set_ead_df(self) -> pl.DataFrame:
        return self.netting_set_ead_dataframe()

    def netting_set_ead_dataframe(self) -> pl.DataFrame:
        """Return the netting-set-EAD list as a DataFrame (for notebook / desk ergonomics)."""
        return pl.DataFrame({
            "netting_set_uid": [n.netting_set_uid for n in self.netting_set_ead],
            "obligor_uid": [n.obligor_uid for n in self.netting_set_ead],
            "ead": [n.ead for n in self.netting_set_ead],
        })


_FieldType = type[int] | type[float] | type[str]


def _read(
    frame: pl.DataFrame,
    name: str,
    *,
    required: Mapping[str, _FieldType],
    optional: Mapping[str, _FieldType] | None = None,
) -> dict[str, list[object]]:
    """Read the field columns of one table as Python cells, admitted by dtype.

    Returns one list of cells per field; an optional field whose column is
    absent gives ``None`` cells. See `CreditExposureRequest.from_dataframe`
    for the dtype and cell rules.
    """
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"from_dataframe: {name} missing column(s) {missing}.")
    fields = {c: (t, False) for c, t in required.items()}
    fields.update({c: (t, True) for c, t in (optional or {}).items()})
    out: dict[str, list[object]] = {}
    for col, (kind, is_optional) in fields.items():
        if col not in frame.columns:
            out[col] = [None] * frame.height
            continue
        out[col] = _cells(frame.get_column(col), name, kind, optional=is_optional)
    return out


def _cells(
    series: pl.Series, name: str, kind: _FieldType, *, optional: bool
) -> list[object]:
    dtype = series.dtype
    if optional and dtype == pl.Null:
        return [None] * series.len()
    if kind is str:
        admitted = dtype == pl.Utf8
        rule = "pl.Utf8" + (" or pl.Null" if optional else "")
    else:
        admitted = dtype.is_integer() or dtype.is_float()
        rule = "an integer or a float dtype" + (" or pl.Null" if optional else "")
    if not admitted:
        raise TypeError(
            f"from_dataframe: {name} column '{series.name}' has dtype {dtype!s}; "
            f"the field admits {rule}."
        )
    values: list[object] = series.to_list()
    if kind is str:
        return values
    is_float_col = dtype.is_float()
    out: list[object] = []
    for v in values:
        if v is None or (optional and is_float_col and math.isnan(cast("float", v))):
            out.append(None)  # absent (optional) or None -> R (required)
        elif kind is int:
            out.append(_int_cell(v))
        else:
            out.append(float(cast("float", v)))
    return out


def _int_cell(value: object) -> object:
    """An int-field cell: an integral float becomes its int; a fractional, NaN
    or infinite float is returned unchanged, so the int field raises R."""
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    return value


def _records(columns: dict[str, list[object]]) -> list[dict[str, object]]:
    names = list(columns)
    return [
        dict(zip(names, row, strict=True))
        for row in zip(*columns.values(), strict=True)
    ]
