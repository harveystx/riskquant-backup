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

from collections.abc import Hashable, Mapping
from typing import Any

import pandas as pd
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from riskquant.solutions.credit_exposure.catalog import (
    EXPOSURE_MODEL_CATALOG,
    exposure_model_entry,
)

_STRICT = ConfigDict(extra="forbid", protected_namespaces=())


# --------------------------------------------------------------------------- input
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
        if entry.requires_saccr_config and (
            not self.subclass_hedging_sets or not self.hedging_set_correlations
        ):
            raise ValueError(
                f"model '{self.model}' requires 'subclass_hedging_sets' and "
                "'hedging_set_correlations' (the SA-CCR hedging-set config)."
            )
        return self

    @classmethod
    def from_dataframe(
        cls,
        trades: pd.DataFrame,
        netting_sets: pd.DataFrame,
        *,
        model: str,
        add_on_factors: pd.DataFrame | None = None,
        pfe_floor: float | None = None,
        pfe_coefficient: float | None = None,
        subclass_hedging_sets: pd.DataFrame | None = None,
        hedging_set_correlations: pd.DataFrame | None = None,
        saccr_alpha: float | None = None,
        saccr_multiplier_floor: float | None = None,
    ) -> CreditExposureRequest:
        """Build a request from flat DataFrames (notebook / desk ergonomics).

        ``trades`` needs ``trade_uid, netting_set_uid, mtm`` (plus ``volume,
        current_price, asset_subclass_uix`` for CEM); ``netting_sets`` needs
        ``netting_set_uid, obligor_uid, net_collateral``; ``add_on_factors`` needs
        ``subclass_uix, factor``. Extra columns are ignored.
        """
        _require(trades, ("trade_uid", "netting_set_uid", "mtm"), "trades")
        _require(
            netting_sets,
            ("netting_set_uid", "obligor_uid", "net_collateral"),
            "netting_sets",
        )
        trade_rows = [
            TradeRow(
                trade_uid=int(r["trade_uid"]),
                netting_set_uid=int(r["netting_set_uid"]),
                mtm=float(r["mtm"]),
                volume=_opt_float(r, "volume"),
                current_price=_opt_float(r, "current_price"),
                asset_subclass_uix=_opt_int(r, "asset_subclass_uix"),
            )
            for r in trades.to_dict("records")
        ]
        ns_rows = [
            NettingSetRow(
                netting_set_uid=int(r["netting_set_uid"]),
                obligor_uid=int(r["obligor_uid"]),
                net_collateral=float(r["net_collateral"]),
            )
            for r in netting_sets.to_dict("records")
        ]
        addon_rows: list[AddOnFactorRow] | None = None
        if add_on_factors is not None:
            _require(add_on_factors, ("subclass_uix", "factor"), "add_on_factors")
            addon_rows = [
                AddOnFactorRow(
                    subclass_uix=int(r["subclass_uix"]), factor=float(r["factor"])
                )
                for r in add_on_factors.to_dict("records")
            ]
        hedging_rows: list[SubclassHedgingSetRow] | None = None
        if subclass_hedging_sets is not None:
            _require(
                subclass_hedging_sets,
                ("subclass_uix", "hedging_set_uix"),
                "subclass_hedging_sets",
            )
            hedging_rows = [
                SubclassHedgingSetRow(
                    subclass_uix=int(r["subclass_uix"]),
                    hedging_set_uix=int(r["hedging_set_uix"]),
                )
                for r in subclass_hedging_sets.to_dict("records")
            ]
        rho_rows: list[HedgingSetRhoRow] | None = None
        if hedging_set_correlations is not None:
            _require(
                hedging_set_correlations,
                ("hedging_set_uix", "rho"),
                "hedging_set_correlations",
            )
            rho_rows = [
                HedgingSetRhoRow(
                    hedging_set_uix=int(r["hedging_set_uix"]), rho=float(r["rho"])
                )
                for r in hedging_set_correlations.to_dict("records")
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
        return pl.DataFrame(
            {
                "obligor_uid": [o.obligor_uid for o in self.obligor_ead],
                "ead": [o.ead for o in self.obligor_ead],
            }
        )

    # --- NETTING-SET EAD DATAFRAME
    @property
    def netting_set_ead_df(self) -> pl.DataFrame:
        return self.netting_set_ead_dataframe()

    def netting_set_ead_dataframe(self) -> pl.DataFrame:
        """Return the netting-set-EAD list as a DataFrame (for notebook / desk ergonomics)."""
        return pl.DataFrame(
            {
                "netting_set_uid": [n.netting_set_uid for n in self.netting_set_ead],
                "obligor_uid": [n.obligor_uid for n in self.netting_set_ead],
                "ead": [n.ead for n in self.netting_set_ead],
            }
        )


def _require(df: pd.DataFrame, cols: tuple[str, ...], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"from_dataframe: {name} missing column(s) {missing}.")


def _opt_float(rec: Mapping[Hashable, Any], key: str) -> float | None:
    if key not in rec or pd.isna(rec[key]):
        return None
    return float(rec[key])


def _opt_int(rec: Mapping[Hashable, Any], key: str) -> int | None:
    if key not in rec or pd.isna(rec[key]):
        return None
    return int(rec[key])
