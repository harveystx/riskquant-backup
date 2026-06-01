# src/riskquant/config/enums/asset_classes.py
"""
Asset class and subclass taxonomy.

System-defined constants — not user configuration.
Users reference these string values in node_meta.asset_class /
curve_meta.asset_class / surface_meta.asset_class and the matching
asset_subclass columns.
"""

from __future__ import annotations

from enum import Enum


class AssetClass(str, Enum):
    FX = "FX"
    IR = "IR"
    CREDIT = "CREDIT"
    EQUITY = "EQUITY"
    COMMODITY = "COMMODITY"


class AssetSubclass(str, Enum):
    # Commodity
    CM_AGRI = "CM_AGRI"
    CM_BIOFUEL = "CM_BIOFUEL"
    CM_EUA = "CM_EUA"
    CM_GOO = "CM_GOO"
    CM_IREC = "CM_IREC"
    CM_METALS = "CM_METALS"
    CM_NATGAS = "CM_NATGAS"
    CM_OIL = "CM_OIL"
    CM_POWER = "CM_POWER"
    CM_REC = "CM_REC"
    CM_REGO = "CM_REGO"
    CM_RNG = "CM_RNG"
    CM_UKA = "CM_UKA"
    # Credit
    CR_INDEX = "CR_INDEX"
    CR_SINGLE_NAME = "CR_SINGLE_NAME"
    # Equity
    EQ_INDEX = "EQ_INDEX"
    EQ_SINGLE_NAME = "EQ_SINGLE_NAME"
    # FX
    FX_EM = "FX_EM"
    FX_G10 = "FX_G10"
    # IR
    IR_IBOR = "IR_IBOR"
    IR_INFLATION = "IR_INFLATION"
    IR_OIS = "IR_OIS"


ASSET_SUBCLASS_TO_CLASS: dict[AssetSubclass, AssetClass] = {
    AssetSubclass.CM_AGRI: AssetClass.COMMODITY,
    AssetSubclass.CM_BIOFUEL: AssetClass.COMMODITY,
    AssetSubclass.CM_EUA: AssetClass.COMMODITY,
    AssetSubclass.CM_GOO: AssetClass.COMMODITY,
    AssetSubclass.CM_IREC: AssetClass.COMMODITY,
    AssetSubclass.CM_METALS: AssetClass.COMMODITY,
    AssetSubclass.CM_NATGAS: AssetClass.COMMODITY,
    AssetSubclass.CM_OIL: AssetClass.COMMODITY,
    AssetSubclass.CM_POWER: AssetClass.COMMODITY,
    AssetSubclass.CM_REC: AssetClass.COMMODITY,
    AssetSubclass.CM_REGO: AssetClass.COMMODITY,
    AssetSubclass.CM_RNG: AssetClass.COMMODITY,
    AssetSubclass.CM_UKA: AssetClass.COMMODITY,
    AssetSubclass.CR_INDEX: AssetClass.CREDIT,
    AssetSubclass.CR_SINGLE_NAME: AssetClass.CREDIT,
    AssetSubclass.EQ_INDEX: AssetClass.EQUITY,
    AssetSubclass.EQ_SINGLE_NAME: AssetClass.EQUITY,
    AssetSubclass.FX_EM: AssetClass.FX,
    AssetSubclass.FX_G10: AssetClass.FX,
    AssetSubclass.IR_IBOR: AssetClass.IR,
    AssetSubclass.IR_INFLATION: AssetClass.IR,
    AssetSubclass.IR_OIS: AssetClass.IR,
}

VALID_ASSET_CLASSES: frozenset[str] = frozenset(e.value for e in AssetClass)
VALID_ASSET_SUBCLASSES: frozenset[str] = frozenset(e.value for e in AssetSubclass)
