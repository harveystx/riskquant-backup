# src/riskquant/portfolio/instruments.py

from __future__ import annotations

from enum import IntEnum
from typing import Final

# ============================================================
# INSTRUMENT (ABI-STABLE IDENTIFIERS)
# ============================================================


class InstrumentType(IntEnum):
    """
    Instrument = dispatch identifier for evaluation units.

    This includes:
    - tradable contracts (spot, forwards, options,bonds, swaps, etc)
    - stochastic loss processes (credit JTD)

    Behavior is defined by EvaluationTopology, NOT by instrument name.
    """

    SPOT = 0

    VANILLA_FORWARD_NODE = 1
    VANILLA_FORWARD_CURVE = 2

    VANILLA_EUROPEAN_OPTION_NODE = 3
    VANILLA_EUROPEAN_OPTION_CURVE = 4

    VANILLA_IRS = 5


NUM_INSTRUMENTS: Final[int] = len(InstrumentType)


INSTRUMENT_TYPE_NAMES = frozenset(inst.name for inst in InstrumentType)

# ============================================================
# IDS FOR DISPATCH (SINGLE SOURCE OF TRUTH)
# ============================================================

SPOT_ID: Final[int] = InstrumentType.SPOT.value
VANILLA_FORWARD_NODE_ID: Final[int] = InstrumentType.VANILLA_FORWARD_NODE.value
VANILLA_FORWARD_CURVE_ID: Final[int] = InstrumentType.VANILLA_FORWARD_CURVE.value
VANILLA_EUROPEAN_OPTION_NODE_ID: Final[int] = (
    InstrumentType.VANILLA_EUROPEAN_OPTION_NODE.value
)
VANILLA_EUROPEAN_OPTION_CURVE_ID: Final[int] = (
    InstrumentType.VANILLA_EUROPEAN_OPTION_CURVE.value
)
VANILLA_IRS_ID: Final[int] = InstrumentType.VANILLA_IRS.value


# ============================================================
# EVALUATION TOPOLOGY (PRIMARY DISPATCH AXIS)
# ============================================================


class EvaluationTopology(IntEnum):
    """
    Defines HOW an object is evaluated.
    This is the primary driver of execution strategy.
    """

    MTM = 0  # direct evaluation (spot, forwards)
    PATH_DEPENDENT = 1  # depends on full path
    OPTIMAL_STOPPING = 2  # requires backward induction


# ============================================================
# OPTIONAL: INSTRUMENT FAMILY (NON-DISPATCH)
# ============================================================


class InstrumentTypeFamily(IntEnum):
    SPOT = 0
    FORWARD = 1
    OPTION = 2
    RATES = 3
    CREDIT = 4


# ============================================================
# CORE REGISTRIES (SINGLE SOURCE OF TRUTH)
# ============================================================


INSTRUMENT_TYPE_TO_EVALUATION_TOPOLOGY: Final[
    dict[InstrumentType, EvaluationTopology]
] = {
    # --- MTM ---
    InstrumentType.SPOT: EvaluationTopology.MTM,
    InstrumentType.VANILLA_FORWARD_NODE: EvaluationTopology.MTM,
    InstrumentType.VANILLA_FORWARD_CURVE: EvaluationTopology.MTM,
    # --- Options ---
    InstrumentType.VANILLA_EUROPEAN_OPTION_NODE: EvaluationTopology.MTM,
    InstrumentType.VANILLA_EUROPEAN_OPTION_CURVE: EvaluationTopology.MTM,
    # --- Interest Rate Swaps ---
    InstrumentType.VANILLA_IRS: EvaluationTopology.MTM,
}


INSTRUMENT_TYPE_TO_INSTRUMENT_FAMILY: Final[
    dict[InstrumentType, InstrumentTypeFamily]
] = {
    # SPOT
    InstrumentType.SPOT: InstrumentTypeFamily.SPOT,
    # FORWARDS
    InstrumentType.VANILLA_FORWARD_NODE: InstrumentTypeFamily.FORWARD,
    InstrumentType.VANILLA_FORWARD_CURVE: InstrumentTypeFamily.FORWARD,
    # OPTIONS
    InstrumentType.VANILLA_EUROPEAN_OPTION_NODE: InstrumentTypeFamily.OPTION,
    InstrumentType.VANILLA_EUROPEAN_OPTION_CURVE: InstrumentTypeFamily.OPTION,
    # RATES
    InstrumentType.VANILLA_IRS: InstrumentTypeFamily.RATES,
}


# ============================================================
# VALIDATION (HARD FAIL ON DRIFT)
# ============================================================


_ALL_INST_TYPES = frozenset(InstrumentType)

if frozenset(INSTRUMENT_TYPE_TO_EVALUATION_TOPOLOGY) != _ALL_INST_TYPES:
    raise RuntimeError("Topology registry drift")

if frozenset(INSTRUMENT_TYPE_TO_INSTRUMENT_FAMILY) != _ALL_INST_TYPES:
    raise RuntimeError("Family registry drift")


# ============================================================
# FAST DISPATCH ARRAYS (CRITICAL FOR PERFORMANCE)
# ============================================================


# Dense arrays for O(1) lookup without dicts
INSTRUMENT_TYPE_TO_TOPOLOGY_ARRAY: Final[tuple[int, ...]] = tuple(
    INSTRUMENT_TYPE_TO_EVALUATION_TOPOLOGY[InstrumentType(i)]
    for i in range(0, NUM_INSTRUMENTS)
)

INSTRUMENT_TYPE_TO_FAMILY_ARRAY: Final[tuple[int, ...]] = tuple(
    INSTRUMENT_TYPE_TO_INSTRUMENT_FAMILY[InstrumentType(i)]
    for i in range(0, NUM_INSTRUMENTS)
)


# ============================================================
# EDGE (STRING) REGISTRIES — ONLY FOR I/O
# ============================================================


INSTRUMENT_NAME_TO_INSTRUMENT_REGISTRY: Final[dict[str, InstrumentType]] = {
    inst.name: inst for inst in InstrumentType
}

INSTRUMENT_NAME_TO_ID_REGISTRY: Final[dict[str, int]] = {
    inst.name: inst.value for inst in InstrumentType
}

INSTRUMENT_ID_TO_NAME_REGISTRY: Final[tuple[str, ...]] = tuple(
    InstrumentType(i).name for i in range(0, NUM_INSTRUMENTS)
)


# ============================================================
# INSTRUMENT FAMILY TO INSTRUMENT NAME REGISTRIES
# ============================================================


# Family → tuple of instrument names (all instruments in a family)
INSTRUMENT_FAMILY_TO_INSTRUMENT_NAMES: Final[
    dict[InstrumentTypeFamily, tuple[str, ...]]
] = {
    family: tuple(
        inst.name
        for inst, fam in INSTRUMENT_TYPE_TO_INSTRUMENT_FAMILY.items()
        if fam == family
    )
    for family in InstrumentTypeFamily
}

# Family → tuple of instrument IDs
INSTRUMENT_FAMILY_TO_INSTRUMENT_IDS: Final[
    dict[InstrumentTypeFamily, tuple[int, ...]]
] = {
    family: tuple(
        inst.value
        for inst, fam in INSTRUMENT_TYPE_TO_INSTRUMENT_FAMILY.items()
        if fam == family
    )
    for family in InstrumentTypeFamily
}

# Family name (string) → tuple of instrument names
INSTRUMENT_FAMILY_NAME_TO_INSTRUMENT_NAMES: Final[dict[str, tuple[str, ...]]] = {
    family.name: tuple(
        inst.name
        for inst, fam in INSTRUMENT_TYPE_TO_INSTRUMENT_FAMILY.items()
        if fam == family
    )
    for family in InstrumentTypeFamily
}

if __name__ == "__main__":
    # Instument.
    print("\nInstruments:")
    for inst in InstrumentType:
        print(f"  {inst.name} (ID={inst.value})")

    # Sanity check: print all registries
    print("\nINSTRUMENT_TO_TOPOLOGY:")
    for inst, topo in INSTRUMENT_TYPE_TO_EVALUATION_TOPOLOGY.items():
        print(f"  {inst.name} → {topo.name}")

    print("\nINSTRUMENT_TO_FAMILY:")
    for inst, fam in INSTRUMENT_TYPE_TO_INSTRUMENT_FAMILY.items():
        print(f"  {inst.name} → {fam.name}")

    print("\nFAMILY_TO_INSTRUMENT_NAMES:")
    for fam, insts in INSTRUMENT_FAMILY_TO_INSTRUMENT_NAMES.items():
        print(f"  {fam.name} → {insts}")

    print("\n --- END: ALL CHECKS PASSED --- \n")
