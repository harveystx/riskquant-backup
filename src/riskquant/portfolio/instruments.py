# src/riskquant/portfolio/instruments.py

from __future__ import annotations

from enum import IntEnum
from typing import Final

# ============================================================
# INSTRUMENT (ABI-STABLE IDENTIFIERS)
# ============================================================


class Instrument(IntEnum):
    """
    Instrument = dispatch identifier for evaluation units.

    This includes:
    - tradable contracts (spot, forwards, options,bonds, swaps, etc)
    - stochastic loss processes (credit JTD)

    Behavior is defined by EvaluationTopology, NOT by instrument name.
    """

    # --- Market Risk ---
    SPOT = 0

    VANILLA_FORWARD_NODE = 1
    VANILLA_FORWARD_CURVE = 2

    VANILLA_EUROPEAN_OPTION_NODE = 3
    VANILLA_EUROPEAN_OPTION_CURVE = 4


NUM_INSTRUMENTS: Final[int] = len(Instrument)


INSTRUMENT_NAMES = frozenset(inst.name for inst in Instrument)

# ============================================================
# IDS FOR DISPATCH (SINGLE SOURCE OF TRUTH)
# ============================================================

SPOT_ID: Final[int] = Instrument.SPOT.value
VANILLA_FORWARD_NODE_ID: Final[int] = Instrument.VANILLA_FORWARD_NODE.value
VANILLA_FORWARD_CURVE_ID: Final[int] = Instrument.VANILLA_FORWARD_CURVE.value
VANILLA_EUROPEAN_OPTION_NODE_ID: Final[int] = (
    Instrument.VANILLA_EUROPEAN_OPTION_NODE.value
)
VANILLA_EUROPEAN_OPTION_CURVE_ID: Final[int] = (
    Instrument.VANILLA_EUROPEAN_OPTION_CURVE.value
)


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


class InstrumentFamily(IntEnum):
    SPOT = 0
    FORWARD = 1
    OPTION = 2
    CREDIT = 3


# ============================================================
# CORE REGISTRIES (SINGLE SOURCE OF TRUTH)
# ============================================================


INSTRUMENT_TO_TOPOLOGY: Final[dict[Instrument, EvaluationTopology]] = {
    # --- MTM ---
    Instrument.SPOT: EvaluationTopology.MTM,
    Instrument.VANILLA_FORWARD_NODE: EvaluationTopology.MTM,
    Instrument.VANILLA_FORWARD_CURVE: EvaluationTopology.MTM,
    # --- Options ---
    Instrument.VANILLA_EUROPEAN_OPTION_NODE: EvaluationTopology.MTM,
    Instrument.VANILLA_EUROPEAN_OPTION_CURVE: EvaluationTopology.MTM,
}


INSTRUMENT_TO_FAMILY: Final[dict[Instrument, InstrumentFamily]] = {
    Instrument.SPOT: InstrumentFamily.SPOT,
    Instrument.VANILLA_FORWARD_NODE: InstrumentFamily.FORWARD,
    Instrument.VANILLA_FORWARD_CURVE: InstrumentFamily.FORWARD,
    Instrument.VANILLA_EUROPEAN_OPTION_NODE: InstrumentFamily.OPTION,
    Instrument.VANILLA_EUROPEAN_OPTION_CURVE: InstrumentFamily.OPTION,
}


# ============================================================
# VALIDATION (HARD FAIL ON DRIFT)
# ============================================================


_ALL = frozenset(Instrument)

if _ALL - frozenset(INSTRUMENT_TO_TOPOLOGY):
    raise RuntimeError("Missing topology mappings")

if _ALL - frozenset(INSTRUMENT_TO_FAMILY):
    raise RuntimeError("Missing family mappings")


# ============================================================
# FAST DISPATCH ARRAYS (CRITICAL FOR PERFORMANCE)
# ============================================================


# Dense arrays for O(1) lookup without dicts
INSTRUMENT_TO_TOPOLOGY_ARRAY: Final[list[int]] = [
    INSTRUMENT_TO_TOPOLOGY[Instrument(i)] for i in range(0, NUM_INSTRUMENTS)
]

INSTRUMENT_TO_FAMILY_ARRAY: Final[list[int]] = [
    INSTRUMENT_TO_FAMILY[Instrument(i)] for i in range(0, NUM_INSTRUMENTS)
]


# ============================================================
# EDGE (STRING) REGISTRIES — ONLY FOR I/O
# ============================================================


INSTRUMENT_NAME_TO_INSTRUMENT_REGISTRY: Final[dict[str, Instrument]] = {
    inst.name: inst for inst in Instrument
}

INSTRUMENT_NAME_TO_ID_REGISTRY: Final[dict[str, int]] = {
    inst.name: inst.value for inst in Instrument
}

INSTRUMENT_ID_TO_NAME_REGISTRY: Final[list[str]] = [
    Instrument(i).name for i in range(0, NUM_INSTRUMENTS)
]


# ============================================================
# INSTRUMENT FAMILY TO INSTRUMENT NAME REGISTRIES
# ============================================================


# Family → list of instrument names (all instruments in a family)
FAMILY_TO_INSTRUMENT_NAMES: Final[dict[InstrumentFamily, list[str]]] = {
    family: [inst.name for inst, fam in INSTRUMENT_TO_FAMILY.items() if fam == family]
    for family in InstrumentFamily
}

# Family → list of instrument IDs
FAMILY_TO_INSTRUMENT_IDS: Final[dict[InstrumentFamily, list[int]]] = {
    family: [inst.value for inst, fam in INSTRUMENT_TO_FAMILY.items() if fam == family]
    for family in InstrumentFamily
}

# Family name (string) → list of instrument names
FAMILY_NAME_TO_INSTRUMENT_NAMES: Final[dict[str, list[str]]] = {
    family.name: [
        inst.name for inst, fam in INSTRUMENT_TO_FAMILY.items() if fam == family
    ]
    for family in InstrumentFamily
}


if __name__ == "__main__":
    # Instument.
    print("\nInstruments:")
    for inst in Instrument:
        print(f"  {inst.name} (ID={inst.value})")

    # Sanity check: print all registries
    print("\nINSTRUMENT_TO_TOPOLOGY:")
    for inst, topo in INSTRUMENT_TO_TOPOLOGY.items():
        print(f"  {inst.name} → {topo.name}")

    print("\nINSTRUMENT_TO_FAMILY:")
    for inst, fam in INSTRUMENT_TO_FAMILY.items():
        print(f"  {inst.name} → {fam.name}")

    print("\nFAMILY_TO_INSTRUMENT_NAMES:")
    for fam, insts in FAMILY_TO_INSTRUMENT_NAMES.items():
        print(f"  {fam.name} → {insts}")

    print("\n --- END: ALL CHECKS PASSED --- \n")
