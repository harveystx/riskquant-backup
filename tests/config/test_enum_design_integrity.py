# tests/config/test_enum_design_integrity.py
"""
Enum design-integrity tests for the `riskquant.config` audit (RQ-AUD-019,
RQ-AUD-020).

These are architecture/design tests, not numerical correctness tests: they
verify that the taxonomy layer in `riskquant.config.enums` is internally
consistent and will not silently produce an incompatible type collision or
an unsupported-but-declared enum member in the future.
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest


def _all_enum_module_names() -> list[str]:
    import riskquant.config.enums as enums_pkg

    return [
        name
        for _, name, _ in pkgutil.iter_modules(enums_pkg.__path__)
    ]


def test_no_duplicate_class_names_across_enum_modules():
    """
    RQ-AUD-019 (MEDIUM, latent defect -- currently zero consumers, but a
    real landmine): `SwapDirection` is defined TWICE in
    `riskquant.config.enums`, with completely different and incompatible
    semantics:

        enums/fixing_type.py   -> SwapDirection.ADVANCE = 0, ARREARS = 1
                                   (this is fixing-timing, not direction)
        enums/swap_direction.py -> SwapDirection.PAY_FIXED = -1, RECEIVE_FIXED = +1
                                   (this is the actual swap direction)

    Whichever one a future consumer imports first (or via an unqualified
    `from riskquant.config.enums.fixing_type import SwapDirection`) silently
    gets the WRONG type if the intent was the other one. For a swap pricer,
    confusing ADVANCE/ARREARS (fixing timing) with PAY_FIXED/RECEIVE_FIXED
    (cashflow sign) is a direct P&L-sign-correctness risk.

    This test enumerates every class defined in every `riskquant.config.enums`
    submodule and fails if the same class name is declared more than once.
    """
    import inspect

    seen: dict[str, list[str]] = {}
    for mod_name in _all_enum_module_names():
        module = importlib.import_module(f"riskquant.config.enums.{mod_name}")
        for cls_name, cls_obj in inspect.getmembers(module, inspect.isclass):
            if cls_obj.__module__ != module.__name__:
                continue  # skip re-exported/imported classes (e.g. XAxisKind in roll_methods)
            seen.setdefault(cls_name, []).append(mod_name)

    duplicates = {name: mods for name, mods in seen.items() if len(mods) > 1}
    assert not duplicates, (
        "Duplicate enum/class names found across riskquant.config.enums "
        "submodules -- each definition may have different semantics and "
        "values, creating a silent-collision risk for any future consumer:\n"
        + "\n".join(f"  {name}: defined in {mods}" for name, mods in duplicates.items())
    )


def test_swap_direction_semantics_are_unambiguous_once_fixed():
    """
    Documents the CURRENT (broken) state precisely, so a future fix has an
    exact target. Once the naming collision above is resolved (e.g. by
    renaming fixing_type.SwapDirection to FixingTiming), this test should be
    updated to assert the single remaining SwapDirection has exactly
    {PAY_FIXED, RECEIVE_FIXED} as members.
    """
    from riskquant.config.enums.fixing_type import SwapDirection as FixingTiming
    from riskquant.config.enums.swap_direction import SwapDirection as ActualDirection

    fixing_members = set(FixingTiming.__members__)
    direction_members = set(ActualDirection.__members__)

    assert fixing_members == {"ADVANCE", "ARREARS"}, (
        f"fixing_type.SwapDirection members changed: {fixing_members}. "
        "If this module was renamed to something other than SwapDirection, "
        "update this test and delete the duplicate-name test above's "
        "reliance on it."
    )
    assert direction_members == {"PAY_FIXED", "RECEIVE_FIXED"}, (
        f"swap_direction.SwapDirection members changed: {direction_members}."
    )
    # The two are DELIBERATELY disjoint member sets today, proving they are
    # semantically unrelated concepts wearing the same class name.
    assert fixing_members.isdisjoint(direction_members)


def test_every_container_topology_type_has_a_reserved_id_namespace():
    """
    RQ-AUD-020 (MEDIUM, design-completeness gap): `ContainerTopologyType`
    declares NODE, CURVE, SURFACE, and CUBE. `GlobalIdNamespace`
    (riskquant.config.namespace) reserves a contiguous int32 ID block for
    NODE, CURVE, and SURFACE only. CUBE has zero supporting
    infrastructure anywhere in the repository: no reserved ID namespace,
    no schema_objects dataclass (no CubeMeta/CubePointMeta), no
    compile.py or validate.py handling.

    This test enumerates every ContainerTopologyType member and asserts a
    matching GlobalIdNamespace attribute exists, using the ContainerTopologyType
    member's name as the expected GlobalIdNamespace attribute name
    (NODE -> NODES, CURVE -> CURVES, SURFACE -> SURFACES pluralisation is
    handled explicitly below since the two enums do not share a naming
    convention today -- itself a minor design inconsistency).
    """
    from riskquant.config.enums.container_topology import ContainerTopologyType
    from riskquant.config.namespace import GlobalIdNamespace

    # Explicit singular -> plural mapping (documents the naming mismatch
    # between ContainerTopologyType members and GlobalIdNamespace attributes).
    expected_namespace_attr = {
        "NODE": "NODES",
        "CURVE": "CURVES",
        "SURFACE": "SURFACES",
        "CUBE": "CUBES",
    }

    missing = []
    for member in ContainerTopologyType:
        attr = expected_namespace_attr.get(member.name, member.name + "S")
        if not hasattr(GlobalIdNamespace, attr):
            missing.append((member.name, attr))

    assert not missing, (
        "ContainerTopologyType members with no corresponding "
        "GlobalIdNamespace ID-space reservation (declared taxonomy without "
        "supporting config infrastructure):\n"
        + "\n".join(f"  {name} (expected GlobalIdNamespace.{attr})" for name, attr in missing)
    )


def test_id_namespace_blocks_are_disjoint_and_within_int32():
    """
    Positive/regression test: confirms the namespace.py module-level
    invariants (currently enforced only via bare `assert` at import time,
    which mypy/pytest do not re-verify) using an explicit, discoverable
    pytest assertion instead. This is EXACT: the namespace blocks must
    never overlap and must fit in int32 (the declared dtype for rf_id
    throughout the numerical core).
    """
    import numpy as np

    from riskquant.config.namespace import GlobalIdNamespace

    blocks = [
        GlobalIdNamespace.NODES,
        GlobalIdNamespace.CURVES,
        GlobalIdNamespace.SURFACES,
    ]
    for block in blocks:
        assert block.min_id <= block.max_id
        assert block.max_id <= np.iinfo(np.int32).max

    sorted_blocks = sorted(blocks, key=lambda b: b.min_id)
    for prev, curr in zip(sorted_blocks, sorted_blocks[1:]):
        assert prev.max_id < curr.min_id, (
            f"Overlapping or adjacent-without-gap ID namespace blocks: "
            f"{prev.name} ends at {prev.max_id}, {curr.name} starts at {curr.min_id}"
        )
