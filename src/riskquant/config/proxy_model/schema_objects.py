# src/riskquant/config/proxy_model/schema_objects.py
"""
Authored Proxy V1 terms: the input contract of the Proxy V1 compiler (D088).

These records carry only the semantic information the compiler needs:
which target, which driver, which driver chart, which loading. They carry no
horizon, no scenario data, no column index and no target chart. The target's
reconstruction chart ``kappa_A`` is owned by the target factor definition
(D088 §3) and is read by the compiler from the static configuration; it is
never authored here.

Two authoring forms exist (D088 §8):

``NodeProxyTerm``
    One term on exact market points. The author selects the pairing
    (EXPLICIT). Names resolve through the run as-of name map, so a relative
    name and its absolute alias denote the same factor-node.

``ContainerProxyTerm``
    Authoring shorthand. At compile time it expands to one node-level term per
    target pillar, pairing each pillar with the driver pillar whose coordinate
    kind and resolved economic coordinate are equal at the run as-of (EXACT).
    The container form does not survive compilation.

The records are plain frozen values. All validation (types, admissibility,
finiteness, resolution) is owned by
``riskquant.config.proxy_model.compile.compile_proxy_model`` so that every
rejection happens at one compiler boundary with a term-addressed message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from riskquant.market_data.risk_factors.shift_types import ShiftType


@dataclass(frozen=True, slots=True, kw_only=True)
class NodeProxyTerm:
    """
    One authored term ``loading * Y(driver, driver_chart, h)`` on target node.

    Parameters
    ----------
    target_name
        Relative or absolute name of the target factor-node.
    driver_name
        Relative or absolute name of the driver factor-node.
    driver_chart
        Chart through which the driver movement is measured (``tau``). It is
        independent of the target's chart and of the driver's own configured
        shift type.
    loading
        Signed binary64 modelling coefficient ``beta``.
    """

    target_name: str
    driver_name: str
    driver_chart: ShiftType
    loading: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ContainerProxyTerm:
    """
    Authoring shorthand for one term per target pillar, paired by EXACT.

    Parameters
    ----------
    target_container_name
        Name of the target curve or surface.
    driver_container_name
        Name of the driver curve or surface.
    driver_chart
        Chart through which each driver pillar movement is measured (``tau``).
    loading
        Signed binary64 modelling coefficient applied to every expanded term.
    """

    target_container_name: str
    driver_container_name: str
    driver_chart: ShiftType
    loading: float


AuthoredProxyTerm: TypeAlias = NodeProxyTerm | ContainerProxyTerm
