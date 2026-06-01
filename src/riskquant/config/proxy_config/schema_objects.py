# src/riskquant/config/proxy_config/schema_objects.py
"""
Immutable value object for one proxy mapping.

The proxying-level discriminator (container vs risk-factor) is carried by
the partition the ProxyMap lives in (ProxyTopology.container_proxies or
.rf_proxies), not by a field on the dataclass. Strings die at the edge;
the runtime never reads a level string.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class ProxyMap:
    """
    A single proxy mapping entry.

    The required name's historical data is unavailable or insufficient
    for risk estimation; the proxy name's data is used in its place,
    optionally scaled by `vol_scaling_factor` to account for known
    relative-vol differences between the two series.

    Attributes
    ----------
    required_name
        The name whose data is being substituted (the "missing" one).
        A container name when used in container_proxies, a relative
        risk-factor name when used in rf_proxies.
    proxy_name
        The name supplying the substitute data. Must be a peer of
        required_name (same level — container or rf — same namespace).
    vol_scaling_factor
        Strictly positive multiplier applied to the proxy's vol/shocks
        before substitution. Loader defaults missing cells to 1.0
        (identity). The validator rejects values <= 0 or > 10.0.
    """

    required_name: str
    proxy_name: str
    vol_scaling_factor: float
