# src/riskquant/config/proxy_config/proxy_topology.py
"""
Runtime proxy-topology container.

Pandas-free. Hot-path lookups are dense int32 / float64 NumPy arrays
indexed by required-id; sentinel -1 means "no proxy registered for this
id, pass through unchanged."

Built once at compile time. Immutable thereafter; the `with_overrides`
factory returns a brand-new instance.

Resolution semantics (single-hop, enforced by the validator)
------------------------------------------------------------
For any required_id `r`:
    proxy_id    = container_id_to_proxy_id[r]            (or rf_id_to_proxy_rf_id[r])
    proxy_scale = container_id_to_proxy_vol_scale[r]     (or rf_id_to_proxy_vol_scale[r])

If proxy_id < 0, no proxy is registered — pass through.
If proxy_id >= 0, the proxy must NOT itself be proxied (the validator
forbids chains). The kernel performs a single gather, no iteration.

The validator also forbids container-level proxies whose children are
rf-level proxied (cross-level conflict) — they are mutually exclusive
policies and coexisting them silently mis-applies the user's intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.config.proxy_config.schema_objects import ProxyMap

if TYPE_CHECKING:
    from riskquant.config.static_config.static_config import RiskStaticConfig


# Sentinel for "no proxy registered for this id" in the dense arrays.
# Scoped to ProxyTopology — not in riskquant.constants because no
# consumer outside this module compares against it directly (the kernel
# uses `proxy_id >= 0` branchless masking instead).
_NO_PROXY_ID: int = -1


# Default vol scale for ids with no proxy. 1.0 is the identity
# multiplier; using NaN here would propagate into downstream math.
_DEFAULT_VOL_SCALE: float = 1.0


@dataclass(frozen=True, slots=True)
class ProxyTopology:
    """
    Compiled proxy topology.

    Init fields are supplied by `ProxyTopology.build` (or by
    `with_overrides`, which re-routes through `build`); they are not
    intended to be constructed directly by user code.

    Hot-path readers consume only the four dense arrays:
        container_id_to_proxy_id
        container_id_to_proxy_vol_scale
        rf_id_to_proxy_rf_id
        rf_id_to_proxy_vol_scale

    The two dicts (container_proxies, rf_proxies) are retained for
    audit and override purposes and are never touched on the hot path.
    """

    # ------------------------------------------------------------------ #
    # Human-readable layer (audit + override path)                       #
    # ------------------------------------------------------------------ #

    container_proxies: dict[str, ProxyMap] = field(repr=False)
    rf_proxies: dict[str, ProxyMap] = field(repr=False)

    # ------------------------------------------------------------------ #
    # Hot-path dense arrays                                              #
    # ------------------------------------------------------------------ #
    # Length = static_config.max_container_id + 1.
    # Indexed by required_container_id.
    container_id_to_proxy_id: npt.NDArray[np.int32] = field(repr=False)
    container_id_to_proxy_vol_scale: npt.NDArray[np.float64] = field(repr=False)

    # Length = static_config.max_rf_id + 1. Indexed by required_rf_id.
    rf_id_to_proxy_rf_id: npt.NDArray[np.int32] = field(repr=False)
    rf_id_to_proxy_vol_scale: npt.NDArray[np.float64] = field(repr=False)

    # ------------------------------------------------------------------ #
    # Factories                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        *,
        container_proxies: dict[str, ProxyMap],
        rf_proxies: dict[str, ProxyMap],
        static_config: RiskStaticConfig,
    ) -> ProxyTopology:
        """
        Build a ProxyTopology by resolving names against the static config
        and emitting the dense lookup arrays.

        Parameters
        ----------
        container_proxies, rf_proxies
            Already validated by `validate_proxy_maps`. Required name and
            proxy name are guaranteed to resolve in static_config; no
            chains, no cycles, no cross-level conflicts.
        static_config
            Source of truth for name -> id resolution and id-space sizes.
        """
        n_container = int(static_config.max_container_id) + 1
        n_rf = int(static_config.max_rf_id) + 1

        c_proxy = np.full(n_container, _NO_PROXY_ID, dtype=np.int32)
        c_scale = np.full(n_container, _DEFAULT_VOL_SCALE, dtype=np.float64)
        r_proxy = np.full(n_rf, _NO_PROXY_ID, dtype=np.int32)
        r_scale = np.full(n_rf, _DEFAULT_VOL_SCALE, dtype=np.float64)

        name_to_cid = static_config.container_name_to_id
        for pm in container_proxies.values():
            req_id = int(name_to_cid[pm.required_name])
            proxy_id = int(name_to_cid[pm.proxy_name])
            c_proxy[req_id] = np.int32(proxy_id)
            c_scale[req_id] = float(pm.vol_scaling_factor)

        name_to_rfid = static_config.rel_name_to_rf_id
        for pm in rf_proxies.values():
            req_id = int(name_to_rfid[pm.required_name])
            proxy_id = int(name_to_rfid[pm.proxy_name])
            r_proxy[req_id] = np.int32(proxy_id)
            r_scale[req_id] = float(pm.vol_scaling_factor)

        return cls(
            container_proxies=container_proxies,
            rf_proxies=rf_proxies,
            container_id_to_proxy_id=c_proxy,
            container_id_to_proxy_vol_scale=c_scale,
            rf_id_to_proxy_rf_id=r_proxy,
            rf_id_to_proxy_vol_scale=r_scale,
        )

    def with_overrides(
        self,
        *,
        static_config: RiskStaticConfig,
        container_overrides: dict[str, ProxyMap] | None = None,
        rf_overrides: dict[str, ProxyMap] | None = None,
    ) -> ProxyTopology:
        """
        Return a new topology with overrides layered on top of self.

        Overrides are full-replace, keyed by required_name: any
        required_name appearing in the overrides supersedes the base
        topology's mapping for that name (or adds a new one).

        Re-runs the full proxy-map validation against the supplied
        static_config so override-induced chains, cycles, or cross-level
        conflicts surface immediately.
        """
        # Local import to avoid a cycle: validate imports schema_objects,
        # which proxy_topology already imports; validate also imports the
        # module that ends up being this one transitively via compile.
        from riskquant.config.proxy_config.validate import validate_proxy_maps

        merged_containers = dict(self.container_proxies)
        if container_overrides:
            merged_containers.update(container_overrides)

        merged_rfs = dict(self.rf_proxies)
        if rf_overrides:
            merged_rfs.update(rf_overrides)

        validate_proxy_maps(
            container_proxies=merged_containers,
            rf_proxies=merged_rfs,
            static_config=static_config,
        )

        return ProxyTopology.build(
            container_proxies=merged_containers,
            rf_proxies=merged_rfs,
            static_config=static_config,
        )

    # ------------------------------------------------------------------ #
    # Single-id helpers (debug / unit-test only — not for hot path)      #
    # ------------------------------------------------------------------ #

    def resolve_container(self, required_container_id: int) -> tuple[int, float]:
        """
        Return (proxy_container_id, vol_scale) for a single required id.

        If no proxy is registered, returns (required_container_id, 1.0)
        — the identity mapping the kernel applies via branchless masking.
        Hot paths must use the dense arrays directly; this is a
        convenience for tests and notebook sessions.
        """
        i = int(required_container_id)
        proxy_id = int(self.container_id_to_proxy_id[i])
        if proxy_id < 0:
            return (i, _DEFAULT_VOL_SCALE)
        return (proxy_id, float(self.container_id_to_proxy_vol_scale[i]))

    def resolve_rf(self, required_rf_id: int) -> tuple[int, float]:
        """Same as resolve_container, but for the rf-id space."""
        i = int(required_rf_id)
        proxy_id = int(self.rf_id_to_proxy_rf_id[i])
        if proxy_id < 0:
            return (i, _DEFAULT_VOL_SCALE)
        return (proxy_id, float(self.rf_id_to_proxy_vol_scale[i]))

    # ------------------------------------------------------------------ #
    # Scalar accessors                                                   #
    # ------------------------------------------------------------------ #

    @property
    def num_container_proxies(self) -> int:
        return len(self.container_proxies)

    @property
    def num_rf_proxies(self) -> int:
        return len(self.rf_proxies)

    @property
    def num_total_proxies(self) -> int:
        return self.num_container_proxies + self.num_rf_proxies

    def summary(self) -> None:
        """Print a console-friendly overview of the compiled topology."""
        lines = [
            "================================",
            "===   ProxyTopology Summary  ===",
            "================================",
            f"  Container proxies   : {self.num_container_proxies}",
            f"  RF proxies          : {self.num_rf_proxies}",
            f"  Total               : {self.num_total_proxies}",
            f"  c_proxy array size  : {self.container_id_to_proxy_id.size}",
            f"  rf_proxy array size : {self.rf_id_to_proxy_rf_id.size}",
            "================================",
        ]
        print("\n".join(lines))
