# src/riskquant/config/proxy_config/proxy_config.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from riskquant.config.proxy_config.schema_objects import ProxyMap


@dataclass(frozen=True, slots=True, kw_only=True)
class ProxyTopology:
    """
    Immutable container for proxy mappings.
    Maintains both the human-readable string maps and the compiled integer maps
    required for fast C-level kernel execution.
    """

    container_proxies_dict: Dict[str, ProxyMap]
    risk_factor_proxies_dict: Dict[str, ProxyMap]

    # Stored lightweight mappings for easy dataframe visualization and override resolution
    container_name_to_container_id: Dict[str, int]
    rf_name_to_rf_id: Dict[str, int]

    # Pre-compiled for O(1) integer kernel access: required_id -> (proxy_id, scaling_factor)
    container_id_to_proxy_container_id: Dict[int, Tuple[int, float]]
    rf_id_to_proxy_rf_id: Dict[int, Tuple[int, float]]

    @classmethod
    def build(
        cls,
        container_proxies: Dict[str, ProxyMap],
        rf_proxies: Dict[str, ProxyMap],
        container_name_to_container_id: Dict[str, int],
        rf_name_to_rf_id: Dict[str, int],
    ) -> "ProxyTopology":
        """Factory method to construct ProxyTopology with pre-compiled integer maps."""

        # Build container ID to proxy container ID map (including multiplier)
        container_id_to_proxy_container_id = {
            container_name_to_container_id[cp.required_name]: (
                container_name_to_container_id[cp.proxy_name],
                float(
                    cp.vol_scaling_factor
                    if cp.vol_scaling_factor is not None
                    else 1.0
                ),
            )
            for cp in container_proxies.values()
        }

        # Build risk factor ID to proxy risk factor ID map (including multiplier)
        rf_id_to_proxy_rf_id = {
            rf_name_to_rf_id[rp.required_name]: (
                rf_name_to_rf_id[rp.proxy_name],
                float(
                    rp.vol_scaling_factor
                    if rp.vol_scaling_factor is not None
                    else 1.0
                ),
            )
            for rp in rf_proxies.values()
        }

        return cls(
            container_proxies_dict=container_proxies,
            risk_factor_proxies_dict=rf_proxies,
            container_name_to_container_id=container_name_to_container_id,
            rf_name_to_rf_id=rf_name_to_rf_id,
            container_id_to_proxy_container_id=container_id_to_proxy_container_id,
            rf_id_to_proxy_rf_id=rf_id_to_proxy_rf_id,
        )

    def with_overrides(
        self,
        container_proxy_overrides: Optional[Dict[str, ProxyMap]] = None,
        rf_proxy_overrides: Optional[Dict[str, ProxyMap]] = None,
        # Optional updated mappings, defaults to existing ones if not provided
        container_name_to_container_id: Optional[Dict[str, int]] = None,
        rf_name_to_rf_id: Optional[Dict[str, int]] = None,
    ) -> "ProxyTopology":
        """
        Creates a new ProxyTopology by safely layering overrides on top of the base defaults.
        Recompiles the fast integer maps automatically.
        """
        new_containers = dict(self.container_proxies_dict)
        if container_proxy_overrides:
            new_containers.update(container_proxy_overrides)

        new_rfs = dict(self.risk_factor_proxies_dict)
        if rf_proxy_overrides:
            new_rfs.update(rf_proxy_overrides)

        return ProxyTopology.build(
            container_proxies=new_containers,
            rf_proxies=new_rfs,
            container_name_to_container_id=container_name_to_container_id
            or self.container_name_to_container_id,
            rf_name_to_rf_id=rf_name_to_rf_id or self.rf_name_to_rf_id,
        )

    def to_df(self) -> pd.DataFrame:
        """
        Consolidates both container and risk factor proxies into a single
        pandas DataFrame for easy visualization and auditing.
        """
        records = []

        # Process container proxies
        for cp in self.container_proxies_dict.values():
            records.append(
                {
                    "proxying_level": cp.proxying_level,
                    "required_name": cp.required_name,
                    "required_name_id": self.container_name_to_container_id[
                        cp.required_name
                    ],
                    "proxy_name": cp.proxy_name,
                    "proxy_name_id": self.container_name_to_container_id[
                        cp.proxy_name
                    ],
                    "vol_scaling_factor": float(
                        cp.vol_scaling_factor
                        if cp.vol_scaling_factor is not None
                        else 1.0
                    ),
                }
            )

        # Process risk factor proxies
        for rp in self.risk_factor_proxies_dict.values():
            records.append(
                {
                    "proxying_level": rp.proxying_level,
                    "required_name": rp.required_name,
                    "required_name_id": self.rf_name_to_rf_id[rp.required_name],
                    "proxy_name": rp.proxy_name,
                    "proxy_name_id": self.rf_name_to_rf_id[rp.proxy_name],
                    "vol_scaling_factor": float(
                        rp.vol_scaling_factor
                        if rp.vol_scaling_factor is not None
                        else 1.0
                    ),
                }
            )

        df = pd.DataFrame(
            records,
            columns=[
                "proxying_level",
                "required_name",
                "required_name_id",
                "proxy_name",
                "proxy_name_id",
                "vol_scaling_factor",
            ],
        )
        return df

    @property
    def df(self) -> pd.DataFrame:
        """Alias for .to_df() for quick notebook access."""
        return self.to_df()
