# src/riskquant/config/proxy_config/schema_objects.py


from dataclasses import dataclass

from typing import Optional


@dataclass(frozen=True, slots=True)
class ProxyMap:
    proxying_level: str  # "container" or "risk_factor"
    required_name: str
    proxy_name: str
    vol_scaling_factor: Optional[float]
