# src/riskquant/portfolio/ingest/builders/__init__.py
"""
Facade for the TradeRecord builders.
Exposes all builder functions so the dispatcher can import them cleanly.
"""

# Assuming you put the functions in files named after the instruments:
from .spot import build_spot
from .forward import (
    build_forward_node,
    build_forward_curve,
)
from .vanilla_european_option import (
    build_vanilla_european_option_node,
    build_vanilla_european_option_curve,
)

# Explicitly declare what is available to the outside world
__all__ = [
    "build_spot",
    "build_forward_node",
    "build_forward_curve",
    "build_vanilla_european_option_node",
    "build_vanilla_european_option_curve",
]
