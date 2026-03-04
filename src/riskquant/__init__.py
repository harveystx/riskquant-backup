from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("riskquant")
except PackageNotFoundError:
    # Package is not installed (e.g. running from source without install)
    __version__ = "0.0.0"
