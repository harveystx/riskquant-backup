# tests/market_data/remediation/conftest.py
"""Make the test-owned oracle and hostile fixture importable by module name."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
