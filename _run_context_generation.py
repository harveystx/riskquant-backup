# _run_context_generation.py
"""
Entry point for RiskQuant context-snapshot generation.

Note: this file lives at the repository root, not under src/riskquant/ --
the previous version of this file carried a stale `# src/riskquant/
_create_master_context.py` first-line comment that did not match its real
location. Corrected here per the estate's own file-header convention.
"""

from __future__ import annotations

import sys

from _context_gen.cli import main

if __name__ == "__main__":
    sys.exit(main())
