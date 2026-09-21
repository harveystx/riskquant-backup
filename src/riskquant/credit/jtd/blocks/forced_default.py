# src/riskquant/credit/jtd/blocks/forced_default.py
"""
Forced-default JTD block.

The forced-default model is scenario-independent: it ignores the
`CreditFactorMatrix` entirely and assumes default with certainty, so the block
carries no copula or factor state — only the `_JtdBlock` base terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from riskquant.credit.jtd.blocks.base import _JtdBlock
from riskquant.credit.jtd.models import FORCED_DEFAULT_ID


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class ForcedDefaultJtdBlock(_JtdBlock):
    """Worst-case JTD: loss_i = ead_i * lgd_i (default indicator forced to 1)."""

    JTD_MODEL_ID: ClassVar[int] = FORCED_DEFAULT_ID
