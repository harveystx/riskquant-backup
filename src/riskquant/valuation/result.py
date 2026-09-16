# src/riskquant/valuation/result.py
"""
Valuation results container.

Row/trade cardinality contract
------------------------------
`values` has ONE ROW PER BLOCK ROW, not one row per trade.

`Block.trade_uids[i]` identifies the trade that PRODUCED row `i`. A single
economic trade may produce many rows: `block_type.py` documents

    VANILLA_IRS
        |-- FIXED_CASHFLOW
        +-- FLOATING_CASHFLOW

so an interest-rate swap contributes one row per cashflow, every one of them
carrying the same `trade_uid`.

Consequences, all of which this container must honour (D045/VAL-05):

    * `trade_uids` is NOT unique.
    * `trade_uids` is NOT sorted. `CompiledPortfolio` concatenates each block's
      uids in block-slot order, so the array is grouped by block, not ordered
      by uid.

`np.searchsorted` is unsound on both counts and was previously used here: on
the adversarial ordering [9007, 41, 120003, 5] it raised `KeyError` for three
of four PRESENT uids, and with duplicate uids it could return one cashflow
leg's PV as though it were the trade's PV.

Lookup is therefore by boolean mask, and the two questions a caller can ask
are kept distinct:

    get_trade_values(uid) -> (n_rows_for_uid, n_scenarios)   every row
    get_trade_pv(uid)     -> (n_scenarios,)                  their sum

Sorting `trade_uids` to make a binary search valid would fix the tool rather
than the contract, and would destroy the block-row ordering that
`portfolio.block_offsets` depends on.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class ValuationResult:
    """
    PV matrix over one scenario batch, keyed by originating trade.

    Attributes
    ----------
    trade_uids
        `(n_rows,)` int64. `trade_uids[i]` is the trade that produced row `i`.
        Neither unique nor sorted -- see the module docstring.
    values
        `(n_rows, n_scenarios)` float64 PVs in settlement currency.
    """

    trade_uids: npt.NDArray[np.int64]  # (n_rows,)
    values: npt.NDArray[np.float64]  # (n_rows, n_scenarios)

    def __post_init__(self) -> None:
        if self.trade_uids.ndim != 1:
            raise ValueError("trade_uids must be 1-D")
        if self.values.ndim != 2:
            raise ValueError("values must be 2-D")
        if self.values.shape[0] != self.trade_uids.shape[0]:
            raise ValueError("UIDs and values row mismatch")

    @property
    def num_rows(self) -> int:
        """Block rows in the result. NOT the number of economic trades."""
        return self.values.shape[0]

    @property
    def num_trades(self) -> int:
        """
        Distinct originating trades.

        Equals `num_rows` only for a portfolio of one-row-per-trade block types
        (spot, forward, option). A cashflow-decomposed instrument contributes
        several rows and one trade.
        """
        return int(np.unique(self.trade_uids).shape[0])

    @property
    def num_scenarios(self) -> int:
        return self.values.shape[1]

    def get_trade_values(self, uid: int) -> npt.NDArray[np.float64]:
        """
        Every row produced by `uid`, shape `(n_rows_for_uid, n_scenarios)`.

        Returns one row for a spot/forward/option trade and one row per
        cashflow for a decomposed instrument. Use `get_trade_pv` for the
        trade-level present value.

        Raises
        ------
        KeyError
            If `uid` produced no rows in this result.
        """
        mask = self.trade_uids == uid
        if not bool(mask.any()):
            raise KeyError(uid)
        return self.values[mask]

    def get_trade_pv(self, uid: int) -> npt.NDArray[np.float64]:
        """
        Trade-level PV across scenarios, shape `(n_scenarios,)`.

        Sums every row the trade produced. This is the aggregation the
        row/trade cardinality contract requires; `get_trade_values` alone is
        not a trade PV whenever the trade decomposes into cashflows.

        Raises
        ------
        KeyError
            If `uid` produced no rows in this result.
        """
        return self.get_trade_values(uid).sum(axis=0)

    def select(self, uids: npt.NDArray[np.int64]) -> "ValuationResult":
        """Every row produced by any uid in `uids`, order preserved."""
        idx = np.isin(self.trade_uids, uids)
        return ValuationResult(
            trade_uids=self.trade_uids[idx],
            values=self.values[idx],
        )
