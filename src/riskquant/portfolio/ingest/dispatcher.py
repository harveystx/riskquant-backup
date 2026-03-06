# src/riskquant/portfolio/ingest/dispatcher.py

from typing import Callable, Iterator

from riskquant.portfolio.trade import TradeRecord
from riskquant.portfolio.instruments import (
    SPOT_ID,
    FORWARD_NODE_ID,
    FORWARD_CURVE_ID,
    VANILLA_EUROPEAN_OPTION_NODE_ID,
    VANILLA_EUROPEAN_OPTION_CURVE_ID,
)
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.ingest.builders import (
    build_spot,
    build_forward_node,
    build_forward_curve,
    build_vanilla_european_option_node,
    build_vanilla_european_option_curve,
)

BuilderFunc = Callable[..., TradeRecord]

# O(1) Routing table - No more if/elif branches!
DISPATCH_TABLE: dict[int, BuilderFunc] = {
    int(SPOT_ID): build_spot,
    int(FORWARD_NODE_ID): build_forward_node,
    int(FORWARD_CURVE_ID): build_forward_curve,
    int(VANILLA_EUROPEAN_OPTION_NODE_ID): build_vanilla_european_option_node,
    int(VANILLA_EUROPEAN_OPTION_CURVE_ID): build_vanilla_european_option_curve,
}


def iter_trades_from_batch(batch: IngestBatch) -> Iterator[TradeRecord]:
    """
    O(1) dispatch loop. Yields compiled TradeRecords.
    Because validation is already done, this loop is blazingly fast.
    """
    instrument_ids = batch.instrument_ids
    size = batch.size

    for ix in range(size):
        # Look up the correct function using the integer ID, then execute it
        yield DISPATCH_TABLE[instrument_ids[ix]](ix=ix, batch=batch)
