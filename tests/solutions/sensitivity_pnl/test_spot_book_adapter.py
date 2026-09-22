# tests/solutions/sensitivity_pnl/test_spot_book_adapter.py
"""
`book_adapter` on the spot sensitizer's real output.

Subject: `build_sensitivity_book_spec` -> `SensitivityPnlCalculator.compile`.
The Greeks come from the shared spot universe (`tests/_support`); their values
are asserted in `tests/sensitivities`. Here the question is only whether the
production output compiles into a book with the expected identity and shape.
"""

from __future__ import annotations

import numpy as np

from riskquant.solutions.sensitivity_pnl.api import (
    SensitivityPnlCalculator,
    build_sensitivity_book_spec,
)


def test_book_spec_compiles_through_the_production_calculator(
    rsc, sensitivities, position_uids
) -> None:
    """The numerical output must compile through the real book compiler."""
    aapl_uid, brent_uid, wti_uid = position_uids
    spec = build_sensitivity_book_spec(
        sensitivities=sensitivities,
        required_factor_ids=sensitivities.active_factor_rf_ids,
    )
    calculator = SensitivityPnlCalculator.compile(spec)

    book = calculator.book
    assert book.position_uids.tolist() == [aapl_uid, brent_uid, wti_uid]
    assert book.active_factor_ids.dtype == np.int32
    assert set(book.active_factor_ids.tolist()) == {
        int(rsc.rel_name_to_rf_id[n])
        for n in ("USDEUR", "AAPL_EQ", "BRENT_CRUDE", "WTI_CRUDE")
    }
    assert book.first_order.shape[0] == 4
    assert book.cross_second_order.shape[0] == 1
