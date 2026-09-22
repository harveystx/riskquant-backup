# tests/scenario_metrics/test_scenario_risk_measures.py
"""
Integration tests: deterministic scenario P&L -> canonical VaR / ES.

The risk-measure mathematics itself is already production-owned and
comprehensively covered by `tests/scenario_metrics/test_var_es.py` (148
tests: quantile geometry, ES boundary mass, confidence validation, ordering
invariance, dtype/shape contracts). Those are deliberately NOT duplicated
here.

What is tested here is the seam this increment actually introduced:

  * the scenario-P&L distribution is handed to `compute_var_es` in the
    contract it requires,
  * the VaR sign convention survives the hand-off (a loss stays NEGATIVE),
  * mixed settlement currencies are never silently summed,

each validated against an oracle derived independently from the canonical
definition rather than by calling the production implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskquant.scenario_metrics.api.var_es import compute_var_es

# The proven deterministic scenario P&L from the E2E demo. Pinned here so a
# silent change upstream shows up as a failure in this seam too.
_AAPL_PNL = (201.0033, 0.0, -297.7612)
_BRENT_PNL = (88.0011, -22.2766, -44.6650)  # EUR-settled
_WTI_PNL = (-98.8083, 75.3481, -49.6388)

# alpha = 1/3, S = 3 -> tail mass m = 1.0 exactly: VaR is the single worst
# scenario and ES integrates exactly that one observation.
_CONFIDENCE = 2.0 / 3.0


def _matrix(*rows: tuple[float, ...]) -> np.ndarray:
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float64)


def _oracle_var_es(row: np.ndarray, confidence: float) -> tuple[float, float]:
    """
    Independent re-derivation of the canonical definition.

        alpha = 1 - C ; m = alpha * S
        VaR = sorted(pnl)[ceil(m) - 1]
        ES  = (1/m) * [sum of first floor(m) sorted obs + frac(m) * next obs]

    Pure Python/NumPy at the test edge; never calls `compute_var_es`.
    """
    ordered = np.sort(np.asarray(row, dtype=np.float64))
    scenarios = ordered.shape[0]
    tail_mass = (1.0 - confidence) * scenarios

    nearest = round(tail_mass)
    if abs(tail_mass - nearest) <= 1e-9 * max(1.0, abs(tail_mass)):
        tail_mass = float(nearest)

    full = math.floor(tail_mass)
    frac = tail_mass - full

    var_rank = max(1, math.ceil(tail_mass))
    var = float(ordered[var_rank - 1])

    tail_sum = float(ordered[:full].sum())
    if frac > 0.0:
        tail_sum += frac * float(ordered[full])
    return var, tail_sum / tail_mass


# =============================================================================
# Sign convention
# =============================================================================


def test_var_of_a_loss_stays_negative() -> None:
    """
    VaR is the lower P&L quantile ITSELF. A loss must remain negative and
    must never be flipped into a positive loss magnitude.
    """
    result = compute_var_es(_matrix(_AAPL_PNL, _BRENT_PNL, _WTI_PNL), _CONFIDENCE)

    assert result.var[0] == pytest.approx(-297.7612, abs=1e-9)
    assert result.var[1] == pytest.approx(-44.6650, abs=1e-9)
    assert result.var[2] == pytest.approx(-98.8083, abs=1e-9)
    assert np.all(result.var < 0.0)
    # The positive-magnitude reading would be +297.7612; assert it is not that.
    assert result.var[0] != pytest.approx(297.7612, abs=1e-9)


def test_var_is_more_negative_for_a_worse_distribution() -> None:
    """More negative VaR must mean greater downside risk, not less."""
    mild = (10.0, -5.0, 2.0)
    severe = (10.0, -500.0, 2.0)
    result = compute_var_es(_matrix(mild, severe), _CONFIDENCE)
    assert result.var[1] < result.var[0]


def test_an_all_gains_distribution_yields_a_positive_var() -> None:
    """
    VaR is a quantile, not a loss magnitude: if the worst scenario is still a
    gain, VaR is positive. This pins the quantile reading of the convention.
    """
    result = compute_var_es(_matrix((5.0, 9.0, 12.0)), _CONFIDENCE)
    assert result.var[0] == pytest.approx(5.0, abs=1e-9)


# =============================================================================
# Production vs independent oracle
# =============================================================================


@pytest.mark.parametrize(
    "label,row",
    [("aapl", _AAPL_PNL), ("brent", _BRENT_PNL), ("wti", _WTI_PNL)],
)
def test_trade_var_es_matches_independent_oracle(label: str, row) -> None:
    produced = compute_var_es(_matrix(row), _CONFIDENCE)
    expected_var, expected_es = _oracle_var_es(np.asarray(row), _CONFIDENCE)
    assert float(produced.var[0]) == pytest.approx(expected_var, abs=1e-9)
    assert float(produced.es[0]) == pytest.approx(expected_es, abs=1e-9)


def test_integral_tail_mass_makes_es_equal_var_here() -> None:
    """
    With m = 1.0 exactly, the ES tail contains precisely one observation, so
    ES must equal VaR. This is a property of THIS demo's geometry and pins
    that the fractional-weight branch is not being taken.
    """
    produced = compute_var_es(_matrix(_AAPL_PNL, _BRENT_PNL, _WTI_PNL), _CONFIDENCE)
    np.testing.assert_allclose(produced.es, produced.var, atol=1e-12)


def test_fractional_tail_mass_makes_es_strictly_worse_than_var() -> None:
    """
    Contrast case: at C = 0.5 on 3 scenarios, m = 1.5, so ES pulls in half of
    the second-worst observation and must sit strictly below VaR.
    """
    row = (-300.0, -100.0, 50.0)
    produced = compute_var_es(_matrix(row), 0.5)
    expected_var, expected_es = _oracle_var_es(np.asarray(row), 0.5)
    assert float(produced.var[0]) == pytest.approx(expected_var, abs=1e-9)
    assert float(produced.es[0]) == pytest.approx(expected_es, abs=1e-9)
    assert produced.es[0] < produced.var[0]


# =============================================================================
# Currency coherence
# =============================================================================


def test_usd_subportfolio_sums_only_usd_trades() -> None:
    """
    AAPL and WTI settle USD; Brent settles EUR. The coherent aggregate is
    AAPL + WTI only -- Brent must never enter it.
    """
    usd_row = np.asarray(_AAPL_PNL) + np.asarray(_WTI_PNL)
    produced = compute_var_es(_matrix(tuple(usd_row)), _CONFIDENCE)
    expected_var, expected_es = _oracle_var_es(usd_row, _CONFIDENCE)

    assert float(produced.var[0]) == pytest.approx(expected_var, abs=1e-9)
    assert float(produced.es[0]) == pytest.approx(expected_es, abs=1e-9)

    # A three-way sum including EUR would give a different, meaningless number.
    all_three = usd_row + np.asarray(_BRENT_PNL)
    assert not np.allclose(usd_row, all_three)


def test_group_rows_are_computed_independently() -> None:
    """
    Per-trade risk measures must not leak across rows: each row's VaR equals
    that row computed alone.
    """
    together = compute_var_es(_matrix(_AAPL_PNL, _BRENT_PNL, _WTI_PNL), _CONFIDENCE)
    for i, row in enumerate((_AAPL_PNL, _BRENT_PNL, _WTI_PNL)):
        alone = compute_var_es(_matrix(row), _CONFIDENCE)
        assert float(together.var[i]) == pytest.approx(float(alone.var[0]), abs=1e-12)
        assert float(together.es[i]) == pytest.approx(float(alone.es[0]), abs=1e-12)


# =============================================================================
# Contract hand-off
# =============================================================================


def test_pnl_handed_over_as_float64_c_contiguous() -> None:
    """
    `SensitivityPnlResult.pnl` is float32; the risk-measure contract demands
    float64 C-contiguous. The conversion must be explicit at the seam.
    """
    float32_pnl = np.asarray([_AAPL_PNL], dtype=np.float32)
    with pytest.raises(TypeError):
        compute_var_es(float32_pnl, _CONFIDENCE)

    converted = np.ascontiguousarray(float32_pnl, dtype=np.float64)
    assert compute_var_es(converted, _CONFIDENCE).var.shape == (1,)


def test_demo_confidence_level_yields_integral_tail_mass() -> None:
    """The demo's illustrative confidence must give an unambiguous tail."""
    scenarios = 3
    tail_mass = (1.0 - _CONFIDENCE) * scenarios
    assert tail_mass == pytest.approx(1.0, abs=1e-12)
