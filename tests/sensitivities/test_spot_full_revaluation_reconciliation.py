# tests/sensitivities/test_spot_full_revaluation_reconciliation.py
"""
Spot Greeks emitted by `riskquant.sensitivities` versus FULL REVALUATION.

The spot payoffs of the shared universe are polynomial in their factors:

    WTI    USD-priced / USD-settled    V = Q*S      affine
    Brent  USD-priced / EUR-settled    V = Q*B*F    bilinear

so a Taylor expansion built from the sensitizer's own output must reproduce
full revaluation up to rounding: first order is exact for WTI; second order
WITH the cross term is exact for Brent. Dropping the emitted cross term must
leave a residual of exactly Q*dB*dF.

The oracle reprices from the instrument identity -- Q from the position, base
levels from the golden snapshot -- never from the Greeks. The Taylor side uses
the emitted rows as they stand, so a missing, extra or mis-scaled Greek fails.
"""

from __future__ import annotations

import numpy as np

from riskquant.sensitivities.result import BlockSensitivities

#: float64 unit roundoff. Tolerances are a few ulps of the largest product in
#: the valuation, so an error of any economic size cannot hide under them.
_EPS = float(np.finfo(np.float64).eps)


def _taylor_pnl(
    sens: BlockSensitivities,
    uid: int,
    displacement: dict[int, np.ndarray],
    *,
    with_cross: bool = True,
) -> np.ndarray:
    """
    Second-order Taylor P&L of one position from its emitted Greeks.

    Diagonal terms carry 1/2; each emitted cross pair (i, j) appears once with
    no 1/2 (D081). A Greek on a factor absent from `displacement` raises
    KeyError: the position must not be exposed to a factor the payoff lacks.
    """
    pnl = np.zeros_like(next(iter(displacement.values())))
    mask = sens.position_uids == uid
    for rf, d1, d2 in zip(
        sens.factor_rf_ids[mask].tolist(),
        sens.first_order[mask].tolist(),
        sens.second_order[mask].tolist(),
        strict=True,
    ):
        dx = displacement[rf]
        pnl += d1 * dx + 0.5 * d2 * dx * dx
    if with_cross:
        cmask = sens.cross_position_uids == uid
        for rf_i, rf_j, c in zip(
            sens.cross_factor_rf_ids_i[cmask].tolist(),
            sens.cross_factor_rf_ids_j[cmask].tolist(),
            sens.cross_second_order[cmask].tolist(),
            strict=True,
        ):
            pnl += c * displacement[rf_i] * displacement[rf_j]
    return pnl


def _quantity(raw_positions, uid: int) -> float:
    return float(raw_positions.loc[raw_positions["uid"] == uid, "quantity"].iloc[0])


def _brent_quanto_case(rsc, golden_levels, raw_positions, position_uids):
    _, brent_uid, _ = position_uids
    return (
        brent_uid,
        _quantity(raw_positions, brent_uid),
        int(rsc.rel_name_to_rf_id["BRENT_CRUDE"]),
        int(rsc.rel_name_to_rf_id["USDEUR"]),
        float(golden_levels["BRENT_CRUDE"]),
        float(golden_levels["USDEUR"]),
    )


def test_linear_spot_reconciles_exactly_under_full_revaluation(
    rsc, sensitivities, golden_levels, raw_positions, position_uids
) -> None:
    """V = Q*S is affine: the emitted first order reproduces full revaluation."""
    _, _, wti_uid = position_uids
    wti_rf = int(rsc.rel_name_to_rf_id["WTI_CRUDE"])
    qty = _quantity(raw_positions, wti_uid)
    s0 = float(golden_levels["WTI_CRUDE"])

    log_shocks = np.array([0.01, -0.02, 0.03, 0.0], dtype=np.float64)
    levels = s0 * np.exp(log_shocks)

    full_reval = qty * levels - qty * s0
    taylor = _taylor_pnl(sensitivities, wti_uid, {wti_rf: levels - s0})

    tol = 8.0 * _EPS * float(np.max(np.abs(qty * levels)))
    np.testing.assert_allclose(taylor, full_reval, rtol=0.0, atol=tol)


def test_quanto_reconciles_exactly_with_the_cross_term(
    rsc, sensitivities, golden_levels, raw_positions, position_uids
) -> None:
    """
    V = Q*B*F is bilinear: first order plus the emitted cross term is exact,
    and removing that cross term leaves exactly Q*dB*dF.
    """
    uid, qty, b_rf, f_rf, b0, f0 = _brent_quanto_case(
        rsc, golden_levels, raw_positions, position_uids
    )
    shocked_b = np.array([80.670354, 83.543808, 81.481104])
    shocked_f = np.array([1.080000, 1.069254, 1.101817])
    displacement = {b_rf: shocked_b - b0, f_rf: shocked_f - f0}

    full_reval = qty * shocked_b * shocked_f - qty * b0 * f0
    with_cross = _taylor_pnl(sensitivities, uid, displacement)
    without_cross = _taylor_pnl(sensitivities, uid, displacement, with_cross=False)

    tol = 32.0 * _EPS * float(np.max(np.abs(qty * shocked_b * shocked_f)))
    np.testing.assert_allclose(with_cross, full_reval, rtol=0.0, atol=tol)

    residual = full_reval - without_cross
    np.testing.assert_allclose(
        residual, qty * displacement[b_rf] * displacement[f_rf], rtol=0.0, atol=tol
    )
    # The cross term is not negligible where FX actually moves.
    assert abs(residual[2]) > 1e-3


def test_quanto_reconciles_across_many_scenarios(
    rsc, sensitivities, golden_levels, raw_positions, position_uids
) -> None:
    """
    Same reconciliation over a seeded sweep of joint log-return moves. The
    sweep uses NumPy's RNG, not a riskquant generator: the generator is a
    simlab concern and is tested there.
    """
    uid, qty, b_rf, f_rf, b0, f0 = _brent_quanto_case(
        rsc, golden_levels, raw_positions, position_uids
    )
    rng = np.random.default_rng(20_260_913)
    shocks = rng.standard_normal((128, 2)) * np.array([0.020, 0.006])
    level_b = b0 * np.exp(shocks[:, 0])
    level_f = f0 * np.exp(shocks[:, 1])

    full_reval = qty * level_b * level_f - qty * b0 * f0
    taylor = _taylor_pnl(sensitivities, uid, {b_rf: level_b - b0, f_rf: level_f - f0})

    tol = 32.0 * _EPS * float(np.max(np.abs(qty * level_b * level_f)))
    np.testing.assert_allclose(taylor, full_reval, rtol=0.0, atol=tol)
