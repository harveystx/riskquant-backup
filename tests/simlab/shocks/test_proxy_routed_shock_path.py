# tests/simlab/shocks/test_proxy_routed_shock_path.py
"""
Proxy-routed shock path over the spot universe.

Subject: the simlab stages that turn source shocks into shocked levels --
`build_shock_universe` routing and `apply_shocks_to_base_levels` -- driven by
the production proxy resolution. The scenario is assembled once in
`tests/_support/spot_scenario.py`; the P&L the calculator produces from the
same levels is asserted in `tests/solutions/sensitivity_pnl/test_scenario_pnl.py`.

Every expected level is derived from the LOG_RETURN identity
`level = base * exp(scale * shock)`, never by re-running the kernel.

`SHOCKS` and `SOURCE_NAMES` come from `tests/_support/spot_scenario.py`; the
directory conftest puts `tests/_support` on sys.path before collection.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.proxy_resolver.proxy import RES_DIRECT, RES_ID_PROXY
from riskquant.simlab.scenarios.base import ScenarioSetBatch
from riskquant.simlab.shocks.apply import apply_shocks_to_base_levels
from tests._support.spot_scenario import SHOCKS, SOURCE_NAMES

# =============================================================================
# Scenario construction conventions
# =============================================================================


def test_scenario_matrix_dtype_and_contiguity(raw_shocks) -> None:
    """`ScenarioSetBatch.shocks` is (S, K) float32 and MUST be F-contiguous."""
    assert raw_shocks.dtype == np.float32
    assert raw_shocks.shape == (3, 3)
    assert raw_shocks.flags["F_CONTIGUOUS"]


def test_shocked_levels_layout(resolved) -> None:
    """The levels buffer is (N_req, S) float64 C-contiguous."""
    shocked = resolved["shocked"]
    assert shocked.dtype == np.float64
    assert shocked.shape == (4, 3)
    assert shocked.flags["C_CONTIGUOUS"]


def test_required_factor_axis_ordering(rsc, resolved) -> None:
    """Factor rows are the required rf_ids, ascending, and include WTI."""
    required = resolved["required"].tolist()
    assert required == sorted(required)
    assert set(required) == {
        int(rsc.rel_name_to_rf_id[n])
        for n in ("USDEUR", "AAPL_EQ", "BRENT_CRUDE", "WTI_CRUDE")
    }


# =============================================================================
# Proxy application
# =============================================================================


def test_wti_resolves_to_brent_with_095_scale(rsc, resolved) -> None:
    res = resolved["proxy_resolution"]
    i = resolved["row_of"][int(rsc.rel_name_to_rf_id["WTI_CRUDE"])]
    assert int(res.source_factor_ids[i]) == int(rsc.rel_name_to_rf_id["BRENT_CRUDE"])
    assert float(res.shock_scale[i]) == pytest.approx(0.95, abs=1e-7)
    assert int(res.resolution_code[i]) == int(RES_ID_PROXY)


def test_unproxied_factors_resolve_directly(rsc, resolved) -> None:
    res = resolved["proxy_resolution"]
    for name in ("USDEUR", "AAPL_EQ", "BRENT_CRUDE"):
        i = resolved["row_of"][int(rsc.rel_name_to_rf_id[name])]
        assert int(res.source_factor_ids[i]) == int(rsc.rel_name_to_rf_id[name])
        assert float(res.shock_scale[i]) == pytest.approx(1.0, abs=1e-7)
        assert int(res.resolution_code[i]) == int(RES_DIRECT)


def test_wti_shock_column_is_brents(rsc, resolved) -> None:
    """WTI must be routed to BRENT's shock column, not its own."""
    su = resolved["shock_universe"]
    wti_i = resolved["row_of"][int(rsc.rel_name_to_rf_id["WTI_CRUDE"])]
    brent_i = resolved["row_of"][int(rsc.rel_name_to_rf_id["BRENT_CRUDE"])]
    assert su.required_to_shock_uix[wti_i] == su.required_to_shock_uix[brent_i]


def test_base_levels_are_not_altered_by_proxying(rsc, resolved) -> None:
    """Proxying changes shock provenance ONLY. Base levels are untouched."""
    base = resolved["base_levels"]
    row = resolved["row_of"]
    assert float(base[row[int(rsc.rel_name_to_rf_id["WTI_CRUDE"])]]) == pytest.approx(
        70.00, abs=1e-4
    )
    assert float(base[row[int(rsc.rel_name_to_rf_id["BRENT_CRUDE"])]]) == pytest.approx(
        82.30, abs=1e-4
    )
    assert float(base[row[int(rsc.rel_name_to_rf_id["USDEUR"])]]) == pytest.approx(
        1.0800, abs=1e-6
    )


# =============================================================================
# LOG_RETURN semantics
# =============================================================================


def test_log_return_is_exponential_not_additive(rsc, resolved, raw_shocks) -> None:
    """
    level = base * exp(shock), NOT base * (1 + shock).

    At a 2% return the two differ by ~0.02% of the level -- far above the
    tolerance here, so an additive slip cannot pass.
    """
    row = resolved["row_of"]
    brent_i = row[int(rsc.rel_name_to_rf_id["BRENT_CRUDE"])]
    base = float(resolved["base_levels"][brent_i])

    for s in range(len(SHOCKS)):
        shock = float(raw_shocks[s, SOURCE_NAMES.index("BRENT_CRUDE")])
        produced = float(resolved["shocked"][brent_i, s])
        assert produced == pytest.approx(base * math.exp(shock), abs=1e-5)
        if abs(shock) > 1e-9:
            assert abs(produced - base * (1.0 + shock)) > 1e-4


def test_proxy_scale_applies_inside_the_exponent(rsc, resolved, raw_shocks) -> None:
    """WTI level = 70 * exp(0.95 * brent_shock), not 0.95 * (a Brent level)."""
    row = resolved["row_of"]
    wti_i = row[int(rsc.rel_name_to_rf_id["WTI_CRUDE"])]
    for s in range(len(SHOCKS)):
        brent_shock = float(raw_shocks[s, SOURCE_NAMES.index("BRENT_CRUDE")])
        assert float(resolved["shocked"][wti_i, s]) == pytest.approx(
            70.00 * math.exp(0.95 * brent_shock), abs=1e-5
        )


# =============================================================================
# dtype seam: ProxyResolution.shock_scale (float32) -> kernel (float64)
# =============================================================================


def test_float32_proxy_shock_scale_is_accepted_by_shock_application() -> None:
    """
    `ProxyResolution.shock_scale` is float32 by contract (proxy_audit enforces
    it), while `_kernel_apply_shocks` is dtype-pinned to a float64 scale. The
    promotion happens at the execution boundary in `apply_shocks_to_base_levels`.

    Before that promotion existed the production path raised TypeError, so this
    regression test stays even if other tests happen to cover the path.
    """
    base = np.array([70.0, 82.30], dtype=np.float64)
    shocks = np.asfortranarray(
        np.array([[0.02, 0.01], [0.03, -0.01], [0.00, 0.02]], dtype=np.float32)
    )
    for dtype in (np.float32, np.float64):
        batch = ScenarioSetBatch(
            shocks=shocks,
            required_to_shock_uix=np.array([0, 1], dtype=np.int32),
            shock_scale=np.array([0.95, 1.0], dtype=dtype),
            required_shift_type_id=np.array([SHIFT_LOG_RETURN_ID] * 2, dtype=np.int8),
            scenario_offset=0,
        )
        out = np.empty((2, 3), dtype=np.float64, order="C")
        apply_shocks_to_base_levels(base_levels=base, batch=batch, out=out)

        # Independent arithmetic check, not a comparison against the kernel.
        # Tolerance is set by the float32 shock operand (~1e-7 relative), not
        # loosened to hide an error: an additive or unscaled slip would be
        # ~1e-2 on these levels, four orders of magnitude larger.
        for s in range(3):
            assert out[0, s] == pytest.approx(
                70.0 * math.exp(0.95 * float(shocks[s, 0])), rel=1e-6
            )
            assert out[1, s] == pytest.approx(
                82.30 * math.exp(1.0 * float(shocks[s, 1])), rel=1e-6
            )


def test_proxy_resolution_shock_scale_remains_float32(resolved) -> None:
    """The canonical ProxyResolution dtype must NOT have been widened."""
    assert resolved["proxy_resolution"].shock_scale.dtype == np.float32
