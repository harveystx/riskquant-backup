# tests/simlab/scenarios/test_parametric_whole_run_stream.py
"""
PR-06: the parametric stochastic world is one whole-run realization (D029,
D037, D087).

Contract proven here, at the level of the canonical random buffer:

    default_rng(seed)                                    one RNG per iter_batches
      -> standard_normal into a C-ordered (S, K_total) float64 buffer
         (draw index of cell (t, k) = t * K_total + k)   ONE call, whole run
      -> one auxiliary draw of size S for a non-normal law, after all normals
      -> per batch: rows [o, o + B) of that buffer, times L^T (columns of the
         required factors), times the law's row scale, times the factor vols,
         rounded ONCE to float32
    batch_size only chooses the row slices.

The oracle reproduces the stream from `numpy.random.default_rng` directly and
the transformation from the model's formulas; it never calls the generator.
EXACT for the random buffer and the auxiliary draw; APPROXIMATE (one float32
ulp, the only admitted difference being the float64 summation order of the
Cholesky product before the single float32 rounding) for the shocks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from riskquant.simlab.scenarios.parametric import ParametricGenerator

_S = 23
_IDS = np.array([3, 11, 17, 29], dtype=np.int32)  # K_total = 4, ascending
_CORR = np.array([
    [1.0, 0.3, -0.2, 0.1],
    [0.3, 1.0, 0.25, 0.0],
    [-0.2, 0.25, 1.0, 0.4],
    [0.1, 0.0, 0.4, 1.0],
])
_CHOL = np.linalg.cholesky(_CORR)
_VOLS = np.array([0.021, 0.013, 0.034, 0.008])
_SEED = 41

_LAWS: dict[str, dict[str, Any]] = {
    "standard_normal": {},
    "student_t": {"student_t_df": 5.0},
    "laplace": {},
    "variance_gamma": {"vg_nu": 0.7},
    "regime_switching_gmm": {
        "prob_crisis": 0.3,
        "crisis_vol_multipliers": np.array([2.0, 1.5, 3.0, 1.2]),
    },
}


def _generator(kind: str, seed: int = _SEED, total: int = _S) -> ParametricGenerator:
    return ParametricGenerator(
        available_factor_ids=_IDS,
        cholesky_matrix=_CHOL,
        volatilities=_VOLS,
        total_scenarios=total,
        distribution_kind=kind,
        seed=seed,
        **_LAWS[kind],
    )


def _stream(
    gen: ParametricGenerator, required: np.ndarray, batch_size: int
) -> np.ndarray:
    n = required.shape[0]
    out = [
        np.array(b.shocks, copy=True)
        for b in gen.iter_batches(
            required_shift_type_id=np.ones(n, dtype=np.int8),
            source_factor_ids=required,
            shock_scale=np.ones(n, dtype=np.float32),
            batch_size=batch_size,
        )
    ]
    return np.vstack(out)


class _Recorder:
    """Transparent proxy over the real Generator: records every draw call as
    (method, shape, values, C-contiguous, dtype)."""

    def __init__(self, seed: Any, log: list[tuple[Any, ...]]) -> None:
        self._g = _REAL_DEFAULT_RNG(seed)
        self._log = log

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._g, name)
        if not callable(target):
            return target

        def call(*args: Any, **kwargs: Any) -> Any:
            result = target(*args, **kwargs)
            value = (
                kwargs["out"] if kwargs.get("out") is not None else np.asarray(result)
            )
            self._log.append((
                name,
                value.shape,
                np.array(value, copy=True),
                bool(value.flags["C_CONTIGUOUS"]),
                value.dtype,
            ))
            return result

        return call


_REAL_DEFAULT_RNG = np.random.default_rng


@pytest.fixture
def draws(monkeypatch):
    log: list[tuple[str, Any, np.ndarray]] = []
    monkeypatch.setattr(
        np.random, "default_rng", lambda seed=None: _Recorder(seed, log)
    )
    return log


def _oracle(kind: str, seed: int = _SEED, total: int = _S):
    """Stream and transformation, from numpy and the model's formulas only."""
    rng = _REAL_DEFAULT_RNG(seed)
    z = rng.standard_normal(total * _IDS.size).reshape(total, _IDS.size)  # C order
    law = _LAWS[kind]
    if kind == "standard_normal":
        aux = None
        scale = np.ones(total)
    elif kind == "student_t":
        aux = rng.chisquare(df=law["student_t_df"], size=total)
        scale = np.sqrt((law["student_t_df"] - 2.0) / aux)
    elif kind == "laplace":
        aux = rng.exponential(scale=1.0, size=total)
        scale = np.sqrt(aux)
    elif kind == "variance_gamma":
        aux = rng.gamma(shape=1.0 / law["vg_nu"], scale=law["vg_nu"], size=total)
        scale = np.sqrt(aux)
    else:
        aux = rng.random(total)
        scale = np.ones(total)
    return z, aux, scale


def _oracle_shocks(kind: str, required: np.ndarray, seed: int = _SEED, total: int = _S):
    z, aux, scale = _oracle(kind, seed, total)
    cols = [int(np.flatnonzero(f == _IDS)[0]) for f in required.tolist()]
    out = np.empty((total, len(cols)))
    for j, c in enumerate(cols):
        # (Z L^T)[:, c] = sum_k Z[:, k] L[c, k]: an explicit sum, not BLAS
        acc = np.zeros(total)
        for k in range(_IDS.size):
            acc = acc + z[:, k] * _CHOL[c, k]
        out[:, j] = acc
    out *= scale[:, None]
    if kind == "regime_switching_gmm":
        mult = _LAWS[kind]["crisis_vol_multipliers"][cols]
        out[aux < _LAWS[kind]["prob_crisis"], :] *= mult
    out *= _VOLS[cols]
    return out.astype(np.float32)


# ---- 1. the whole-run buffer and the draw order (EXACT) ------------------------------


@pytest.mark.parametrize("kind", list(_LAWS))
def test_one_whole_run_draw_in_canonical_order(draws, kind) -> None:
    _stream(_generator(kind), _IDS[[2, 0]], batch_size=5)
    names = [d[0] for d in draws]
    expected = ["standard_normal"] + (
        []
        if kind == "standard_normal"
        else [
            {
                "student_t": "chisquare",
                "laplace": "exponential",
                "variance_gamma": "gamma",
                "regime_switching_gmm": "random",
            }[kind]
        ]
    )
    assert names == expected  # one normal fill for the whole run, then one aux
    assert draws[0][1] == (_S, _IDS.size)  # the whole run, in one fill
    assert draws[0][3] and draws[0][4] == np.dtype(np.float64)  # C order, float64
    z, aux, _ = _oracle(kind)
    assert draws[0][2].tobytes() == z.tobytes()  # cell (t, k) = draw t * K + k
    if aux is not None:
        assert draws[-1][2].tobytes() == aux.tobytes()


@pytest.mark.parametrize("kind", list(_LAWS))
@pytest.mark.parametrize("batch_size", [1, 2, 3, 7, 13, _S - 1, _S, _S + 1])
def test_random_buffer_and_stream_are_batch_invariant(draws, kind, batch_size) -> None:
    whole = _stream(_generator(kind), _IDS, batch_size=_S)
    reference = [d[2].tobytes() for d in draws]
    draws.clear()
    parts = _stream(_generator(kind), _IDS, batch_size=batch_size)
    assert [d[2].tobytes() for d in draws] == reference  # same random buffer
    assert parts.tobytes() == whole.tobytes()  # same concatenated shocks


# ---- 2. transformation against the independent oracle (APPROXIMATE) ------------------


@pytest.mark.parametrize("kind", list(_LAWS))
@pytest.mark.parametrize("batch_size", [1, 6, _S])
def test_shocks_equal_the_independent_construction(kind, batch_size) -> None:
    required = _IDS[[3, 1, 0]]  # a permuted subset of the data's factors
    got = _stream(_generator(kind), required, batch_size=batch_size)
    exp = _oracle_shocks(kind, required)
    assert got.dtype == np.float32 and got.shape == exp.shape
    assert (
        np.abs(got.astype(np.float64) - exp.astype(np.float64))
        <= np.spacing(np.abs(exp)).astype(np.float64)
    ).all()


# ---- 3. the axis selects columns; it never changes the world (EXACT) -----------------


@pytest.mark.parametrize("kind", list(_LAWS))
def test_requested_columns_do_not_perturb_each_other(kind) -> None:
    """Adding a factor to the requested set (e.g. a reporting-only suffix row)
    or permuting it leaves every other factor's shocks bitwise unchanged."""
    only = _stream(_generator(kind), _IDS[[1]], batch_size=4)
    more = _stream(_generator(kind), _IDS[[1, 3]], batch_size=4)
    perm = _stream(_generator(kind), _IDS[[3, 1]], batch_size=4)
    assert only[:, 0].tobytes() == more[:, 0].tobytes()
    assert np.array_equal(perm[:, [1, 0]], more)


def test_the_world_is_seeded_by_the_seed_alone(draws) -> None:
    a = _stream(_generator("student_t", seed=7), _IDS, 5)
    za = draws[0][2]
    draws.clear()
    b = _stream(_generator("student_t", seed=7), _IDS, 11)
    zb = draws[0][2]
    draws.clear()
    _stream(_generator("student_t", seed=8), _IDS, 5)
    zc = draws[0][2]
    assert za.tobytes() == zb.tobytes() and a.tobytes() == b.tobytes()
    assert za.tobytes() != zc.tobytes()


@pytest.mark.parametrize("kind", list(_LAWS))
def test_float32_rounding_happens_once_after_all_scaling(kind) -> None:
    """EXACT. With an identity Cholesky factor the product is exact, so the
    only rounding left is the single float32 cast after row scale and vols;
    an earlier or repeated rounding point changes the bits."""
    gen = ParametricGenerator(
        available_factor_ids=_IDS,
        cholesky_matrix=np.eye(_IDS.size),
        volatilities=_VOLS,
        total_scenarios=_S,
        distribution_kind=kind,
        seed=_SEED,
        **_LAWS[kind],
    )
    got = _stream(gen, _IDS, batch_size=6)
    z, aux, scale = _oracle(kind)
    exp = z * scale[:, None]
    if kind == "regime_switching_gmm":
        exp[aux < _LAWS[kind]["prob_crisis"], :] *= _LAWS[kind][
            "crisis_vol_multipliers"
        ]
    exp = (exp * _VOLS).astype(np.float32)
    assert got.tobytes() == exp.tobytes()
