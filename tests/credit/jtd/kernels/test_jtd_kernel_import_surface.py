# tests/credit/jtd/kernels/test_jtd_kernel_import_surface.py
"""
Regression guard for RQ-AUD-014 / audit C-14: every JTD kernel must import
cleanly and execute through its real evaluator.

`riskquant.credit.jtd.kernels.stress_regime_bernoulli` used `_NB_VOID` in its
`@parallel_njit` signature without importing it. `registry.evaluator_table()`
imports every evaluator eagerly, so one kernel's undefined alias made
`jtd.engine`, `jtd.streaming` and the whole of `solutions.credit_var`
unimportable. Nothing caught it:

* RQ-AUD-001 (`tests/test_no_undefined_jit_aliases.py`) checks only the
  OPPOSITE direction — a name imported FROM `riskquant.jit` that `jit.py` does
  not define.
* RQ-AUD-005 (`tests/credit/exposure/kernels/test_cem_import_and_run.py`)
  covers the three EXPOSURE kernels, not the nine JTD kernels.

This module closes both halves for the JTD surface:

1. a static AST check that every `riskquant.jit` alias a credit module USES is
   bound in that module (catches "used but not imported" without executing);
2. an execution check that drives all nine kernels through
   `registry.evaluator_table()` — the production dispatch path — so a kernel
   cannot pass by merely being importable.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtri

import riskquant.jit as jit_module
from riskquant.credit.factors.state import CreditFactorMatrix
from riskquant.credit.jtd.blocks.beta_binomial import BetaBinomialJtdBlock
from riskquant.credit.jtd.blocks.forced_default import ForcedDefaultJtdBlock
from riskquant.credit.jtd.blocks.independent_bernoulli import (
    IndependentBernoulliJtdBlock,
)
from riskquant.credit.jtd.blocks.marshall_olkin import MarshallOlkinJtdBlock
from riskquant.credit.jtd.blocks.mf_copula_gaussian import MfGaussianCopulaJtdBlock
from riskquant.credit.jtd.blocks.mf_copula_students_t import (
    MfStudentsTCopulaJtdBlock,
)
from riskquant.credit.jtd.blocks.sf_copula_gaussian import SfGaussianCopulaJtdBlock
from riskquant.credit.jtd.blocks.sf_copula_students_t import (
    SfStudentsTCopulaJtdBlock,
)
from riskquant.credit.jtd.blocks.stress_regime_bernoulli import (
    StressRegimeBernoulliJtdBlock,
)
from riskquant.credit.jtd.models import JtdModel
from riskquant.credit.jtd.registry import JTD_MODEL_SPECS, evaluator_table

CREDIT_ROOT = Path(__file__).resolve().parents[3] / "src" / "riskquant" / "credit"

# `riskquant.jit` alias naming: scalar `_NB_*` and array `_<DTYPE>_<RANK>_<LAYOUT>`.
_ALIAS_PREFIXES = (
    "_NB_",
    "_F64_",
    "_F32_",
    "_I64_",
    "_I32_",
    "_I16_",
    "_I8_",
    "_BOOL_",
)


def _is_alias(name: str) -> bool:
    return name.startswith(_ALIAS_PREFIXES)


def _bound_names(tree: ast.Module) -> set[str]:
    """Every module-level name the module itself binds (import or assignment)."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
    return bound


def test_every_used_jit_alias_is_bound_in_its_module() -> None:
    """A `riskquant.jit` alias that is USED but never imported is a NameError at
    import time. Static so it reports every offender, not just the first."""
    offenders: dict[str, list[str]] = {}

    for path in sorted(CREDIT_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bound = _bound_names(tree)
        used = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and _is_alias(node.id)
        }
        missing = sorted(used - bound)
        if missing:
            offenders[str(path.relative_to(CREDIT_ROOT))] = missing

    assert not offenders, "riskquant.jit aliases used but not imported:\n" + "\n".join(
        f"  {f}: {names}" for f, names in sorted(offenders.items())
    )


def test_every_imported_jit_alias_exists() -> None:
    """The RQ-AUD-001 direction, scoped to credit: an imported alias that
    `riskquant.jit` does not define."""
    defined = set(dir(jit_module))
    offenders: dict[str, list[str]] = {}

    for path in sorted(CREDIT_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        missing: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "riskquant.jit":
                missing += [
                    a.name for a in node.names if (a.asname or a.name) not in defined
                ]
        if missing:
            offenders[str(path.relative_to(CREDIT_ROOT))] = sorted(missing)

    assert not offenders, "undefined riskquant.jit imports:\n" + "\n".join(
        f"  {f}: {names}" for f, names in sorted(offenders.items())
    )


def test_evaluator_table_resolves_every_model() -> None:
    """`evaluator_table()` eagerly imports all nine kernels: the exact call that
    C-14 broke."""
    table = evaluator_table()
    assert len(table) == len(JtdModel)
    assert all(ev is not None for ev in table)
    assert {spec.model for spec in JTD_MODEL_SPECS} == set(JtdModel)


def _blocks() -> dict[JtdModel, object]:
    """One single-obligor block per model, wired to the factor rows below.

    factor rows: 0 = U(0,1) | 1 = second U(0,1) | 2 = systemic Z | 3 = idio eps
                 4 = 1/sqrt(W) | 5 = stress indicator | 6 = Beta common PD
    """
    uid = np.array([1], dtype=np.int64)
    lgd = np.array([0.6])
    thr = np.array([ndtri(0.05)])
    return {
        JtdModel.FORCED_DEFAULT: ForcedDefaultJtdBlock(obligor_uids=uid, lgd=lgd),
        JtdModel.INDEPENDENT_BERNOULLI: IndependentBernoulliJtdBlock(
            obligor_uids=uid,
            lgd=lgd,
            pd=np.array([0.05]),
            uniform_uix=np.array([0], dtype=np.int32),
        ),
        JtdModel.BETA_BINOMIAL: BetaBinomialJtdBlock(
            obligor_uids=uid,
            lgd=lgd,
            uniform_uix=np.array([0], dtype=np.int32),
            beta_uix=np.array([6], dtype=np.int32),
        ),
        JtdModel.STRESS_REGIME_BERNOULLI: StressRegimeBernoulliJtdBlock(
            obligor_uids=uid,
            lgd=lgd,
            pd=np.array([0.05]),
            uniform_uix=np.array([0], dtype=np.int32),
            stress_regime_indicator_uix=np.array([5], dtype=np.int32),
            stress_pd_multiplier=np.array([3.0]),
        ),
        JtdModel.COPULA_SF_GAUSSIAN: SfGaussianCopulaJtdBlock(
            obligor_uids=uid,
            lgd=lgd,
            thresholds=thr,
            factor_loading=np.array([0.5]),
            systemic_uix=np.array([2], dtype=np.int32),
            idiosyncratic_uix=np.array([3], dtype=np.int32),
        ),
        JtdModel.COPULA_SF_STUDENTS_T: SfStudentsTCopulaJtdBlock(
            obligor_uids=uid,
            lgd=lgd,
            thresholds=thr,
            factor_loading=np.array([0.5]),
            systemic_uix=np.array([2], dtype=np.int32),
            idiosyncratic_uix=np.array([3], dtype=np.int32),
            chi_squared_uix=np.array([4], dtype=np.int32),
            sqrt_nu=np.array([np.sqrt(5.0)]),
        ),
        JtdModel.COPULA_MF_GAUSSIAN: MfGaussianCopulaJtdBlock(
            obligor_uids=uid,
            lgd=lgd,
            thresholds=thr,
            loading_indptr=np.array([0, 1], dtype=np.int64),
            loading_values=np.array([0.5]),
            systemic_uix=np.array([2], dtype=np.int32),
            idiosyncratic_uix=np.array([3], dtype=np.int32),
            idio_weight=np.array([np.sqrt(1.0 - 0.25)]),
        ),
        JtdModel.COPULA_MF_STUDENTS_T: MfStudentsTCopulaJtdBlock(
            obligor_uids=uid,
            lgd=lgd,
            thresholds=thr,
            loading_indptr=np.array([0, 1], dtype=np.int64),
            loading_values=np.array([0.5]),
            systemic_uix=np.array([2], dtype=np.int32),
            idiosyncratic_uix=np.array([3], dtype=np.int32),
            idio_weight=np.array([np.sqrt(1.0 - 0.25)]),
            chi_squared_uix=np.array([4], dtype=np.int32),
            sqrt_nu=np.array([np.sqrt(5.0)]),
        ),
        JtdModel.MARSHALL_OLKIN: MarshallOlkinJtdBlock(
            obligor_uids=uid,
            lgd=lgd,
            pd_idio=np.array([0.03]),
            pd_sys=np.array([0.02]),
            uniform_uix=np.array([0], dtype=np.int32),
            systemic_uix=np.array([1], dtype=np.int32),
        ),
    }


@pytest.mark.parametrize("model", list(JtdModel), ids=lambda m: m.name)
def test_every_jtd_kernel_executes_through_its_evaluator(model: JtdModel) -> None:
    """Drive each kernel through the production dispatch table, not by import.

    Compiles the kernel (a missing alias or a bad signature fails here) and
    checks the only model-agnostic post-condition: loss is finite and lies in
    {0, EAD*LGD} because the default indicator is binary.
    """
    s = 8
    rng = np.random.default_rng(0)
    latent = np.ascontiguousarray(
        np.vstack([
            rng.random(s),  # 0 uniform
            rng.random(s),  # 1 uniform (MO systemic)
            rng.standard_normal(s),  # 2 systemic Z
            rng.standard_normal(s),  # 3 idiosyncratic eps
            1.0 / np.sqrt(2.0 * rng.standard_gamma(2.5, s)),  # 4 1/sqrt(W)
            (rng.random(s) < 0.3).astype(np.float64),  # 5 stress indicator
            rng.beta(2.0, 38.0, s),  # 6 Beta common PD
        ])
    )
    factors = CreditFactorMatrix(latent_factors=latent)
    ead = np.ascontiguousarray(np.full((1, s), 1_000_000.0))
    out = np.empty((1, s), dtype=np.float64, order="C")

    evaluator_table()[int(model)](
        block=_blocks()[model], ead=ead, credit_factors=factors, out=out
    )

    assert np.isfinite(out).all()
    assert np.isin(out, (0.0, 1_000_000.0 * 0.6)).all()
