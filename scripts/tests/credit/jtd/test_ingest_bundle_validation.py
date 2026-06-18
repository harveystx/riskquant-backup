# scripts/tests/credit/jtd/test_ingest_bundle_validation.py
"""
Standalone structural-validation tests for the JTD ingest bundle (step a).

Numba is not exercised here — this is pure pandas/numpy contract validation, so
it runs and asserts directly. Each predicate is checked on a passing case and at
least one failing case. Run: `python test_ingest_bundle_validation.py`.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from riskquant.credit.jtd.ingest.bundle import JtdIngestBundle
from riskquant.credit.jtd.ingest.contracts import (
    JTD_MODEL_CONTRACTS,
)
from riskquant.credit.jtd.ingest.validate import validate_ingest_bundle
from riskquant.credit.jtd.models import JtdModel
from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit
from riskquant.utils.printer import print_calling_file, print_header


def _master(uids: list[int]) -> CompiledObligorCredit:
    arr = np.array(sorted(uids), dtype=np.int64)
    n = arr.shape[0]
    kw: dict[str, object] = dict(uid=arr, pd=np.full(n, 0.05), lgd=np.full(n, 0.6))
    for f in dataclasses.fields(CompiledObligorCredit):
        if f.name not in kw:
            kw[f.name] = np.zeros(
                n, dtype=np.int8 if f.name == "rating_id" else np.int64
            )
    return CompiledObligorCredit(**kw)  # type: ignore[arg-type]


def _expect_ok(
    label: str, bundle: JtdIngestBundle, master: CompiledObligorCredit
) -> None:
    validate_ingest_bundle(bundle, master)
    print(f"  ok (passes): {label}")


def _expect_fail(
    label: str, bundle: JtdIngestBundle, master: CompiledObligorCredit
) -> None:
    try:
        validate_ingest_bundle(bundle, master)
    except ValueError as e:
        print(f"  ok (rejects): {label} -> {str(e)[:74]}")
        return
    raise AssertionError(f"expected failure but passed: {label}")


def main() -> None:

    print_calling_file()

    print_header(title="STRUCTURAL VALIDATION TESTS FOR JTD INGEST BUNDLE")

    master = _master([1, 2, 3, 4, 5, 6])

    # ---- contract/bundle coverage ----
    assert set(JTD_MODEL_CONTRACTS) == set(JtdModel)
    assert set(JtdIngestBundle().frames_by_model()) == set()
    print("coverage: every JtdModel has a contract and a bundle slot")

    # ---- a clean mixed bundle passes ----
    beta = pd.DataFrame(
        {
            "uid": [1, 2],
            "uniform_factor": ["U_01", "U_02"],
            "beta_factor": ["GLOBAL_BETA_PD"] * 2,
        }
    )
    sfg = pd.DataFrame(
        {
            "uid": [3],
            "factor_loading": [0.4],
            "systemic_factor": ["GLOBAL_CREDIT"],
            "idiosyncratic_factor": ["IDIO_03"],
        }
    )
    mfg = pd.DataFrame({"uid": [4, 5], "idiosyncratic_factor": ["IDIO_04", "IDIO_05"]})
    loadings = pd.DataFrame(
        {
            "obligor_uid": [4, 4, 5],
            "factor_name": ["GLOBAL_CREDIT", "SECTOR_ENERGY", "GLOBAL_CREDIT"],
            "factor_loading": [0.3, 0.7, 0.5],
        }
    )
    print("\nstructural pass:")
    _expect_ok(
        "mixed bundle (beta + sf_gaussian + mf_gaussian + loadings)",
        JtdIngestBundle(
            beta_binomial_df=beta,
            sf_gaussian_copula_df=sfg,
            mf_gaussian_copula_df=mfg,
            mf_loadings_df=loadings,
        ),
        master,
    )

    print("\nstructural rejects:")
    # empty bundle
    _expect_fail("empty bundle", JtdIngestBundle(), master)
    # missing required column
    _expect_fail(
        "beta_binomial missing beta_factor",
        JtdIngestBundle(beta_binomial_df=beta.drop(columns=["beta_factor"])),
        master,
    )
    # null in required column
    nb = beta.copy()
    nb.loc[0, "uniform_factor"] = None
    _expect_fail("null in uniform_factor", JtdIngestBundle(beta_binomial_df=nb), master)
    # bad dtype: uid as string
    bd = beta.copy()
    bd["uid"] = bd["uid"].astype(str)
    _expect_fail("uid not integer dtype", JtdIngestBundle(beta_binomial_df=bd), master)
    # bad dtype: float column non-numeric
    st = pd.DataFrame(
        {
            "uid": [1],
            "uniform_factor": ["U_01"],
            "stress_regime_factor": ["STRESS_GLOBAL"],
            "stress_pd_multiplier": ["high"],
        }
    )
    _expect_fail(
        "stress_pd_multiplier non-numeric",
        JtdIngestBundle(stress_regime_bernoulli_df=st),
        master,
    )
    # duplicate uid within frame
    dup = pd.DataFrame(
        {
            "uid": [1, 1],
            "uniform_factor": ["U_01", "U_02"],
            "beta_factor": ["GLOBAL_BETA_PD"] * 2,
        }
    )
    _expect_fail(
        "duplicate uid within frame", JtdIngestBundle(beta_binomial_df=dup), master
    )
    # uid not in master
    _expect_fail(
        "uid absent from master",
        JtdIngestBundle(
            beta_binomial_df=pd.DataFrame(
                {
                    "uid": [999],
                    "uniform_factor": ["U_01"],
                    "beta_factor": ["GLOBAL_BETA_PD"],
                }
            )
        ),
        master,
    )
    # same obligor in two frames
    _expect_fail(
        "obligor in two model frames",
        JtdIngestBundle(
            beta_binomial_df=pd.DataFrame(
                {
                    "uid": [1],
                    "uniform_factor": ["U_01"],
                    "beta_factor": ["GLOBAL_BETA_PD"],
                }
            ),
            independent_bernoulli_df=pd.DataFrame(
                {"uid": [1], "uniform_factor": ["U_02"]}
            ),
        ),
        master,
    )
    # MF frame present but loadings missing
    _expect_fail(
        "mf frame without loadings", JtdIngestBundle(mf_gaussian_copula_df=mfg), master
    )
    # loadings present but no MF obligors
    _expect_fail(
        "loadings without mf obligors",
        JtdIngestBundle(beta_binomial_df=beta, mf_loadings_df=loadings),
        master,
    )
    # NaN loading
    bad_load = loadings.copy()
    bad_load.loc[0, "factor_loading"] = np.nan
    _expect_fail(
        "NaN in mf loading",
        JtdIngestBundle(mf_gaussian_copula_df=mfg, mf_loadings_df=bad_load),
        master,
    )
    # stray loading uid (routability / silent-drop guard)
    stray_load = pd.DataFrame(
        {
            "obligor_uid": [4, 99],
            "factor_name": ["GLOBAL_CREDIT", "GLOBAL_CREDIT"],
            "factor_loading": [0.3, 0.4],
        }
    )
    _expect_fail(
        "stray loading uid",
        JtdIngestBundle(mf_gaussian_copula_df=mfg, mf_loadings_df=stray_load),
        master,
    )

    print_header(title="ALL STRUCTURAL VALIDATION TESTS PASS")


if __name__ == "__main__":
    main()
