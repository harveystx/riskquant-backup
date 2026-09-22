# tests/credit/exposure/test_golden_master_runs.py
"""
P2-11 closure: the credit-exposure golden master must remain runnable.

`golden_master_credit_exposure.py` runs RC / CEM / SA-CCR through the public
request path and asserts hand-computed EADs plus contract invariants. Adding the
SA-CCR direction / maturity requirements (P2-11A/B) broke its frozen SA-CCR
fixture; this guards that the local adaptation keeps it running and that its
numbers are unchanged (the fixture is a long, >=1y book, so delta=+1 and MF=1
leave every value as before).

The canonical-book case depends on `riskquant_lab.fixtures`, which is a test-
support package outside this repository snapshot; it is skipped when absent.
"""

from __future__ import annotations

import pytest

from riskquant.solutions.credit_exposure import golden_master_credit_exposure as gm


def test_golden_master_frozen_case_runs():
    """Executes the frozen hand-computed book end to end (RC/CEM/SA-CCR)."""
    gm.test_frozen_case()


def test_golden_master_canonical_case_runs_if_fixture_available():
    try:
        gm.test_invariants_on_canonical_book()
    except ModuleNotFoundError as e:
        if "riskquant_lab" in str(e):
            pytest.skip("riskquant_lab.fixtures not in this snapshot")
        raise
