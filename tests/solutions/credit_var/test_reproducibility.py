# tests/solutions/credit_var/test_reproducibility.py
"""
WS_CREDIT_VAR_API S1: scenario-construction guard and D-C15 reproducibility.

S1.T13 (C.12): the recorded digest of the generator source equals the digest of
the source that is imported. A change of `mixed.py` fails here, and the owner
then decides: keep `SCENARIO_CONSTRUCTION_ID` and record the new digest (the
construction is unchanged), or set a new identifier (it changed).

S1.T14 (C.13): one request run twice in this process and once in a fresh
interpreter gives byte-equal `model_dump_json` output after removal of
`provenance.generated_at` (A_CONTRACT section 4.3). The comparison is
environment-relative (D-C15): the fresh interpreter imports the same source
tree and the same libraries, and the compared bytes include engine_version and
the library versions, so a different environment cannot pass silently.

The batch-size invariance of the result (S1.T15) lives in
`tests/credit/factors/test_batch_size_reproducibility.py`.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import riskquant
import riskquant.credit.factors.simulate.mixed as mixed_generator
from riskquant.solutions.credit_var import provenance
from riskquant.solutions.credit_var.api import run_credit_var

_EXCLUDE_GENERATED_AT: dict[str, Any] = {"provenance": {"generated_at"}}

_CHILD = """
import json
import sys
from riskquant.solutions.credit_var.api import run_credit_var
result = run_credit_var(json.loads(sys.stdin.buffer.read().decode("utf-8")))
sys.stdout.buffer.write(
    result.model_dump_json(exclude={"provenance": {"generated_at"}}).encode("utf-8")
)
"""


def _request() -> dict[str, Any]:
    """Student-t copula: systemic, idiosyncratic and chi factors in one stream,
    plus a forced default and a stress-test flag, so every result section is
    present."""
    return {
        "model": "sf_students_t",
        "obligors": [
            {"obligor_uid": 7, "ead": 1.0e6, "pd": 0.02, "lgd": 0.45},
            {"obligor_uid": 3, "ead": 2.5e6, "pd": 0.01, "lgd": 0.60},
            {"obligor_uid": 5, "ead": 4.0e5, "pd": 0.05, "lgd": 0.40},
            {
                "obligor_uid": 9,
                "ead": 8.0e5,
                "pd": 0.03,
                "lgd": 0.50,
                "force_default": True,
            },
            {
                "obligor_uid": 1,
                "ead": 6.0e5,
                "pd": 0.04,
                "lgd": 0.35,
                "in_stress_test": True,
            },
        ],
        "params": {"rho": 0.2, "nu": 6.0},
        "n_scenarios": 20_000,
        "alphas": [0.999, 0.95, 0.99],
        "seed": 20261030,
    }


def test_s1_t13_construction_source_digest_is_recorded() -> None:
    source = Path(mixed_generator.__file__)
    assert riskquant.__file__ is not None
    package_root = Path(riskquant.__file__).parent
    assert (package_root / provenance.SCENARIO_CONSTRUCTION_SOURCE).resolve() == (
        source.resolve()
    )
    digest = hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert digest == provenance.SCENARIO_CONSTRUCTION_SOURCE_SHA256, (
        "credit/factors/simulate/mixed.py changed. Decide: keep "
        f"SCENARIO_CONSTRUCTION_ID and record the new digest {digest}, or set a "
        "new identifier (WS_CREDIT_VAR_API C.12)."
    )
    assert provenance.SCENARIO_CONSTRUCTION_ID == "MixedCreditFactorGenerator/D-C15"


def test_s1_t14_same_request_gives_byte_equal_results() -> None:
    request = _request()
    first = run_credit_var(request).model_dump_json(exclude=_EXCLUDE_GENERATED_AT)
    second = run_credit_var(request).model_dump_json(exclude=_EXCLUDE_GENERATED_AT)
    assert first == second
    assert "generated_at" not in first

    assert riskquant.__file__ is not None
    source_root = str(Path(riskquant.__file__).parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        source_root,
        *filter(None, [env.get("PYTHONPATH")]),
    ])
    child = subprocess.run(
        [sys.executable, "-c", _CHILD],
        input=json.dumps(request).encode("utf-8"),
        capture_output=True,
        env=env,
        timeout=600,
        check=False,
    )
    assert child.returncode == 0, child.stderr.decode("utf-8", errors="replace")
    assert child.stdout.decode("utf-8") == first
