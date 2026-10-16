# tests/solutions/factor_regression/test_digest.py
"""
Tests for S6 digest functions and label coverage (S6.T7 to S6.T10, S6.T14).

Test IDs
--------
S6.T7  - same inputs in-process and fresh subprocess -> three digests equal
S6.T8  - five single mutations each change exactly the expected digest(s)
S6.T9  - rebuild A.6 bytes with hashlib -> equals facade digests (oracle)
S6.T10 - every code in codes.py has exactly one name in labels.py;
          reverse map returns the original code
S6.T14 - AST scan: only digest.py imports hashing; no direct riskquant.config
          import in the facade

Notes
-----
Oracles share no code with the facade modules (A.7).  The oracle in S6.T9
re-implements ``digest_array`` and ``digest_named_arrays`` using only
``hashlib`` and ``numpy``.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from riskquant.factor_regression import codes
from riskquant.solutions.factor_regression import labels
from riskquant.solutions.factor_regression.contracts import ENGINE_VERSION
from riskquant.solutions.factor_regression.digest import (
    restriction_digest,
    row_digest,
    sample_digest,
    spec_digest,
)

# ------------------------------------------------------------------ fixtures


def _row_start() -> np.ndarray:
    return np.array([10, 11, 12], dtype=np.int32)


def _row_end() -> np.ndarray:
    return np.array([15, 16, 17], dtype=np.int32)


def _y() -> np.ndarray:
    return np.array([1.0, 2.0, 3.0], dtype=np.float64)


def _x() -> np.ndarray:
    return np.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]], dtype=np.float64)


def _base_row_digest() -> str:
    return row_digest(_row_start(), _row_end())


def _base_sample_digest() -> str:
    return sample_digest(
        row_start_date_ord=_row_start(),
        row_end_date_ord=_row_end(),
        response_name="P",
        intercept=True,
        regressor_names=["Z"],
        y=_y(),
        x=_x(),
    )


def _base_spec_digest() -> str:
    return spec_digest(
        engine_version=ENGINE_VERSION,
        response_name="P",
        response_unit="EUR",
        response_transform="LOG_RETURN",
        regressor_names=["Z"],
        regressor_units=["EUR"],
        regressor_transforms=["LOG_RETURN"],
        intercept=True,
        step=1,
        stride=1,
        start_ord=0,
        end_ord=20,
        covariance_kind=0,
        covariance_lag=-1,
        rank_rtol=1e-10,
        condition_threshold=None,
    )


# ================================================================ S6.T7
# cross-process digest stability

_SUBPROCESS_SCRIPT = """
import sys
sys.path[0:0] = {paths!r}
import json
import numpy as np
from riskquant.solutions.factor_regression.contracts import ENGINE_VERSION
from riskquant.solutions.factor_regression.digest import (
    row_digest, sample_digest, spec_digest
)

row_s = np.array([10, 11, 12], dtype=np.int32)
row_e = np.array([15, 16, 17], dtype=np.int32)
y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
x = np.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]], dtype=np.float64)

rd = row_digest(row_s, row_e)
sd = sample_digest(
    row_start_date_ord=row_s,
    row_end_date_ord=row_e,
    response_name="P",
    intercept=True,
    regressor_names=["Z"],
    y=y,
    x=x,
)
spd = spec_digest(
    engine_version=ENGINE_VERSION,
    response_name="P",
    response_unit="EUR",
    response_transform="LOG_RETURN",
    regressor_names=["Z"],
    regressor_units=["EUR"],
    regressor_transforms=["LOG_RETURN"],
    intercept=True,
    step=1,
    stride=1,
    start_ord=0,
    end_ord=20,
    covariance_kind=0,
    covariance_lag=-1,
    rank_rtol=1e-10,
    condition_threshold=None,
)
print(json.dumps({{"row": rd, "sample": sd, "spec": spd}}))
"""


def test_t7_cross_process_digests_equal() -> None:
    """
    The three digest functions produce the same output in a fresh subprocess
    that shares no module state with this process (C.16).
    """
    in_proc_row = _base_row_digest()
    in_proc_sample = _base_sample_digest()
    in_proc_spec = _base_spec_digest()

    script = _SUBPROCESS_SCRIPT.format(paths=sys.path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(result.stdout.strip())

    assert out["row"] == in_proc_row, "row_digest differs across processes"
    assert out["sample"] == in_proc_sample, "sample_digest differs across processes"
    assert out["spec"] == in_proc_spec, "spec_digest differs across processes"


# ================================================================ S6.T8
# five single mutations change exactly the expected digest(s)
#
# y change      -> sample only
# rename        -> sample and spec
# unit change   -> spec only
# window change -> spec, and row and sample when rows change
# lag change    -> spec only


def test_t8_y_change_affects_sample_only() -> None:
    row_s, row_e = _row_start(), _row_end()
    y_orig = _y()
    y_ulp = y_orig.copy()
    y_ulp[0] = np.nextafter(y_ulp[0], np.inf)  # one-ulp change

    def _sd(y_arr: np.ndarray) -> str:
        return sample_digest(
            row_start_date_ord=row_s,
            row_end_date_ord=row_e,
            response_name="P",
            intercept=True,
            regressor_names=["Z"],
            y=y_arr,
            x=_x(),
        )

    rd_orig = row_digest(row_s, row_e)
    sd_orig = _sd(y_orig)
    sd_ulp = _sd(y_ulp)
    spd_orig = _base_spec_digest()

    assert sd_ulp != sd_orig, "one-ulp y change must change sample_digest"
    assert row_digest(row_s, row_e) == rd_orig, "y change must not change row_digest"
    assert _base_spec_digest() == spd_orig, "y change must not change spec_digest"


def test_t8_rename_affects_sample_and_spec() -> None:
    row_s, row_e = _row_start(), _row_end()
    y = _y()
    x = _x()

    def _sd(name: str) -> str:
        return sample_digest(
            row_start_date_ord=row_s,
            row_end_date_ord=row_e,
            response_name=name,
            intercept=True,
            regressor_names=["Z"],
            y=y,
            x=x,
        )

    def _spd(name: str) -> str:
        return spec_digest(
            engine_version=ENGINE_VERSION,
            response_name=name,
            response_unit="EUR",
            response_transform="LOG_RETURN",
            regressor_names=["Z"],
            regressor_units=["EUR"],
            regressor_transforms=["LOG_RETURN"],
            intercept=True,
            step=1,
            stride=1,
            start_ord=0,
            end_ord=20,
            covariance_kind=0,
            covariance_lag=-1,
            rank_rtol=1e-10,
            condition_threshold=None,
        )

    rd = row_digest(row_s, row_e)

    assert _sd("P") != _sd("Q"), "rename must change sample_digest"
    assert _spd("P") != _spd("Q"), "rename must change spec_digest"
    assert row_digest(row_s, row_e) == rd, "rename must not change row_digest"


def test_t8_unit_change_affects_spec_only() -> None:
    row_s, row_e = _row_start(), _row_end()

    def _spd(unit: str) -> str:
        return spec_digest(
            engine_version=ENGINE_VERSION,
            response_name="P",
            response_unit=unit,
            response_transform="LOG_RETURN",
            regressor_names=["Z"],
            regressor_units=["EUR"],
            regressor_transforms=["LOG_RETURN"],
            intercept=True,
            step=1,
            stride=1,
            start_ord=0,
            end_ord=20,
            covariance_kind=0,
            covariance_lag=-1,
            rank_rtol=1e-10,
            condition_threshold=None,
        )

    rd_orig = row_digest(row_s, row_e)
    sd_orig = _base_sample_digest()

    assert _spd("EUR") != _spd("USD"), "unit change must change spec_digest"
    assert row_digest(row_s, row_e) == rd_orig, "unit change must not change row_digest"
    assert _base_sample_digest() == sd_orig, "unit change must not change sample_digest"


def test_t8_window_change_affects_spec_and_rows_when_rows_change() -> None:
    # Original rows cover [1,5] and [6,10].
    row_s1 = np.array([1, 6], dtype=np.int32)
    row_e1 = np.array([5, 10], dtype=np.int32)
    # Narrowed window excludes the second row.
    row_s2 = np.array([1], dtype=np.int32)
    row_e2 = np.array([5], dtype=np.int32)
    y1 = np.array([1.0, 2.0], dtype=np.float64)
    y2 = np.array([1.0], dtype=np.float64)
    x1 = np.array([[1.0, 0.5], [1.0, 1.5]], dtype=np.float64)
    x2 = np.array([[1.0, 0.5]], dtype=np.float64)

    def _sd(row_s: np.ndarray, row_e: np.ndarray, y: np.ndarray, x: np.ndarray) -> str:
        return sample_digest(
            row_start_date_ord=row_s,
            row_end_date_ord=row_e,
            response_name="P",
            intercept=True,
            regressor_names=["Z"],
            y=y,
            x=x,
        )

    def _spd(start: int, end: int) -> str:
        return spec_digest(
            engine_version=ENGINE_VERSION,
            response_name="P",
            response_unit="EUR",
            response_transform="LOG_RETURN",
            regressor_names=["Z"],
            regressor_units=["EUR"],
            regressor_transforms=["LOG_RETURN"],
            intercept=True,
            step=1,
            stride=1,
            start_ord=start,
            end_ord=end,
            covariance_kind=0,
            covariance_lag=-1,
            rank_rtol=1e-10,
            condition_threshold=None,
        )

    assert _spd(0, 20) != _spd(0, 8), "window change must change spec_digest"
    assert row_digest(row_s1, row_e1) != row_digest(row_s2, row_e2), (
        "row change must change row_digest"
    )
    assert _sd(row_s1, row_e1, y1, x1) != _sd(row_s2, row_e2, y2, x2), (
        "row change must change sample_digest"
    )


def test_t8_lag_change_affects_spec_only() -> None:
    row_s, row_e = _row_start(), _row_end()

    def _spd(lag: int) -> str:
        return spec_digest(
            engine_version=ENGINE_VERSION,
            response_name="P",
            response_unit="EUR",
            response_transform="LOG_RETURN",
            regressor_names=["Z"],
            regressor_units=["EUR"],
            regressor_transforms=["LOG_RETURN"],
            intercept=True,
            step=1,
            stride=1,
            start_ord=0,
            end_ord=20,
            covariance_kind=2,  # HAC_BARTLETT
            covariance_lag=lag,
            rank_rtol=1e-10,
            condition_threshold=None,
        )

    rd = row_digest(row_s, row_e)
    sd = _base_sample_digest()

    assert _spd(3) != _spd(5), "lag change must change spec_digest"
    assert row_digest(row_s, row_e) == rd, "lag change must not change row_digest"
    assert _base_sample_digest() == sd, "lag change must not change sample_digest"


# ================================================================ S6.T9
# Oracle: rebuild A.6 bytes with hashlib; equals facade digests
#
# The oracle shares NO code with the facade modules (A.7).
# It re-implements digest_array and digest_named_arrays from hashing.py
# using only hashlib and numpy.


def _oracle_digest_array(arr: np.ndarray) -> str:
    """Independent re-implementation of hashing.digest_array."""
    a = np.ascontiguousarray(arr)
    if a.dtype.byteorder not in ("=", "|"):
        a = a.astype(a.dtype.newbyteorder("="), copy=False)
    h = hashlib.sha256()
    h.update(str(arr.dtype.str).encode("utf-8"))
    h.update(b"|")
    h.update(str(arr.shape).encode("utf-8"))
    h.update(b"|")
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def _oracle_digest_named(**arrays: np.ndarray) -> str:
    """Independent re-implementation of hashing.digest_named_arrays."""
    h = hashlib.sha256()
    for name, arr in arrays.items():
        h.update(name.encode("utf-8"))
        h.update(b"=")
        h.update(_oracle_digest_array(arr).encode("utf-8"))
        h.update(b";")
    return h.hexdigest()


def _oracle_text_uint8(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()


def _oracle_row_digest(row_s: np.ndarray, row_e: np.ndarray) -> str:
    return _oracle_digest_named(
        row_start_date_ord=np.ascontiguousarray(row_s, dtype=np.int32),
        row_end_date_ord=np.ascontiguousarray(row_e, dtype=np.int32),
    )


def _oracle_sample_digest(
    row_s: np.ndarray,
    row_e: np.ndarray,
    response_name: str,
    intercept: bool,
    regressor_names: list[str],
    y: np.ndarray,
    x: np.ndarray,
) -> str:
    parts = [f"Y\x1f{response_name}"]
    if intercept:
        parts.append("C\x1f")
    for n in regressor_names:
        parts.append(f"X\x1f{n}")
    col_arr = _oracle_text_uint8("\x1e".join(parts))
    return _oracle_digest_named(
        row_start_date_ord=np.ascontiguousarray(row_s, dtype=np.int32),
        row_end_date_ord=np.ascontiguousarray(row_e, dtype=np.int32),
        column_names=col_arr,
        y=np.ascontiguousarray(y, dtype=np.float64),
        X=np.ascontiguousarray(x, dtype=np.float64),
    )


def _oracle_spec_digest(
    engine_version: str,
    response_name: str,
    response_unit: str,
    response_transform: str,
    regressor_names: list[str],
    regressor_units: list[str],
    regressor_transforms: list[str],
    intercept: bool,
    step: int,
    stride: int,
    start_ord: int,
    end_ord: int,
    covariance_kind: int,
    covariance_lag: int,
    rank_rtol: float,
    condition_threshold: float | None,
) -> str:
    lines = [f"Y\x1f{response_name}\x1f{response_unit}\x1f{response_transform}"]
    for n, u, t in zip(
        regressor_names, regressor_units, regressor_transforms, strict=False
    ):
        lines.append(f"X\x1f{n}\x1f{u}\x1f{t}")
    terms_arr = _oracle_text_uint8("\x1e".join(lines))
    engine_arr = _oracle_text_uint8(engine_version)
    intercept_arr = np.array([1 if intercept else 0], dtype=np.int8)
    schema_arr = np.array([1], dtype=np.int64)
    step_arr = np.array([step], dtype=np.int64)
    stride_arr = np.array([stride], dtype=np.int64)
    window_arr = np.array([start_ord, end_ord], dtype=np.int32)
    cov_kind_arr = np.array([covariance_kind], dtype=np.int8)
    cov_lag_arr = np.array([covariance_lag], dtype=np.int64)
    rank_rtol_arr = np.array([rank_rtol], dtype=np.float64)
    if condition_threshold is None:
        ct_arr = np.empty(0, dtype=np.float64)
    else:
        ct_arr = np.array([condition_threshold], dtype=np.float64)
    return _oracle_digest_named(
        schema_version=schema_arr,
        engine_version=engine_arr,
        terms=terms_arr,
        intercept=intercept_arr,
        step=step_arr,
        stride=stride_arr,
        window=window_arr,
        covariance_kind=cov_kind_arr,
        covariance_lag=cov_lag_arr,
        rank_rtol=rank_rtol_arr,
        condition_threshold=ct_arr,
    )


def test_t9_oracle_row_digest() -> None:
    row_s, row_e = _row_start(), _row_end()
    assert row_digest(row_s, row_e) == _oracle_row_digest(row_s, row_e)


def test_t9_oracle_sample_digest() -> None:
    row_s, row_e = _row_start(), _row_end()
    y, x = _y(), _x()
    facade = sample_digest(
        row_start_date_ord=row_s,
        row_end_date_ord=row_e,
        response_name="P",
        intercept=True,
        regressor_names=["Z"],
        y=y,
        x=x,
    )
    oracle = _oracle_sample_digest(row_s, row_e, "P", True, ["Z"], y, x)
    assert facade == oracle


def test_t9_oracle_spec_digest_no_threshold() -> None:
    facade = _base_spec_digest()
    oracle = _oracle_spec_digest(
        engine_version=ENGINE_VERSION,
        response_name="P",
        response_unit="EUR",
        response_transform="LOG_RETURN",
        regressor_names=["Z"],
        regressor_units=["EUR"],
        regressor_transforms=["LOG_RETURN"],
        intercept=True,
        step=1,
        stride=1,
        start_ord=0,
        end_ord=20,
        covariance_kind=0,
        covariance_lag=-1,
        rank_rtol=1e-10,
        condition_threshold=None,
    )
    assert facade == oracle


def test_t9_oracle_spec_digest_with_threshold() -> None:
    facade = spec_digest(
        engine_version=ENGINE_VERSION,
        response_name="P",
        response_unit="EUR",
        response_transform="LOG_RETURN",
        regressor_names=["Z"],
        regressor_units=["EUR"],
        regressor_transforms=["LOG_RETURN"],
        intercept=True,
        step=1,
        stride=1,
        start_ord=0,
        end_ord=20,
        covariance_kind=0,
        covariance_lag=-1,
        rank_rtol=1e-10,
        condition_threshold=100.0,
    )
    oracle = _oracle_spec_digest(
        engine_version=ENGINE_VERSION,
        response_name="P",
        response_unit="EUR",
        response_transform="LOG_RETURN",
        regressor_names=["Z"],
        regressor_units=["EUR"],
        regressor_transforms=["LOG_RETURN"],
        intercept=True,
        step=1,
        stride=1,
        start_ord=0,
        end_ord=20,
        covariance_kind=0,
        covariance_lag=-1,
        rank_rtol=1e-10,
        condition_threshold=100.0,
    )
    assert facade == oracle


def test_t9_oracle_restriction_digest() -> None:
    r = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    rhs = np.array([0.0, 1.0], dtype=np.float64)
    facade = restriction_digest(r, rhs)
    oracle = _oracle_digest_named(
        restriction=np.ascontiguousarray(r, dtype=np.float64),
        rhs=np.ascontiguousarray(rhs, dtype=np.float64),
    )
    assert facade == oracle


def test_t9_sample_no_intercept() -> None:
    row_s, row_e = _row_start(), _row_end()
    y = _y()
    x = np.array([[0.5], [1.5], [2.5]], dtype=np.float64)
    facade = sample_digest(
        row_start_date_ord=row_s,
        row_end_date_ord=row_e,
        response_name="P",
        intercept=False,
        regressor_names=["Z"],
        y=y,
        x=x,
    )
    oracle = _oracle_sample_digest(row_s, row_e, "P", False, ["Z"], y, x)
    assert facade == oracle


# ================================================================ S6.T10
# Every code in codes.py has exactly one name; reverse map returns the code


@pytest.mark.parametrize("group_name", list(labels.ALL_CODE_GROUPS))
def test_t10_all_codes_labeled(group_name: str) -> None:
    codes_tuple, name_map, code_from_name = labels.ALL_CODE_GROUPS[group_name]
    for code in codes_tuple:
        key = int(code)
        assert key in name_map, (
            f"group {group_name!r}: code {code!r} has no entry in {group_name}_NAMES"
        )
        name = name_map[key]
        assert isinstance(name, str) and name, (
            f"group {group_name!r}: code {code!r} maps to invalid name {name!r}"
        )
        recovered = code_from_name[name]
        assert int(recovered) == key, (
            f"group {group_name!r}: reverse map for {name!r} returned "
            f"{recovered!r}, expected {code!r}"
        )


def test_t10_no_unlabeled_code_group() -> None:
    """Every *_CODES tuple in codes.py must appear in ALL_CODE_GROUPS."""
    expected_groups = {
        "TRANSFORM",
        "COVARIANCE",
        "EXCLUSION",
        "DIAGNOSTIC",
        "INPUT_ERROR",
        "ALIGNMENT_ERROR",
        "PARAMETER_ERROR",
        "RANK_ERROR",
        "SAMPLE_SIZE_ERROR",
        "DEGENERATE_FIT_ERROR",
        "RESTRICTION_ERROR",
    }
    assert set(labels.ALL_CODE_GROUPS.keys()) == expected_groups


def test_t10_diagnostic_messages_cover_all_diagnostic_codes() -> None:
    for code in codes.DIAGNOSTIC_CODES:
        name = labels.DIAGNOSTIC_NAMES[int(code)]
        assert name in labels.DIAGNOSTIC_MESSAGES, (
            f"DIAGNOSTIC_MESSAGES missing entry for {name!r}"
        )


# ================================================================ S6.T14
# AST scan: only digest.py imports hashing; no direct riskquant.config import


def _pkg_dir() -> pathlib.Path:
    spec = importlib.util.find_spec("riskquant.solutions.factor_regression")
    assert spec is not None
    return pathlib.Path(spec.origin).parent  # type: ignore[arg-type]


def _all_imports(source: str) -> list[ast.stmt]:
    """Return all Import and ImportFrom nodes, including inside TYPE_CHECKING."""
    tree = ast.parse(source)
    result: list[ast.stmt] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            result.append(node)
    return result


def test_t14_only_digest_imports_hashing() -> None:
    pkg = _pkg_dir()
    for py_file in sorted(pkg.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        is_digest = py_file.name == "digest.py"
        source = py_file.read_text()
        for node in _all_imports(source):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "run_identity" in module and "hashing" in module:
                    assert is_digest, (
                        f"{py_file.name} imports hashing (only digest.py is allowed)"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "run_identity" in alias.name and "hashing" in alias.name:
                        assert is_digest, (
                            f"{py_file.name} imports hashing "
                            f"(only digest.py is allowed)"
                        )


def test_t14_no_direct_riskquant_config_import() -> None:
    pkg = _pkg_dir()
    for py_file in sorted(pkg.glob("*.py")):
        source = py_file.read_text()
        for node in _all_imports(source):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith("riskquant.config"), (
                    f"{py_file.name} directly imports riskquant.config"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("riskquant.config"), (
                        f"{py_file.name} directly imports riskquant.config"
                    )
