# tests/solutions/factor_regression/test_digest.py
"""
Tests for S6 digest functions and label coverage (S6.T7 to S6.T10, S6.T14).

Test IDs
--------
S6.T7  - same inputs in-process and fresh subprocess -> three digests equal
S6.T8  - five single mutations each change exactly the expected digest(s)
S6.T9  - rebuild A.6 bytes with hashlib -> equals facade digests (oracle)
S6.T10 - every *_CODES tuple of codes.py (enumerated from dir(codes)) has a
          labels group; every code has exactly one name, the name equals its
          constant name, and the reverse map returns the code
S6.T14 - AST scan of every facade module: each import (any form, relative
          imports resolved, TYPE_CHECKING blocks and function bodies included)
          is in the C.21 allow list; hashing in digest.py only
R-017  - the digest functions reject a wrong type, dtype or shape and never
          cast or truncate

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
from collections.abc import Callable

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
# Every *_CODES tuple of codes.py has a labels group; every code has exactly
# one name, equal to its constant name; the reverse map returns the code.
# The tuples are enumerated from dir(codes), so a new tuple without a labels
# group fails here (R-018).

_CODES_SUFFIX = "_CODES"


def _codes_tuple_names() -> list[str]:
    return sorted(name for name in dir(codes) if name.endswith(_CODES_SUFFIX))


def test_t10_codes_module_declares_code_tuples() -> None:
    """Non-vacuity: the enumeration finds the code tuples of codes.py."""
    names = _codes_tuple_names()
    assert names, "codes.py declares no *_CODES tuple"
    for name in names:
        assert isinstance(getattr(codes, name), tuple), f"codes.{name} is not a tuple"


@pytest.mark.parametrize("tuple_name", _codes_tuple_names())
def test_t10_all_codes_labeled(tuple_name: str) -> None:
    group_name = tuple_name.removesuffix(_CODES_SUFFIX)
    assert group_name in labels.ALL_CODE_GROUPS, (
        f"codes.{tuple_name} has no labels group {group_name!r}"
    )
    codes_tuple = getattr(codes, tuple_name)
    registered, name_map, code_from_name = labels.ALL_CODE_GROUPS[group_name]
    assert registered is codes_tuple, (
        f"labels group {group_name!r} does not register codes.{tuple_name}"
    )
    values = [int(code) for code in codes_tuple]
    assert len(set(values)) == len(values), (
        f"codes.{tuple_name} holds a duplicate value: {values}"
    )
    assert set(name_map) == set(values), (
        f"group {group_name!r}: name keys {sorted(name_map)} != codes {values}"
    )
    assert len(code_from_name) == len(values), (
        f"group {group_name!r}: reverse map has {len(code_from_name)} names "
        f"for {len(values)} codes"
    )
    for code in codes_tuple:
        name = name_map[int(code)]
        assert isinstance(name, str) and name, (
            f"group {group_name!r}: code {code!r} maps to invalid name {name!r}"
        )
        # The label is the name of the constant itself: the attribute of
        # codes.py with that name is this exact code object.
        assert getattr(codes, name, None) is code, (
            f"group {group_name!r}: label {name!r} is not the constant name "
            f"of code {code!r}"
        )
        recovered = code_from_name[name]
        assert recovered is code, (
            f"group {group_name!r}: reverse map for {name!r} returned "
            f"{recovered!r}, expected {code!r}"
        )


def test_t10_no_labels_group_without_code_tuple() -> None:
    """Every labels group names a *_CODES tuple of codes.py."""
    expected = {name.removesuffix(_CODES_SUFFIX) for name in _codes_tuple_names()}
    assert set(labels.ALL_CODE_GROUPS) == expected


def test_t10_diagnostic_messages_cover_all_diagnostic_codes() -> None:
    for code in codes.DIAGNOSTIC_CODES:
        name = labels.DIAGNOSTIC_NAMES[int(code)]
        assert name in labels.DIAGNOSTIC_MESSAGES, (
            f"DIAGNOSTIC_MESSAGES missing entry for {name!r}"
        )


# ================================================================ S6.T14
# C.21 import scan of the facade (R-018).
#
# _c21_violations is a pure function of the source text and the module path.
# It resolves every import to absolute dotted names and checks each name
# against the C.21 allow list.  ast.walk visits every node, so imports under
# TYPE_CHECKING, inside functions, classes and try blocks are scanned.
# Known limits (OD-3): getattr, importlib.import_module, __import__ and quoted
# annotations are not imports and are not scanned.

_FACADE_PACKAGE = "riskquant.solutions.factor_regression"
_HASHING_MODULE = "riskquant.solutions.run_identity.hashing"
_DIGEST_MODULE = f"{_FACADE_PACKAGE}.digest"
_FACADE_ALLOWED = (
    "numpy",
    "riskquant.factor_regression",
    "riskquant.exceptions",
    _FACADE_PACKAGE,
)
_CLIMB = "relative import climbs above the top package"


def _module_identity(module_path: str) -> tuple[str, tuple[str, ...]]:
    """
    Return the dotted module name and the package parts for a source path.

    ``module_path`` is repository-relative and starts with ``src/``.
    """
    path = pathlib.PurePosixPath(module_path)
    assert path.parts[0] == "src" and path.suffix == ".py", module_path
    parts = path.with_suffix("").parts[1:]
    if parts[-1] == "__init__":
        package = parts[:-1]
        return ".".join(package), package
    return ".".join(parts), parts[:-1]


def _resolved_imports(source: str, module_path: str) -> list[tuple[int, str]]:
    """
    Return ``(lineno, name)`` for every name that the source imports.

    ``from x import y`` yields ``x.y``, because ``y`` can be a submodule (for
    example ``from riskquant.solutions.run_identity import hashing``).  A
    relative import is resolved against the package of ``module_path``.  A
    relative import above the top package yields the name ``_CLIMB``.
    """
    _, package = _module_identity(module_path)
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                if node.level - 1 >= len(package):
                    found.append((node.lineno, _CLIMB))
                    continue
                base_parts = package[: len(package) - (node.level - 1)]
                base = ".".join(
                    (*base_parts, node.module) if node.module else base_parts
                )
            else:
                base = node.module or ""
            for alias in node.names:
                name = base if alias.name == "*" else f"{base}.{alias.name}"
                found.append((node.lineno, name))
    return found


def _is_allowed(name: str, *, is_digest: bool) -> bool:
    if name.split(".")[0] in sys.stdlib_module_names:
        return True
    allowed = (*_FACADE_ALLOWED, _HASHING_MODULE) if is_digest else _FACADE_ALLOWED
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in allowed)


def _c21_violations(source: str, module_path: str) -> list[str]:
    """Return one message per import of the source that C.21 forbids."""
    module_name, _ = _module_identity(module_path)
    is_digest = module_name == _DIGEST_MODULE
    return [
        f"{module_path}:{lineno}: {name}"
        for lineno, name in _resolved_imports(source, module_path)
        if name == _CLIMB or not _is_allowed(name, is_digest=is_digest)
    ]


_CONTRACTS_PATH = "src/riskquant/solutions/factor_regression/contracts.py"
_LABELS_PATH = "src/riskquant/solutions/factor_regression/labels.py"
_DIGEST_PATH = "src/riskquant/solutions/factor_regression/digest.py"

_REJECTED_SOURCES: list[tuple[str, str, str]] = [
    ("pandas", "import pandas as pd\n", _CONTRACTS_PATH),
    (
        "hashing-from-in-labels",
        "from riskquant.solutions.run_identity import hashing\n",
        _LABELS_PATH,
    ),
    (
        "hashing-import-in-labels",
        "import riskquant.solutions.run_identity.hashing\n",
        _LABELS_PATH,
    ),
    (
        "hashing-member-in-labels",
        "from riskquant.solutions.run_identity.hashing import digest_named_arrays\n",
        _LABELS_PATH,
    ),
    (
        "hashing-relative-in-labels",
        "from ..run_identity import hashing\n",
        _LABELS_PATH,
    ),
    (
        "config-compiled",
        "from riskquant.config.proxy_model import compiled\n",
        _DIGEST_PATH,
    ),
    ("config-relative", "from ...config import proxy_model\n", _LABELS_PATH),
    # The target name is allowed; the only defect is the climb.
    ("relative-above-top-package", "from .... import numpy\n", _LABELS_PATH),
    (
        "import-inside-function",
        "def f() -> None:\n    import pandas\n",
        _CONTRACTS_PATH,
    ),
    (
        "import-under-type-checking",
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from riskquant.config.proxy_model.compiled import CompiledProxyModel\n",
        _CONTRACTS_PATH,
    ),
    ("scipy", "from scipy import stats\n", _CONTRACTS_PATH),
]

_ACCEPTED_SOURCES: list[tuple[str, str, str]] = [
    (
        "hashing-member-in-digest",
        "from riskquant.solutions.run_identity.hashing import digest_named_arrays\n",
        _DIGEST_PATH,
    ),
    (
        "hashing-from-in-digest",
        "from riskquant.solutions.run_identity import hashing\n",
        _DIGEST_PATH,
    ),
    (
        "hashing-import-in-digest",
        "import riskquant.solutions.run_identity.hashing\n",
        _DIGEST_PATH,
    ),
    (
        "numpy",
        "import numpy as np\nfrom numpy.typing import NDArray\n",
        _CONTRACTS_PATH,
    ),
    ("core-codes", "from riskquant.factor_regression import codes\n", _LABELS_PATH),
    ("own-package-relative", "from .errors import ModelSpecError\n", _CONTRACTS_PATH),
    (
        "stdlib",
        "from __future__ import annotations\nimport unicodedata\n",
        _CONTRACTS_PATH,
    ),
]


@pytest.mark.parametrize(
    ("source", "module_path"),
    [(src, path) for _, src, path in _REJECTED_SOURCES],
    ids=[case_id for case_id, _, _ in _REJECTED_SOURCES],
)
def test_t14_helper_rejects_forbidden_import(source: str, module_path: str) -> None:
    assert _c21_violations(source, module_path), (
        f"forbidden import accepted in {module_path}: {source!r}"
    )


def test_t14_helper_names_the_climb() -> None:
    violations = _c21_violations("from .... import numpy\n", _LABELS_PATH)
    assert violations == [f"{_LABELS_PATH}:1: {_CLIMB}"]


@pytest.mark.parametrize(
    ("source", "module_path"),
    [(src, path) for _, src, path in _ACCEPTED_SOURCES],
    ids=[case_id for case_id, _, _ in _ACCEPTED_SOURCES],
)
def test_t14_helper_accepts_allowed_import(source: str, module_path: str) -> None:
    assert _c21_violations(source, module_path) == []


def _pkg_dir() -> pathlib.Path:
    spec = importlib.util.find_spec(_FACADE_PACKAGE)
    assert spec is not None
    assert spec.origin is not None
    return pathlib.Path(spec.origin).parent


def _facade_modules() -> list[tuple[str, str]]:
    """Return ``(module_path, source)`` for every .py file of the facade."""
    pkg = _pkg_dir()
    src_root = pkg.parents[2]
    return [
        (
            f"src/{py_file.relative_to(src_root).as_posix()}",
            py_file.read_text(encoding="utf-8"),
        )
        for py_file in sorted(pkg.rglob("*.py"))
    ]


def test_t14_facade_imports_obey_c21() -> None:
    modules = _facade_modules()
    paths = {path for path, _ in modules}
    assert _DIGEST_PATH in paths and _CONTRACTS_PATH in paths, sorted(paths)
    violations = [
        message for path, source in modules for message in _c21_violations(source, path)
    ]
    assert violations == [], "C.21 import violations:\n" + "\n".join(violations)


def test_t14_scan_sees_the_hashing_import_of_digest() -> None:
    """Non-vacuity: the resolver finds the one allowed hashing import."""
    hashing_users = sorted(
        path
        for path, source in _facade_modules()
        for _, name in _resolved_imports(source, path)
        if name == _HASHING_MODULE or name.startswith(f"{_HASHING_MODULE}.")
    )
    assert hashing_users == [_DIGEST_PATH]


# ================================================================ R-017
# The digest functions reject invalid input and never cast or truncate it.


def _spec_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = dict(
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
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    "overrides",
    [
        {"regressor_names": ["Z", "S"]},
        {"regressor_units": ["EUR", "USD"]},
        {"regressor_transforms": ["LOG_RETURN", "LOG_RETURN"]},
        {"regressor_names": []},
    ],
    ids=["two-names-one-unit", "one-name-two-units", "two-transforms", "no-names"],
)
def test_r017_spec_digest_unequal_regressor_lengths_raise(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        spec_digest(**_spec_kwargs(**overrides))  # type: ignore[arg-type]


_SWAPPED_INT32 = np.dtype(np.int32).newbyteorder()
_SWAPPED_FLOAT64 = np.dtype(np.float64).newbyteorder()


def _call_row(start: object, end: object) -> str:
    return row_digest(start, end)  # type: ignore[arg-type]


def _call_restriction(restriction: object, rhs: object) -> str:
    return restriction_digest(restriction, rhs)  # type: ignore[arg-type]


def _call_sample(
    start: object = None, end: object = None, y: object = None, x: object = None
) -> str:
    return sample_digest(
        row_start_date_ord=_row_start() if start is None else start,  # type: ignore[arg-type]
        row_end_date_ord=_row_end() if end is None else end,  # type: ignore[arg-type]
        response_name="P",
        intercept=True,
        regressor_names=["Z"],
        y=_y() if y is None else y,  # type: ignore[arg-type]
        x=_x() if x is None else x,  # type: ignore[arg-type]
    )


_TYPE_ERROR_CASES: list[tuple[str, Callable[[], str]]] = [
    ("row-int64-start", lambda: _call_row(_row_start().astype(np.int64), _row_end())),
    ("row-int64-end", lambda: _call_row(_row_start(), _row_end().astype(np.int64))),
    (
        "row-swapped-byte-order",
        lambda: _call_row(_row_start().astype(_SWAPPED_INT32), _row_end()),
    ),
    ("row-list", lambda: _call_row([10, 11, 12], _row_end())),
    (
        "row-masked-array",
        lambda: _call_row(np.ma.array([10, 11, 12], dtype=np.int32), _row_end()),
    ),
    ("sample-int64-start", lambda: _call_sample(start=_row_start().astype(np.int64))),
    ("sample-float32-y", lambda: _call_sample(y=_y().astype(np.float32))),
    ("sample-int64-y", lambda: _call_sample(y=np.array([1, 2, 3], dtype=np.int64))),
    ("sample-swapped-y", lambda: _call_sample(y=_y().astype(_SWAPPED_FLOAT64))),
    ("sample-float32-x", lambda: _call_sample(x=_x().astype(np.float32))),
    ("sample-list-x", lambda: _call_sample(x=_x().tolist())),
    (
        "restriction-float32",
        lambda: _call_restriction(
            np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float64)
        ),
    ),
    (
        "rhs-int64",
        lambda: _call_restriction(
            np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.int64)
        ),
    ),
    (
        "restriction-swapped",
        lambda: _call_restriction(
            np.eye(2, dtype=np.float64).astype(_SWAPPED_FLOAT64),
            np.zeros(2, dtype=np.float64),
        ),
    ),
]


@pytest.mark.parametrize(
    "call",
    [call for _, call in _TYPE_ERROR_CASES],
    ids=[case_id for case_id, _ in _TYPE_ERROR_CASES],
)
def test_r017_wrong_dtype_raises_type_error(call: Callable[[], str]) -> None:
    with pytest.raises(TypeError):
        call()


_VALUE_ERROR_CASES: list[tuple[str, Callable[[], str]]] = [
    (
        "row-unequal-length",
        lambda: _call_row(_row_start(), np.array([15, 16], dtype=np.int32)),
    ),
    (
        "row-2d",
        lambda: _call_row(_row_start().reshape(3, 1), _row_end().reshape(3, 1)),
    ),
    (
        "sample-unequal-rows",
        lambda: _call_sample(end=np.array([15, 16], dtype=np.int32)),
    ),
    ("sample-y-short", lambda: _call_sample(y=_y()[:2].copy())),
    ("sample-y-2d", lambda: _call_sample(y=_y().reshape(3, 1))),
    ("sample-x-short", lambda: _call_sample(x=_x()[:2].copy())),
    ("sample-x-1d", lambda: _call_sample(x=_x()[:, 1].copy())),
    ("sample-x-3d", lambda: _call_sample(x=_x().reshape(3, 2, 1))),
]


@pytest.mark.parametrize(
    "call",
    [call for _, call in _VALUE_ERROR_CASES],
    ids=[case_id for case_id, _ in _VALUE_ERROR_CASES],
)
def test_r017_wrong_shape_raises_value_error(call: Callable[[], str]) -> None:
    with pytest.raises(ValueError):
        call()


def test_r017_restriction_digest_leaves_shape_to_the_core() -> None:
    """Guards C.18: a wrong shape with the right dtype digests; wald_core rejects it."""
    digest = restriction_digest(
        np.zeros(3, dtype=np.float64), np.zeros((2, 2), dtype=np.float64)
    )
    assert len(digest) == 64


def test_r017_fortran_order_x_digests_as_c_order() -> None:
    """Guards A.6: the layout is canonicalised, not rejected."""
    x_f = np.asfortranarray(_x())
    assert not x_f.flags["C_CONTIGUOUS"]
    assert _call_sample(x=x_f) == _base_sample_digest()
