# tests/utils/test_printer_polars.py
"""
Contract of ``riskquant.utils.printer`` on Polars (WS_CREDIT_POLARS C.13) and
of ``scripts/tools/demo_import_smoke.py`` (C.19).

- S2.T1: ``print_df`` prints a ``pl.DataFrame`` and a ``pl.Series`` in a
  process where ``import pandas`` fails.
- S2.T3: ``print_dict_of_dfs`` prints a wide Polars frame transposed with a
  header column (``pl.DataFrame`` has no ``T``).
- Legacy branch: an object that is not Polars and has ``to_string`` prints
  through its own ``to_string`` with the v0.2.0 keywords; the printer imports
  no pandas. A pandas frame prints the v0.2.0 text.
- S2.T4: the smoke tool runs only top-level imports and names a failed demo.

S2.T4 is in this file because the S2 Files list names no test file for the
tool (owner choice pending; see the S2 report).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import polars as pl
import pytest

from riskquant.utils.printer import print_df, print_dict_of_dfs

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOKE_TOOL = _REPO_ROOT / "scripts" / "tools" / "demo_import_smoke.py"


def _run_without_pandas(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a child process where ``import pandas`` fails."""
    blocker = tmp_path / "pandas"
    blocker.mkdir()
    (blocker / "__init__.py").write_text('raise ImportError("pandas blocked")\n')
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), *sys.path])
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
    )


def _table_rows(text: str) -> list[list[str]]:
    """Cells of each body and header row of a Polars table text."""
    return [
        [cell.strip() for cell in line.strip().strip("│").split("┆")]
        for line in text.splitlines()
        if line.startswith("│")
    ]


# ---------------------------------------------------------------------------
# S2.T1
# ---------------------------------------------------------------------------

_T1_CHILD = """
import sys

import polars as pl

from riskquant.utils.printer import print_df

df = pl.DataFrame(
    {
        "n_int": [1, 2_000_000, None],
        "x_float": [1.5, None, -3_000_000.25],
        "s_str": ["aa", None, "cc"],
    }
)
print("== FORMATTED")
print_df(df, title="t1", float_format="_.0f", add_space_before=False)
print("== NATIVE")
print_df(df, float_format=None, add_space_before=False)
print("== MAXROWS")
print_df(df, max_rows=2, add_space_before=False)
print("== SERIES")
print_df(pl.Series("only", [10, 20]), add_space_before=False)
print("== END")
assert "pandas" not in sys.modules or sys.modules["pandas"] is None
print("OK no pandas")
"""


def _block(stdout: str, start: str, end: str) -> str:
    return stdout.split(f"== {start}\n", 1)[1].split(f"== {end}\n", 1)[0]


def test_s2_t1_print_df_polars_without_pandas(tmp_path: Path) -> None:
    proc = _run_without_pandas(tmp_path, _T1_CHILD)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = proc.stdout
    assert "OK no pandas" in out

    formatted = _table_rows(_block(out, "FORMATTED", "NATIVE"))
    assert formatted[0] == ["n_int", "x_float", "s_str"]
    assert formatted[1:] == [
        ["1", "2", "aa"],
        ["2_000_000", "null", "null"],
        ["null", "-3_000_000", "cc"],
    ]
    assert "shape=(3, 3)" in _block(out, "FORMATTED", "NATIVE")

    native = _table_rows(_block(out, "NATIVE", "MAXROWS"))
    assert native[0] == ["n_int", "x_float", "s_str"]
    assert native[1] == ["1", "1.5", "aa"]
    assert len(native) == 4

    capped = _table_rows(_block(out, "MAXROWS", "SERIES"))
    assert capped[0] == ["n_int", "x_float", "s_str"]
    assert len(capped) == 4  # header, first row, ellipsis row, last row
    assert capped[2] == ["…", "…", "…"]
    assert ["2_000_000", "null", "null"] not in capped

    series = _table_rows(_block(out, "SERIES", "END"))
    assert series == [["only"], ["10"], ["20"]]


def test_s2_t1_unnamed_series_prints_as_value_column(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_df(pl.Series([3, 4]), add_space_before=False, add_space_after=False)
    assert _table_rows(capsys.readouterr().out) == [["value"], ["3"], ["4"]]


# ---------------------------------------------------------------------------
# S2.T3
# ---------------------------------------------------------------------------


def test_s2_t3_print_dict_of_dfs_transposes_wide_polars_frame(
    capsys: pytest.CaptureFixture[str],
) -> None:
    wide = pl.DataFrame({f"c{i}": [i, 100 + i] for i in range(25)})
    narrow = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
    print_dict_of_dfs(
        {"wide": wide, "narrow": narrow},
        add_space_before=False,
        add_space_after=False,
    )
    out = capsys.readouterr().out
    wide_text, narrow_text = out.split("narrow (shape=", 1)

    assert "wide (transposed) (shape=(25, 3))" in wide_text
    wide_rows = _table_rows(wide_text)
    assert wide_rows[0] == ["column", "column_0", "column_1"]
    assert wide_rows[1:] == [[f"c{i}", str(i), str(100 + i)] for i in range(25)]

    assert "transposed" not in narrow_text
    assert _table_rows(narrow_text) == [["a", "b", "c"], ["1", "2", "3"]]


# ---------------------------------------------------------------------------
# Legacy branch (C.13: non-Polars object with to_string)
# ---------------------------------------------------------------------------

_LEGACY_CHILD = """
import sys

from riskquant.utils.printer import print_df


class Table:
    shape = (2, 1)
    calls = []

    def to_string(self, **kwargs):
        Table.calls.append(kwargs)
        return "LEGACY-TABLE-TEXT"


print_df(Table(), title="legacy", float_format=None, max_rows=7)
assert Table.calls == [{"index": False, "max_rows": 7, "max_cols": None}], Table.calls
print_df(Table(), float_format="_.0f", index=True)
kwargs = Table.calls[1]
assert sorted(kwargs) == ["float_format", "index", "max_cols", "max_rows"], kwargs
assert kwargs["index"] is True
assert kwargs["float_format"](1234567.4) == "1_234_567"
try:
    print_df(object())
except TypeError as exc:
    print("OK TypeError", exc)
assert "pandas" not in sys.modules or sys.modules["pandas"] is None
print("OK legacy without pandas")
"""


def test_legacy_branch_prints_through_own_to_string_without_pandas(
    tmp_path: Path,
) -> None:
    proc = _run_without_pandas(tmp_path, _LEGACY_CHILD)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.count("LEGACY-TABLE-TEXT") == 2
    assert "legacy (shape=(2, 1))" in proc.stdout
    assert "OK TypeError print_df expects a polars.DataFrame" in proc.stdout
    assert "OK legacy without pandas" in proc.stdout


def test_legacy_branch_keeps_the_v0_2_0_pandas_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [1, 2_000_000], "b": [1.5, -3.25], "s": ["x", "y"]})

    print_df(frame, add_space_before=False, add_space_after=False)

    # Independent oracle: the v0.2.0 steps written out with pandas.
    cast = frame.copy()
    cast[["a", "b"]] = cast[["a", "b"]].astype(float)
    expected = cast.to_string(index=False, float_format=lambda x: format(x, "_.0f"))
    assert capsys.readouterr().out == expected + "\n"


# ---------------------------------------------------------------------------
# S2.T4 (smoke tool)
# ---------------------------------------------------------------------------


def _run_smoke(*paths: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SMOKE_TOOL), *map(str, paths)],
        capture_output=True,
        text=True,
    )


def test_s2_t4_smoke_runs_imports_only(tmp_path: Path) -> None:
    marker = tmp_path / "body_ran.txt"
    demo = tmp_path / "demo_ok.py"
    demo.write_text(
        "import json\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(json.dumps(1))\n",
        encoding="utf-8",
    )
    proc = _run_smoke(demo)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all demo imports resolve (1 files)" in proc.stdout
    assert not marker.exists()


def test_s2_t4_smoke_names_the_demo_with_a_failed_import(tmp_path: Path) -> None:
    good = tmp_path / "demo_good.py"
    good.write_text("import json\n", encoding="utf-8")
    bad = tmp_path / "demo_bad.py"
    bad.write_text(
        "import json\nimport riskquant_no_such_module_s2\n", encoding="utf-8"
    )
    proc = _run_smoke(good, bad)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "demo_bad.py: line 2: import riskquant_no_such_module_s2" in proc.stdout
    assert "ModuleNotFoundError" in proc.stdout
    assert "demo_good.py" not in proc.stdout


def test_s2_t4_smoke_refuses_a_path_that_is_not_a_py_file(tmp_path: Path) -> None:
    proc = _run_smoke(tmp_path / "missing.py")
    assert proc.returncode == 2
    assert "not a .py file" in proc.stdout
