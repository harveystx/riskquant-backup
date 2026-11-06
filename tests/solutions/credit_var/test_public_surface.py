# tests/solutions/credit_var/test_public_surface.py
"""
WS_CREDIT_VAR_API S5: stability declaration, surface test and pins (C.28 to C.31).

S5.T1 (C.28): `sorted(api.__all__)` equals the 22 names of D106
clause 1. The expected list is written out here, not read from `api`, so a
change of the surface fails until the owner makes a version decision.

S5.T2 (C.29): `model_json_schema()` of `CreditVarRequest` and `CreditVarResult`
equals the golden files in `golden/`. The comparison is on text, not on parsed
dicts: in Python ``1 == 1.0 == True``, so a dict comparison does not see a
change between an int, a float and a bool bound. The text form is
``json.dumps(schema, indent=2, sort_keys=True) + "\\n"``. CRLF in the golden
file becomes LF before the comparison, so a git checkout with autocrlf does not
fail the test. A failure needs a new golden file and a version decision under
D106; do not regenerate the file only to make the test pass.

S5.T3 (C.30, O-11): `tomllib` reads `pyproject.toml`. The version is "0.2.1".
The numpy, scipy and pydantic entries are exact strings. The numba entry is
``numba>=0.59,<0.N`` with an integer N, and the installed numba satisfies it.
The installed version comes from the distribution metadata, so the test does
not import numba.

S5.T4 (C.31, O-4, D001): `src/riskquant/credit/STABLE_API.md` stays an internal
contract. Every non-blank line of its "Listed names" section is one entry
``- `riskquant.credit.<module>.<name>```. The set of (module, name) pairs
equals the union of the literal `__all__` lists of the `riskquant.credit`
modules, read with `ast` (no import, so no kernel compile). Each stated module
path is a file. The first non-blank line starts with "Status: internal". No
line matches ``(?i)\\b(public|stable|versioned)\\b``.
"""

from __future__ import annotations

import ast
import importlib.metadata
import json
import re
import tomllib
from pathlib import Path
from typing import Any, Final

import pytest
from pydantic import BaseModel

from riskquant.solutions.credit_var import api
from riskquant.solutions.credit_var.api import CreditVarRequest, CreditVarResult

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_GOLDEN_DIR: Final[Path] = Path(__file__).resolve().parent / "golden"
_PYPROJECT: Final[Path] = _REPO_ROOT / "pyproject.toml"
_CREDIT_DIR: Final[Path] = _REPO_ROOT / "src" / "riskquant" / "credit"
_STABLE_API: Final[Path] = _CREDIT_DIR / "STABLE_API.md"

# D106 clause 1, written out in clause order.
_SURFACE: Final[tuple[str, ...]] = (
    "AlphaRisk",
    "CreditVarEngineError",
    "CreditVarRequest",
    "CreditVarResult",
    "GroupRow",
    "LoadingRow",
    "LossHistogram",
    "MODEL_CATALOG",
    "ModelEntry",
    "ModelTail",
    "ObligorContribution",
    "ObligorRow",
    "ParamSpec",
    "PercentileLoss",
    "RiskSummary",
    "RunProvenance",
    "StressTestObligor",
    "StressTestResult",
    "model_entry",
    "run_credit_var",
    "simulate_portfolio_losses",
    "supported_model_keys",
)

_GOLDEN: Final[tuple[tuple[type[BaseModel], str], ...]] = (
    (CreditVarRequest, "credit_var_request.schema.json"),
    (CreditVarResult, "credit_var_result.schema.json"),
)

# C.30: exact entries. The numba entry has a free integer N (O-11).
_EXACT_PINS: Final[dict[str, str]] = {
    "numpy": "numpy>=1.26,<3",
    "scipy": "scipy>=1.15,<2",
    "pydantic": "pydantic>=2.0,<3",
}
_NUMBA_PIN: Final[re.Pattern[str]] = re.compile(r"numba>=0\.59,<0\.(\d+)")
_NUMBA_FLOOR_MINOR: Final[int] = 59
_REQUIREMENT_NAME: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_RELEASE: Final[re.Pattern[str]] = re.compile(r"(\d+)\.(\d+)")

_STATUS_PREFIX: Final[str] = "Status: internal"
_FORBIDDEN_CLAIM: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(public|stable|versioned)\b"
)
_LISTED_HEADING: Final[str] = "## Listed names"
_LISTED_ENTRY: Final[re.Pattern[str]] = re.compile(
    r"- `riskquant\.credit\.((?:\w+\.)*\w+)\.(\w+)`"
)


# ---------------------------------------------------------------------------
# S5.T1: surface names
# ---------------------------------------------------------------------------


def test_s5_t1_api_all_equals_the_declared_surface() -> None:
    assert len(_SURFACE) == 22
    assert len(set(api.__all__)) == len(api.__all__), "duplicate name in __all__"
    assert sorted(api.__all__) == sorted(_SURFACE)
    missing = [name for name in api.__all__ if not hasattr(api, name)]
    assert missing == [], f"__all__ names without a binding in api: {missing}"


# ---------------------------------------------------------------------------
# S5.T2: JSON schema golden files
# ---------------------------------------------------------------------------


def _schema_text(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize(
    ("model", "golden_name"),
    _GOLDEN,
    ids=[golden_name for _, golden_name in _GOLDEN],
)
def test_s5_t2_json_schema_equals_golden_file(
    model: type[BaseModel], golden_name: str
) -> None:
    golden = (_GOLDEN_DIR / golden_name).read_text(encoding="utf-8")
    golden = golden.replace("\r\n", "\n")
    assert _schema_text(model) == golden, (
        f"{model.__name__}.model_json_schema() differs from golden/{golden_name}. "
        "A schema change needs a new golden file and a version decision under "
        "D106 (WS_CREDIT_VAR_API C.29)."
    )


# ---------------------------------------------------------------------------
# S5.T3: version and dependency pins
# ---------------------------------------------------------------------------


def _dependencies_by_name(project: dict[str, Any]) -> dict[str, str]:
    by_name: dict[str, str] = {}
    for entry in project["dependencies"]:
        assert isinstance(entry, str)
        match = _REQUIREMENT_NAME.match(entry)
        assert match is not None, f"unparsable dependency entry {entry!r}"
        name = match.group(0).lower()
        assert name not in by_name, f"dependency {name!r} is declared twice"
        by_name[name] = entry
    return by_name


def test_s5_t3_pyproject_version_and_pins() -> None:
    with _PYPROJECT.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    assert project["version"] == "0.2.1"

    dependencies = _dependencies_by_name(project)
    for name, expected in _EXACT_PINS.items():
        assert dependencies.get(name) == expected

    numba_pin = _NUMBA_PIN.fullmatch(dependencies.get("numba", ""))
    assert numba_pin is not None, (
        f"numba entry {dependencies.get('numba')!r} is not 'numba>=0.59,<0.N'"
    )
    upper_minor = int(numba_pin.group(1))
    assert upper_minor > _NUMBA_FLOOR_MINOR

    installed = importlib.metadata.version("numba")
    release = _RELEASE.match(installed)
    assert release is not None, f"unparsable numba version {installed!r}"
    major, minor = int(release.group(1)), int(release.group(2))
    assert major == 0
    assert _NUMBA_FLOOR_MINOR <= minor < upper_minor, (
        f"installed numba {installed} does not satisfy {dependencies['numba']!r}"
    )


# ---------------------------------------------------------------------------
# S5.T4: STABLE_API.md is an internal contract that matches __all__
# ---------------------------------------------------------------------------


def _literal_all(path: Path) -> list[str] | None:
    """Return the literal module-level `__all__` of `path`, or None.

    A non-literal or augmented `__all__` fails the test: the union must be
    exact, and only a literal list can be read without an import.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] | None = None
    for node in tree.body:
        if isinstance(node, ast.AugAssign) and (
            isinstance(node.target, ast.Name) and node.target.id == "__all__"
        ):
            pytest.fail(f"{path}: augmented __all__ is not readable without import")
        targets: list[ast.expr]
        value_node: ast.expr
        if isinstance(node, ast.Assign):
            targets, value_node = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value_node = [node.target], node.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        assert found is None, f"{path}: __all__ is assigned twice"
        value = ast.literal_eval(value_node)
        assert isinstance(value, (list, tuple))
        assert all(isinstance(name, str) for name in value)
        found = list(value)
    return found


def _credit_all_union() -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for path in sorted(_CREDIT_DIR.rglob("*.py")):
        names = _literal_all(path)
        if names is None:
            continue
        module = ".".join(path.relative_to(_CREDIT_DIR).with_suffix("").parts)
        for name in names:
            assert (module, name) not in pairs, f"{module}.{name} is exported twice"
            pairs.add((module, name))
    return pairs


def _listed_pairs(lines: list[str]) -> list[tuple[str, str]]:
    assert lines.count(_LISTED_HEADING) == 1
    start = lines.index(_LISTED_HEADING) + 1
    pairs: list[tuple[str, str]] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if not line.strip() or line.startswith("### "):
            continue
        match = _LISTED_ENTRY.fullmatch(line)
        assert match is not None, f"unparsable line in the listed section: {line!r}"
        pairs.append((match.group(1), match.group(2)))
    return pairs


def test_s5_t4_stable_api_md_is_internal_and_matches_credit_all() -> None:
    lines = _STABLE_API.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")

    first_content = next(line for line in lines if line.strip())
    assert first_content.startswith(_STATUS_PREFIX)

    claims = [
        (number, line)
        for number, line in enumerate(lines, start=1)
        if _FORBIDDEN_CLAIM.search(line)
    ]
    assert claims == [], f"public, stable or versioned claim: {claims}"

    listed = _listed_pairs(lines)
    assert len(set(listed)) == len(listed), "a name is listed twice"

    missing_paths = sorted(
        module
        for module, _ in listed
        if not (_CREDIT_DIR / Path(*module.split("."))).with_suffix(".py").is_file()
    )
    assert missing_paths == [], (
        f"stated module paths that do not exist: {missing_paths}"
    )

    union = _credit_all_union()
    assert union, "no __all__ found under riskquant.credit"
    assert set(listed) == union, (
        f"listed only: {sorted(set(listed) - union)}; "
        f"__all__ only: {sorted(union - set(listed))}"
    )
