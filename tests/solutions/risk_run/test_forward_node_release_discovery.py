# tests/solutions/risk_run/test_forward_node_release_discovery.py
"""
`scripts/run_all_scripts_release.py` discovers the forward NODE release script.

Static: the real `main()` runs its own discovery, but the runner module's
`subprocess` and `time` names are replaced for this test only, so no script
executes and no engine symbol is touched. The test records the paths the
runner would execute.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNNER = _REPO_ROOT / "scripts" / "run_all_scripts_release.py"
_SCRIPT = (
    _REPO_ROOT
    / "scripts"
    / "release"
    / "forward_node"
    / "certify_forward_node_production_scale.py"
)


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_release_runner_under_test", _RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_runner_discovers_the_forward_node_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _SCRIPT.is_file()
    runner = _load_runner()
    executed: list[Path] = []

    def record(args: list[str], **_: Any) -> SimpleNamespace:
        executed.append(Path(args[-1]).resolve())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        runner,
        "subprocess",
        SimpleNamespace(run=record, CalledProcessError=subprocess.CalledProcessError),
    )
    monkeypatch.setattr(runner, "time", SimpleNamespace(sleep=lambda _s: None))

    runner.main()

    assert executed.count(_SCRIPT.resolve()) == 1
