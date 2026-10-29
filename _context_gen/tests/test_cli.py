# _context_gen/tests/test_cli.py
from __future__ import annotations

import pytest

from _context_gen.cli import _build_parser


# Literal list, not cli.SPECIAL_PROFILES: a test parametrized over the
# constant it checks cannot detect a name removed from that constant.
@pytest.mark.parametrize("profile", ["all", "index", "core", "tests", "scripts"])
def test_every_special_profile_is_accepted(profile: str) -> None:
    # Regression test: `scripts` was handled in main() but missing from the
    # argparse choices, so `--profile scripts` failed at parse time.
    args = _build_parser(["credit"]).parse_args(["--profile", profile])
    assert args.profile == profile


def test_discovered_domain_is_accepted() -> None:
    args = _build_parser(["credit", "solutions_credit_var"]).parse_args([
        "--profile",
        "solutions_credit_var",
    ])
    assert args.profile == "solutions_credit_var"


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(SystemExit):
        _build_parser(["credit"]).parse_args(["--profile", "tests_credit"])
