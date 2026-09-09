# tests/config/test_pandas_free_boundary.py
"""
Architecture test for the `riskquant.config` design audit (RQ-AUD-018).

The constitution (`.github/copilot-instructions.md` §3, §13) states:

    "Polars is the sole approved internal dataframe technology... Pandas
    is not an approved internal RiskQuant dataframe representation. Pandas
    may be used only at an unavoidable external interoperability boundary
    ... such use must be isolated in EDGE and converted immediately to a
    polars.LazyFrame."

This test documents the CURRENT state of `riskquant.config`, which is
NOT pandas-free: `pandas` is imported throughout the `static_config` and
`proxy_config` ingest/validate/compile pipelines, and `pandas.DataFrame`
is never converted to `polars.LazyFrame` anywhere in the package (zero
`polars` imports exist in `riskquant.config` at all).

This test is EXPECTED TO FAIL until the pandas->Polars migration is
completed. It exists so the violation is an explicit, tracked, executable
fact rather than something that has to be rediscovered by a future audit.
Do not silence this test by weakening the assertion; either fix the
package or explicitly mark the test `xfail` with a linked decision record.
"""
from __future__ import annotations

from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "src" / "riskquant" / "config"

# Files that self-declare an isolated, zero-consumer EDGE diagnostic role
# (never imported by valuation/simlab/credit/any kernel — verified by the
# RQ-AUD-018 audit). These are a *separate*, lower-severity finding: pandas
# is not technically required here at all (a Polars-based diagnostic dumper
# would work identically), but the isolation means the constitutional
# violation does not propagate into any production numerical path.
_ISOLATED_DIAGNOSTIC_FILES = {
    "static_config/audit.py",
    "proxy_config/audit.py",
}


def _files_importing_pandas() -> dict[str, str]:
    offenders = {}
    for path in sorted(CONFIG_ROOT.rglob("*.py")):
        rel = str(path.relative_to(CONFIG_ROOT))
        source = path.read_text(encoding="utf-8", errors="replace")
        if "import pandas" in source:
            offenders[rel] = source
    return offenders


def test_no_polars_usage_at_all_documents_the_gap():
    """`riskquant.config` never imports polars anywhere. This is not a
    failure by itself (a package could legitimately need no dataframe
    library), but combined with test_config_package_is_pandas_free below,
    it demonstrates that pandas was never replaced BY polars -- the
    ingest/validate/compile pipeline is pandas end-to-end, not
    'pandas at an isolated EDGE adapter converting immediately to Polars'."""
    polars_files = [
        str(p.relative_to(CONFIG_ROOT))
        for p in CONFIG_ROOT.rglob("*.py")
        if "import polars" in p.read_text(encoding="utf-8", errors="replace")
    ]
    assert polars_files == [], (
        "Expected zero polars usage today (documents the current gap); "
        f"found polars in: {polars_files}. If this now fails because polars "
        "usage was added, the fix is in progress -- update this test to "
        "reflect the new (improving) state rather than deleting it."
    )


def test_config_package_is_pandas_free():
    """
    EXPECTED TO FAIL TODAY. The full ingest/validate/compile pipeline
    (`loader.py`, `validate.py`, `compile.py` in both `static_config/` and
    `proxy_config/`) uses pandas for genuine data processing
    (`.loc`, `.groupby`, `pd.concat`, `.sort_values`, `pd.to_datetime`),
    not merely as a type annotation -- this is the exact "Prohibited
    internal pattern" named in copilot-instructions.md §13.
    """
    offenders = _files_importing_pandas()
    non_isolated = {
        f: src for f, src in offenders.items() if f not in _ISOLATED_DIAGNOSTIC_FILES
    }
    assert not non_isolated, (
        "riskquant.config imports pandas outside the isolated audit/"
        "diagnostic files (RQ-AUD-018, CRITICAL for a Tier-1 institution "
        "per the constitution's explicit Pandas prohibition):\n"
        + "\n".join(f"  {f}" for f in sorted(non_isolated))
    )


def test_isolated_diagnostic_files_have_zero_production_importers():
    """The two files where pandas usage is currently tolerated
    (`audit.py` in each sub-package) must remain genuinely isolated: no
    other module in the repository may import them. If a production path
    starts depending on them, the pandas usage stops being an isolated
    diagnostic and becomes a live constitutional violation."""
    import subprocess

    repo_src = CONFIG_ROOT.parent.parent  # src/riskquant
    for rel in _ISOLATED_DIAGNOSTIC_FILES:
        module_suffix = rel.replace("/", ".").removesuffix(".py")
        pattern = f"config.{module_suffix}"
        result = subprocess.run(
            ["grep", "-rl", pattern, str(repo_src)],
            capture_output=True,
            text=True,
        )
        importers = [
            line
            for line in result.stdout.splitlines()
            if not line.endswith(rel.replace("/", "/") if "/" in rel else rel)
            and Path(line).name != Path(rel).name
        ]
        assert importers == [], (
            f"{rel} is documented as an isolated diagnostic with zero "
            f"production importers, but found: {importers}. Pandas usage "
            f"here is no longer isolated and must be re-audited."
        )
