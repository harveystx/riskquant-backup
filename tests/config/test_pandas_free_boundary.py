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
ingest/validate/compile pipelines, and `pandas.DataFrame`
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
}


def _files_importing_pandas() -> dict[str, str]:
    offenders = {}
    for path in sorted(CONFIG_ROOT.rglob("*.py")):
        rel = path.relative_to(CONFIG_ROOT).as_posix()
        source = path.read_text(encoding="utf-8", errors="replace")
        if "import pandas" in source:
            offenders[rel] = source
    return offenders


def test_polars_is_now_the_ingest_pipeline_technology():
    """
    UPDATED (RQ-AUD-018 remediation, 2026-09-09): pandas has been fully
    replaced by polars throughout the static_config
    loader/validate/compile pipelines. This test previously asserted ZERO
    polars usage (documenting the gap); it now asserts polars is used in
    every one of the 6 files that previously imported pandas, confirming
    the migration actually happened rather than merely removing the
    pandas import without replacing the underlying functionality.

    See test_config_package_is_pandas_free below for the corresponding
    "pandas is gone" half of this same fact.
    """
    expected_polars_files = {
        "static_config/loader.py",
        "static_config/validate.py",
        "static_config/compile.py",
    }
    polars_files = {
        p.relative_to(CONFIG_ROOT).as_posix()
        for p in CONFIG_ROOT.rglob("*.py")
        if "import polars" in p.read_text(encoding="utf-8", errors="replace")
    }
    missing = expected_polars_files - polars_files
    assert not missing, (
        f"Expected polars usage in every migrated file; missing from: {sorted(missing)}"
    )


def test_config_package_is_pandas_free():
    """
    EXPECTED TO FAIL TODAY. The full ingest/validate/compile pipeline
    (`loader.py`, `validate.py`, `compile.py` in both `static_config/` and
    `static_config/`) uses pandas for genuine data processing
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
    diagnostic and becomes a live constitutional violation.

    Pure-Python text search (no subprocess/`grep` dependency): earlier
    versions of this test shelled out to the `grep` binary, which is not
    guaranteed present on every platform/CI runner (notably Windows without
    a POSIX toolchain). A missing `grep` would make this test fail for a
    reason entirely unrelated to what it actually checks.
    """
    repo_src = CONFIG_ROOT.parent.parent  # src/riskquant
    for rel in _ISOLATED_DIAGNOSTIC_FILES:
        module_suffix = rel.replace("/", ".").removesuffix(".py")
        pattern = f"config.{module_suffix}"
        this_file_name = Path(rel).name

        importers = []
        for path in repo_src.rglob("*.py"):
            if path.name == this_file_name:
                continue  # skip the file's own definition
            source = path.read_text(encoding="utf-8", errors="replace")
            if pattern in source:
                importers.append(str(path))

        assert importers == [], (
            f"{rel} is documented as an isolated diagnostic with zero "
            f"production importers, but found: {importers}. Pandas usage "
            f"here is no longer isolated and must be re-audited."
        )
