# src/riskquant/config/proxy_config/loader.py
"""
Loads proxy configuration from an Excel workbook into a clean Polars DataFrame.

The workbook MUST carry an `active` column. Rows with active=True are
returned; the `active` column is dropped from the result. If `active`
is absent the loader raises — silently treating an absent column as
"all rows active" is a real production foot-gun (drafts and historical
entries get compiled).

This loader only handles the human-facing xlsx convention. The in-memory
canonical fixture is post-filtered (no active column) and goes through
compile_proxy_topology_from_df directly.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# NOTE: pl.read_excel()'s default "calamine" engine requires the
# `fastexcel` package (>= 0.7.0) as an explicit runtime dependency --
# it is NOT bundled with a bare `pip install polars`. If this module
# raises ImportError mentioning fastexcel, add it to the project's
# dependency manifest (pyproject.toml / requirements). Polars itself
# raises a clear, actionable error in this case (see
# polars.io.spreadsheet.functions, `import_optional("fastexcel", ...)`
# ); this comment exists so the fix is a one-line dependency add, not
# a debugging session.

# "active" column is not a hard requirement.
_REQUIRED_COLS: frozenset[str] = frozenset(
    {
        "proxying_level",
        "required_name",
        "proxy_name",
        "vol_scaling_factor",
    }
)

# Columns that survive the active-filter and are returned to compile.py.
_POST_FILTER_COLS: tuple[str, ...] = (
    "proxying_level",
    "required_name",
    "proxy_name",
    "vol_scaling_factor",
)


def load_proxy_config_xlsx_into_df(
    *,
    path: str | Path,
    sheet_name: str | None = None,
) -> pl.DataFrame:
    """
    Load the proxy-config sheet and return active rows only.

    Parameters
    ----------
    path
        Path to the Excel workbook.
    sheet_name
        Sheet to read. Defaults to the first sheet if None.

    Returns
    -------
    pl.DataFrame
        Columns: proxying_level, required_name, proxy_name,
        vol_scaling_factor. `active` is consumed and dropped.

    Raises
    ------
    ValueError
        If a required column is missing, or if the sheet name does not
        exist in the workbook.
    """
    path_str = str(path)

    available_sheets = list(pl.read_excel(path_str, sheet_id=0).keys())
    if sheet_name is not None:
        if sheet_name not in available_sheets:
            raise ValueError(
                f"Missing required sheet {sheet_name!r} in {path_str!r}. "
                f"Available sheets: {available_sheets}"
            )
        _sheet: str = sheet_name
    else:
        _sheet = available_sheets[0]

    df: pl.DataFrame = pl.read_excel(path_str, sheet_name=_sheet)

    # --- required columns: explicit (no silent active fallback) ---
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s) in sheet {_sheet!r} of {path_str!r}: "
            f"{sorted(missing)}.\n"
            f"Required: {sorted(_REQUIRED_COLS)}"
        )

    # --- active filter ---
    if "active" in df.columns:
        mask = df["active"].fill_null(False).cast(pl.Boolean)
        df = df.filter(mask).select(list(_POST_FILTER_COLS))
    else:
        df = df.select(list(_POST_FILTER_COLS))

    # --- type coercions ---
    # String columns: strip whitespace, leave nulls as nulls (validator
    # rejects both null and blank; loader does NOT fill nulls with "").
    df = df.with_columns(
        [
            pl.col(col).cast(pl.Utf8).str.strip_chars()
            for col in ("proxying_level", "required_name", "proxy_name")
        ]
    )

    # vol_scaling_factor: float64 with missing -> 1.0 (identity).
    df = df.with_columns(
        pl.col("vol_scaling_factor").fill_null(1.0).cast(pl.Float64)
    )

    return df


if __name__ == "__main__":
    import pathlib  # Add this at the top of your file

    from riskquant.utils.printer import (
        print_calling_file,
        print_success_message,
    )

    print_calling_file()

    # 1. Get the absolute path of the directory containing THIS script
    current_dir = pathlib.Path(__file__).parent.resolve()

    # Project directory is 4 levels up from the current script
    project_dir = current_dir.parent.parent.parent.parent
    print(f"Project directory: {project_dir}")

    # 2. Build the path dynamically relative to the current script.
    DEV_DATA_DIR = project_dir / "data" / "dev"
    DEV_DATA_FILE_NAME = "dev_risk_base_proxy_config_v01.xlsx"

    full_path = DEV_DATA_DIR / DEV_DATA_FILE_NAME

    df = load_proxy_config_xlsx_into_df(path=str(full_path))

    print(f"\nLoaded proxy config DataFrame (shape: {df.shape}):")
    print(df)

    print_success_message()
