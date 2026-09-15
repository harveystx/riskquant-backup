# riskquant_lab/polars_display.py
from __future__ import annotations

import shutil

import polars as pl


def configure_polars_display(
    *,
    max_rows: int = 100,
    max_cols: int = 50,
    float_precision: int = 2,
) -> None:
    """Set global Polars display configurations for console output."""

    # 1. Strings and Numerics
    pl.Config.set_fmt_str_lengths(1_000)
    pl.Config.set_thousands_separator("_")
    pl.Config.set_float_precision(float_precision)
    pl.Config.set_tbl_cell_numeric_alignment("RIGHT")

    # 2. Dimensions (Safety over infinite)
    pl.Config.set_tbl_rows(max_rows)
    pl.Config.set_tbl_cols(max_cols)

    # 3. Terminal Awareness
    # Fallback to 120 if not running in a true TTY (like CI/CD pipelines)
    terminal_width = shutil.get_terminal_size(fallback=(120, 50)).columns
    pl.Config.set_tbl_width_chars(terminal_width)

    # 4. Aesthetics and Nested Data
    pl.Config.set_tbl_formatting("ASCII_FULL_CONDENSED")
    pl.Config.set_fmt_table_cell_list_len(5)
