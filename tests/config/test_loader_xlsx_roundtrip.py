# tests/config/test_loader_xlsx_roundtrip.py
"""
Regression test for RQ-AUD-018: pins the Polars-based xlsx loaders'
behaviour (active-row filtering, required-column checks, missing-sheet
errors, type coercion) against real in-memory workbooks.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from riskquant.config.proxy_config.loader import load_proxy_config_xlsx_into_df
from riskquant.config.static_config.loader import (
    load_risk_static_config_xlsx_into_dict,
)


@pytest.fixture()
def tmp_xlsx_dir(tmp_path: Path) -> Path:
    return tmp_path


def _write_proxy_workbook(
    path: Path, rows: list[tuple], active_col: bool = True
) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "proxy_config"
    header = ["proxying_level", "required_name", "proxy_name", "vol_scaling_factor"]
    if active_col:
        header.append("active")
    ws.append(header)
    for row in rows:
        ws.append(list(row))
    wb.save(str(path))


def test_proxy_loader_filters_inactive_rows(tmp_xlsx_dir):
    path = tmp_xlsx_dir / "proxy.xlsx"
    _write_proxy_workbook(
        path,
        [
            ("CONTAINER", "A", "B", 1.5, True),
            ("RISK_FACTOR", "C", "D", 2.0, False),
        ],
    )
    df = load_proxy_config_xlsx_into_df(path=str(path))
    assert df.shape[0] == 1
    assert df["required_name"][0] == "A"
    assert "active" not in df.columns


def test_proxy_loader_defaults_missing_vol_scale_to_one(tmp_xlsx_dir):
    path = tmp_xlsx_dir / "proxy.xlsx"
    _write_proxy_workbook(path, [("CONTAINER", " A ", "B", None, True)])
    df = load_proxy_config_xlsx_into_df(path=str(path))
    assert df["vol_scaling_factor"][0] == 1.0
    assert df["required_name"][0] == "A"  # stripped


def test_proxy_loader_missing_column_raises(tmp_xlsx_dir):
    path = tmp_xlsx_dir / "bad.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "sheet1"
    ws.append(["proxying_level", "required_name"])
    ws.append(["CONTAINER", "A"])
    wb.save(str(path))
    with pytest.raises(ValueError, match="Missing required column"):
        load_proxy_config_xlsx_into_df(path=str(path))


def test_proxy_loader_no_active_column_treats_all_as_active(tmp_xlsx_dir):
    path = tmp_xlsx_dir / "noactive.xlsx"
    _write_proxy_workbook(path, [("CONTAINER", "A", "B", 1.0)], active_col=False)
    df = load_proxy_config_xlsx_into_df(path=str(path))
    assert df.shape[0] == 1


def test_static_config_loader_missing_sheet_raises(tmp_xlsx_dir):
    path = tmp_xlsx_dir / "incomplete.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ccy"
    ws.append(["ccy_name", "ccy_id"])
    ws.append(["USD", 0])
    wb.save(str(path))
    with pytest.raises(ValueError, match="missing required sheet"):
        load_risk_static_config_xlsx_into_dict(path=str(path))


def test_static_config_loader_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_risk_static_config_xlsx_into_dict(path="/tmp/does_not_exist_12345.xlsx")


def test_static_config_seven_sheet_xlsx_roundtrip(tmp_xlsx_dir):
    """
    Write the canonical config to a real 7-sheet .xlsx, read it back through
    the loader, validate, and compile. Proves the roll-convention columns
    (roll_method, anchor_offset, roll_month, roll_day, fh_bh_mmdd) survive an
    Excel round-trip and that there is no spot_roll_rule sheet.
    """
    from datetime import date

    import xlsxwriter

    from riskquant.config.static_config.compile import (
        compile_risk_static_config_from_dict_of_dfs,
    )
    from riskquant.config.static_config.rf_resolution import (
        build_risk_factor_name_mapping_for_as_of_date,
    )
    from riskquant.config.static_config.validate import validate_static_config_dfs
    from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
        CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS as DFS,
    )

    path = tmp_xlsx_dir / "static_config.xlsx"
    with xlsxwriter.Workbook(str(path)) as wb:
        for sheet_name, df in DFS.items():
            df.write_excel(workbook=wb, worksheet=sheet_name)

    loaded = load_risk_static_config_xlsx_into_dict(path=str(path))
    assert "spot_roll_rule" not in loaded
    assert set(loaded) == set(DFS)  # exactly the 7 canonical sheets

    validate_static_config_dfs(loaded)
    rsc = compile_risk_static_config_from_dict_of_dfs(loaded)
    assert len(rsc.curve_meta_dict) == 13
    assert len(rsc.surface_meta_dict) == 3
    assert rsc.num_risk_factors == 168

    mapping = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=rsc, as_of_date=date(2026, 5, 23)
    )
    assert (
        mapping.get_abs_name_for_rel_name("BRENT_CRUDE_FWD[M+0]")
        == "BRENT_CRUDE_FWD_202606"
    )


def test_xlsx_and_in_memory_ingestion_are_equivalent(tmp_xlsx_dir):
    """
    The XLSX ingestion path and the in-memory path must produce semantically
    equivalent roll metadata. Load the canonical config both ways, compile
    both, and assert every curve/surface roll field is equal.
    """
    import xlsxwriter

    from riskquant.config.static_config.compile import (
        compile_risk_static_config_from_dict_of_dfs,
    )
    from riskquant.config.static_config.loader import normalize_static_config_dfs
    from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
        CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS as DFS,
    )

    path = tmp_xlsx_dir / "equiv.xlsx"
    with xlsxwriter.Workbook(str(path)) as wb:
        for sheet_name, df in DFS.items():
            df.write_excel(workbook=wb, worksheet=sheet_name)

    from_xlsx = load_risk_static_config_xlsx_into_dict(path=str(path))
    from_mem = normalize_static_config_dfs(DFS)

    # DataFrame-level: roll columns share dtype and values across both paths.
    for sheet in ("curve_meta", "surface_meta"):
        for col in ("anchor_offset", "roll_month", "roll_day", "fh_bh_mmdd"):
            assert from_xlsx[sheet].schema[col] == from_mem[sheet].schema[col]
            assert from_xlsx[sheet][col].to_list() == from_mem[sheet][col].to_list()

    # Compiled-level: per-container roll metadata is identical.
    rsc_x = compile_risk_static_config_from_dict_of_dfs(from_xlsx)
    rsc_m = compile_risk_static_config_from_dict_of_dfs(from_mem)
    for dict_name in ("curve_meta_dict", "surface_meta_dict"):
        dx = getattr(rsc_x, dict_name)
        dm = getattr(rsc_m, dict_name)
        assert dx.keys() == dm.keys()
        for key in dx:
            for field in (
                "roll_method",
                "anchor_offset",
                "roll_month",
                "roll_day",
                "fh_bh_mmdd",
            ):
                assert getattr(dx[key], field) == getattr(dm[key], field), (
                    f"{dict_name}[{key}].{field} differs between paths"
                )
