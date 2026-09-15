# src/riskquant/ingest_fixtures/config/risk_static_config/canonical_risk_static_config.py

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import polars as pl

from riskquant.ingest_fixtures.config.risk_static_config._parts.ccy import (
    CANONICAL_CCY_DATA,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.curve_meta import (
    CANONICAL_CURVE_META_DATA,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.curve_point_meta import (
    CANONICAL_CURVE_POINT_META_DATA,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.holiday_calendars import (
    CANONICAL_HOLIDAY_CALENDARS_DATA,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.node_meta import (
    CANONICAL_NODE_META_DATA,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.surface_meta import (
    CANONICAL_SURFACE_META_DATA,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.surface_point_meta import (
    CANONICAL_SURFACE_POINT_META_DATA,
)
from riskquant.ingest_fixtures.schemas import RiskStaticConfigData
from riskquant.ingest_fixtures.utils import validate_column_lengths


def _get_data() -> RiskStaticConfigData:
    return {
        "ccy": CANONICAL_CCY_DATA,
        "holiday_calendars": CANONICAL_HOLIDAY_CALENDARS_DATA,
        "node_meta": CANONICAL_NODE_META_DATA,
        "curve_meta": CANONICAL_CURVE_META_DATA,
        "curve_point_meta": CANONICAL_CURVE_POINT_META_DATA,
        "surface_meta": CANONICAL_SURFACE_META_DATA,
        "surface_point_meta": CANONICAL_SURFACE_POINT_META_DATA,
    }


CANONICAL_RISK_STATIC_CONFIG_DATA: RiskStaticConfigData = _get_data()


# Per-table column-length consistency check. The compiler ALSO validates
# schema membership and referential integrity downstream; this layer
# only enforces "each table's columns are the same length".
for _table_name, _table_data in CANONICAL_RISK_STATIC_CONFIG_DATA.items():
    validate_column_lengths(_table_data)


# Polars, matching riskquant.config.static_config's compile pipeline
# (riskquant.config.static_config.compile.compile_risk_static_config_from_dict_of_dfs
# takes dict[str, pl.DataFrame], not pandas -- see RQ-AUD-018).
CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS: dict[str, pl.DataFrame] = {
    table_name: pl.DataFrame(cast(Mapping[str, Sequence[Any]], table_data))
    for table_name, table_data in CANONICAL_RISK_STATIC_CONFIG_DATA.items()
}


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print_dict(
        CANONICAL_RISK_STATIC_CONFIG_DATA, title="Canonical Risk Static Config Data"
    )

    print("As DataFrames:")
    for sheet_name, df in CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS.items():
        print(f"Sheet: {sheet_name}  (rows={df.height})")
        print(df)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    from pathlib import Path

    import xlsxwriter

    _debug_print()

    # --- SAVE TO EXCEL FOR MANUAL INSPECTION ---

    # Path to data folder is relative to the root of the repository.
    fp = Path(
        Path(__file__).resolve().parent.parent.parent.parent.parent.parent,
        "data",
        "canonical_examples",
        "canonical_risk_static_config.xlsx",
    )
    fp.parent.mkdir(parents=True, exist_ok=True)

    print(fp)
    # Save the dict of dataframes into a single multi-sheet Excel workbook
    # for manual inspection, if desired. Polars' write_excel accepts a
    # shared, not-yet-closed xlsxwriter.Workbook so multiple sheets can be
    # written to one file, mirroring pandas' ExcelWriter context-manager
    # pattern used before the RQ-AUD-018 pandas -> Polars migration.
    workbook = xlsxwriter.Workbook(str(fp))
    for sheet_name, df in CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS.items():
        df.write_excel(workbook=workbook, worksheet=sheet_name)
    workbook.close()
    print(f"Saved canonical risk static config to Excel at: {fp}")
