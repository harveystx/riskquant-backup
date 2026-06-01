# src/riskquant/ingest_fixtures/config/risk_static_config/canonical_risk_static_config.py

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import pandas as pd

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
from riskquant.ingest_fixtures.config.risk_static_config._parts.spot_roll_rule import (
    CANONICAL_SPOT_ROLL_RULE_DATA,
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
        "spot_roll_rule": CANONICAL_SPOT_ROLL_RULE_DATA,
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


CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS: dict[str, pd.DataFrame] = {
    table_name: pd.DataFrame(cast(Mapping[str, Sequence[Any]], table_data))
    for table_name, table_data in CANONICAL_RISK_STATIC_CONFIG_DATA.items()
}


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print_dict(
        CANONICAL_RISK_STATIC_CONFIG_DATA, title="Canonical Risk Static Config Data"
    )

    print("As DataFrames:")
    for sheet_name, df in CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS.items():
        title = f"Sheet: {sheet_name}  (rows={len(df)})"
        print_df(df, index=False, title=title)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
