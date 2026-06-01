# src/riskquant/ingest_fixtures/config/risk_static_config/_parts/spot_roll_rule.py

from __future__ import annotations

import pandas as pd

from riskquant.ingest_fixtures.schemas import RiskStaticConfigSpotRollRule
from riskquant.ingest_fixtures.utils import validate_column_lengths

# Three rules, each covering one allowed (XAxisKind, RollMethod) pair
# that is referenced by at least one curve or surface in the canonical
# fixture. Rules with no referrer have been dropped — the goal is a
# fixture that exactly matches the runtime needs, not a feature catalogue.
#
# Allowed pairs (see ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS in
# riskquant.config.enums.roll_methods):
#
#   NEXT_MONTH               : (OFFSET_M,  CALENDAR_PERIOD)
#   ANNUAL_ROLL_APRIL_BGN    : (OFFSET_Y,  ROLL_ON_MMDD)
#   HALF_YEAR_LABEL_APR_OCT  : (OFFSET_HY, HALF_YEAR_BY_LABEL)


def _get_data() -> RiskStaticConfigSpotRollRule:
    return {
        "rule_name": [
            "NEXT_MONTH",
            "ANNUAL_ROLL_APRIL_BGN",
            "HALF_YEAR_LABEL_APR_OCT",
        ],
        "roll_method": [
            "CALENDAR_PERIOD",
            "ROLL_ON_MMDD",
            "HALF_YEAR_BY_LABEL",
        ],
        "offset_in_tenor_units": [1, 0, 0],
        "roll_month": [None, 4, None],
        "roll_day": [None, 1, None],
        "fh_bh_mmdd": [None, None, "0401|1001"],
        "post_roll_shift": [0, 0, 0],
    }


CANONICAL_SPOT_ROLL_RULE_DATA: RiskStaticConfigSpotRollRule = _get_data()

validate_column_lengths(CANONICAL_SPOT_ROLL_RULE_DATA)

_CANONICAL_SPOT_ROLL_RULE_DF = pd.DataFrame(CANONICAL_SPOT_ROLL_RULE_DATA)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical Spot Roll Rules Data:")
    print_dict(CANONICAL_SPOT_ROLL_RULE_DATA)

    print("As DataFrame:")
    print_df(_CANONICAL_SPOT_ROLL_RULE_DF, index=False)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
