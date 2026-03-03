# src/riskquant/config/resolution.py
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
from typing import (
    Dict,
    Final,
    Mapping,
    Sequence,
    Tuple,
    overload,
)

import numpy as np
import pandas as pd

from riskquant.config.compile import compile_risk_static_config_from_xlsx
from riskquant.config.configs import RiskStaticConfig
from riskquant.utils.dates import get_previous_business_date_as_iso_str
from riskquant.utils.printer import print_success_message

# Ensure these utility functions are available in riskquant.utils.tenors
from riskquant.utils.tenors import (
    add_months_to_date_and_return_yyyymm01,
    fmt_date_as_str_yyyymm,
    get_spot_year,
    parse_bracket_token,
)


def build_risk_factor_name_mapping_for_as_of_date(
    *,
    config: RiskStaticConfig,
    as_of_date: date,
) -> RiskFactorNameAbsToRelMapping:
    """
    Resolve relative risk factor names into absolute names for a given as_of_date.

    Design:
        - Interrogates `config.risk_factors_dict` to define the universe of valid factors.
        - Translates each relative factor to its absolute name based on spot roll rules.
    """
    abs_to_rf_name_and_id: Dict[str, Tuple[str, int]] = {}

    for rf in config.risk_factors_dict.values():
        rf_rel_name = rf.risk_factor_name
        rf_rel_name_id = rf.risk_factor_name_id
        rf_curve_name = rf.container_name
        rf_tenor_token = rf.tenor_token

        # Get container meta
        rf_cm = config.container_meta_dict.get(rf_curve_name)
        if rf_cm is None:
            raise ValueError(
                "risk_factors_dict references unknown container_name.\n"
                f"container_name={rf_curve_name!r}\nrelative={rf_rel_name!r}"
            )

        # Get alias schema
        rf_alias = config.alias_schema_dict.get(rf_cm.alias_type_name)
        if rf_alias is None:
            raise ValueError(
                "container_meta_dict references unknown alias_type_name.\n"
                f"container_name={rf_cm.container_name!r}\nalias_type_name={rf_cm.alias_type_name!r}"
            )

        # ================================ #
        # === RESOLVE TO ABSOLUTE NAME === #
        # ================================ #

        if rf_alias.alias_kind == "NONE":
            if rf_tenor_token is not None:
                raise ValueError(
                    f"Risk factor ID tenor_token ({rf_tenor_token}) must be None for alias_kind = 'NONE'.\n"
                    f"relative={rf_rel_name!r}\ncurve_name={rf_curve_name!r}"
                )
            abs_to_rf_name_and_id[rf_rel_name] = (rf_rel_name, rf_rel_name_id)
            continue

        elif rf_alias.alias_kind == "LABEL":
            if rf_tenor_token is None:
                raise ValueError(
                    "Risk factor missing tenor_token for alias_kind = 'LABEL'.\n"
                    f"relative={rf_rel_name!r}\ncurve_name={rf_curve_name!r}"
                )
            label, signed_offset = parse_bracket_token(rf_tenor_token)
            if signed_offset is not None:
                raise ValueError(
                    f"alias_kind = 'LABEL' tenor_token ('{rf_tenor_token}') must not have +/- offset.\n"
                    f"relative={rf_rel_name!r}\ncurve_name={rf_curve_name!r}"
                )
            rf_abs_name = f"{rf_curve_name}_{label}"
            abs_to_rf_name_and_id[rf_abs_name] = (rf_rel_name, rf_rel_name_id)
            continue

        elif rf_alias.alias_kind == "OFFSET":
            if rf_tenor_token is None:
                raise ValueError(
                    "alias_kind = OFFSET risk factor missing tenor_token.\n"
                    f"relative={rf_rel_name!r}\ncurve_name={rf_curve_name!r}"
                )

            if (
                not rf_cm.spot_roll_rule_name
                or rf_cm.spot_roll_rule_name == "None"
            ):
                raise ValueError(
                    "alias_kind = OFFSET curve missing spot_roll_rule_name in container_meta.\n"
                    f"relative={rf_rel_name!r}\ncurve_name={rf_curve_name!r}"
                )

            tenor_token_unit, tenor_token_signed_offset = parse_bracket_token(
                rf_tenor_token
            )
            if tenor_token_signed_offset is None:
                raise ValueError(
                    "alias_kind = OFFSET tenor_token must have +/- integer.\n"
                    f"relative={rf_rel_name!r}\ncurve_name={rf_curve_name!r}\ntenor_token={rf_tenor_token!r}\n"
                )

            # Retrieve rule
            rf_srr = config.spot_roll_rules_dict.get(
                (rf_cm.spot_roll_rule_name, rf_cm.tenor_type_name)
            )
            if rf_srr is None:
                raise ValueError(
                    "Missing spot roll rule in spot_roll_rules_dict!\n"
                    f"spot_roll_rule_name={rf_cm.spot_roll_rule_name!r}\n"
                    f"tenor_type_name={rf_cm.tenor_type_name!r}\n"
                    f"relative={rf_rel_name!r}\ncurve_name={rf_curve_name!r}"
                )

            rf_spot_roll_method = rf_srr.spot_roll_method
            rf_tenor_type_name = rf_cm.tenor_type_name

            rf_offset = int(rf_srr.offset_in_tenor_units or 0)
            rf_roll_month = int(rf_srr.roll_month or 0)
            rf_roll_day = int(rf_srr.roll_day or 0)
            rf_post_shift = int(rf_srr.post_roll_shift or 0)

            # Resolution Logic
            if (
                rf_tenor_type_name == "TENOR_YM"
                and rf_spot_roll_method == "CALENDAR_PERIOD"
            ):
                spot_month_yyyymm01 = add_months_to_date_and_return_yyyymm01(
                    d=date(as_of_date.year, as_of_date.month, 1),
                    months_to_add=rf_offset,
                )
                rf_abs_date_yyyymm01 = add_months_to_date_and_return_yyyymm01(
                    d=spot_month_yyyymm01,
                    months_to_add=int(tenor_token_signed_offset),
                )
                rf_abs_name = f"{rf_curve_name}_{fmt_date_as_str_yyyymm(rf_abs_date_yyyymm01)}"
                abs_to_rf_name_and_id[rf_abs_name] = (
                    rf_rel_name,
                    rf_rel_name_id,
                )
                continue

            elif (
                rf_tenor_type_name == "TENOR_Y"
                and rf_spot_roll_method == "ROLL_ON_MMDD"
            ):
                spot_year = get_spot_year(
                    as_of_date=as_of_date,
                    roll_month=rf_roll_month,
                    roll_day=rf_roll_day,
                    post_roll_shift=rf_post_shift,
                )
                year = spot_year + int(tenor_token_signed_offset)
                rf_abs_name = f"{rf_curve_name}_{year:04d}"
                abs_to_rf_name_and_id[rf_abs_name] = (
                    rf_rel_name,
                    rf_rel_name_id,
                )
                continue

            elif (
                rf_tenor_type_name == "TENOR_YMD"
                and rf_spot_roll_method == "HALF_YEAR_BY_DATE"
            ):
                fh_bh_ddmm = rf_srr.fh_bh_ddmm
                if not fh_bh_ddmm:
                    raise ValueError(
                        "spot_roll_method = 'HALF_YEAR_BY_DATE' missing fh_bh_ddmm!\n"
                        f"spot_roll_rule_name={rf_cm.spot_roll_rule_name!r}\n"
                        f"relative={rf_rel_name!r}\ncurve_name={rf_curve_name!r}"
                    )

                fh_mmdd, bh_mmdd = fh_bh_ddmm.split("|")
                if len(fh_mmdd) != 4 or len(bh_mmdd) != 4:
                    raise ValueError(
                        f"fh_bh_ddmm ('{fh_bh_ddmm}') must be of form 'ddmm|ddmm'!"
                    )

                as_of_year = as_of_date.year
                look_years = max(3, abs(int(tenor_token_signed_offset)) + 1)

                boundaries: list[date] = []
                for y in range(
                    as_of_year - look_years, as_of_year + look_years + 1
                ):
                    boundaries.append(
                        date(y, int(fh_mmdd[:2]), int(fh_mmdd[2:]))
                    )
                    boundaries.append(
                        date(y, int(bh_mmdd[:2]), int(bh_mmdd[2:]))
                    )
                boundaries.sort()

                idx_hy0 = bisect_left(boundaries, as_of_date)
                idx_target = idx_hy0 + int(tenor_token_signed_offset)

                if idx_target < 0 or idx_target >= len(boundaries):
                    raise ValueError(
                        "HALF_YEAR_BY_DATE offset out of range for constructed boundary grid.\n"
                        f"curve_name={rf_curve_name!r}\nrelative={rf_rel_name!r}\n"
                    )

                abs_dt = boundaries[idx_target]
                yyyymmdd_str = (
                    f"{abs_dt.year:04d}{abs_dt.month:02d}{abs_dt.day:02d}"
                )
                rf_abs_name = f"{rf_curve_name}_{yyyymmdd_str}"

                abs_to_rf_name_and_id[rf_abs_name] = (
                    rf_rel_name,
                    rf_rel_name_id,
                )
                continue

            else:
                raise ValueError(
                    "Unsupported OFFSET resolution combination: "
                    f"(tenor_type_name, rf_spot_roll_method) = ('{rf_tenor_type_name}', '{rf_spot_roll_method}')\n"
                    f"curve_name={rf_curve_name!r}\nrelative={rf_rel_name!r}\n"
                )

    return RiskFactorNameAbsToRelMapping(
        as_of_date=as_of_date,
        abs_to_rf_name_and_id=abs_to_rf_name_and_id,
    )


@dataclass(frozen=True, slots=True)
class RiskFactorNameAbsToRelMapping:
    as_of_date: date
    abs_to_rf_name_and_id: Mapping[str, Tuple[str, int]]  # abs -> (rel, id)

    def __post_init__(self) -> None:
        seen: dict[int, str] = {}

        for abs_name, (rel_name, rf_id) in self.abs_to_rf_name_and_id.items():
            if not isinstance(rel_name, str):
                raise ValueError(
                    f"rel_name for '{abs_name}' must be str, got {type(rel_name).__name__}"
                )
            if not isinstance(rf_id, (int, np.integer)):
                raise ValueError(
                    f"id for '{abs_name}' must be an integer, got {type(rf_id).__name__}={rf_id!r}"
                )

            iv = int(rf_id)
            prev = seen.get(iv)
            if prev is not None and prev != abs_name:
                raise ValueError(
                    "Mapping contains duplicate ID assignment:\n"
                    f"  '{prev}' and '{abs_name}' both map to ID={iv}\n"
                    "Each absolute risk factor must map to a unique risk_factor_name_id."
                )
            seen[iv] = abs_name

    @staticmethod
    def from_config(
        config: RiskStaticConfig,
        as_of_date: date,
    ) -> "RiskFactorNameAbsToRelMapping":
        return build_risk_factor_name_mapping_for_as_of_date(
            config=config,
            as_of_date=as_of_date,
        )

    @overload
    def get(self, abs_name: str) -> tuple[str, int]: ...

    @overload
    def get(self, abs_name: Sequence[str]) -> list[tuple[str, int]]: ...

    def get(
        self,
        abs_name: str | Sequence[str],
    ) -> tuple[str, int] | list[tuple[str, int]]:
        m = self.abs_to_rf_name_and_id
        if isinstance(abs_name, str):
            return m[abs_name]
        return [m[a] for a in abs_name]

    @overload
    def get_rf_name_for_abs_name(self, abs_name: str) -> str: ...

    @overload
    def get_rf_name_for_abs_name(
        self, abs_name: Sequence[str]
    ) -> list[str]: ...

    def get_rf_name_for_abs_name(
        self,
        abs_name: str | Sequence[str],
    ) -> str | list[str]:
        m = self.abs_to_rf_name_and_id
        if isinstance(abs_name, str):
            return m[abs_name][0]
        return [m[a][0] for a in abs_name]

    @overload
    def get_rf_id_for_abs_name(self, abs_name: str) -> int: ...

    @overload
    def get_rf_id_for_abs_name(self, abs_name: Sequence[str]) -> list[int]: ...

    def get_rf_id_for_abs_name(
        self,
        abs_name: str | Sequence[str],
    ) -> int | list[int]:
        m = self.abs_to_rf_name_and_id
        if isinstance(abs_name, str):
            return m[abs_name][1]
        return [m[a][1] for a in abs_name]

    def to_df(self) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
          [risk_factor_name, risk_factor_id]
        Rows are deterministically ordered by absolute_name (the index).
        """
        items = sorted(self.abs_to_rf_name_and_id.items(), key=lambda x: x[0])
        abs_names = [k for k, _ in items]
        rel_names = [v[0] for _, v in items]
        ids = np.fromiter(
            (int(v[1]) for _, v in items), dtype=np.int32, count=len(items)
        )

        df = pd.DataFrame(
            {
                "risk_factor_name": rel_names,
                "risk_factor_id": ids,
            },
            index=pd.Index(abs_names, name="absolute_risk_factor_name"),
        )
        return df

    @property
    def df(self) -> pd.DataFrame:
        """Alias for to_dataframe(), for convenient access."""
        return self.to_df()

    @property
    def abs_to_rf_name(self) -> Dict[str, str]:
        return {
            abs_name: str(rf_name_and_id[0])
            for abs_name, rf_name_and_id in self.abs_to_rf_name_and_id.items()
        }

    @property
    def abs_to_rf_id(self) -> Dict[str, int]:
        return {
            abs_name: int(rf_name_and_id[1])
            for abs_name, rf_name_and_id in self.abs_to_rf_name_and_id.items()
        }

    @property
    def rf_name_to_rf_id(self) -> Dict[str, int]:
        return {
            rf_name_and_id[0]: int(rf_name_and_id[1])
            for _, rf_name_and_id in self.abs_to_rf_name_and_id.items()
        }


if __name__ == "__main__":
    DEV_DATA_DIR: Final[str] = (
        "C:/Users/mharvey/Documents/code/riskquant/data/dev"
    )

    DEV_DATA_FILE_NAME_STATIC_CONFIG: Final[str] = (
        "dev_risk_static_data_config_v01.xlsx"
    )

    # Path to the .xlsx workbook containing the static config sheets.
    xlsx_path = DEV_DATA_DIR + "/" + DEV_DATA_FILE_NAME_STATIC_CONFIG

    # Run the full compile process and print the resulting config object.
    compiled_config = compile_risk_static_config_from_xlsx(path=xlsx_path)

    # Get as of date.
    as_of_date = get_previous_business_date_as_iso_str()

    # Get risk factor mapping.
    mapping = RiskFactorNameAbsToRelMapping.from_config(
        as_of_date=as_of_date,
        config=compiled_config,
    )

    # Test the mapping methods.
    print("\nTesting mapping methods:")

    # Test single absolute name lookup.
    print("\n\nSingle lookup:")
    print(mapping.get("ABS_RF_1"))  # Output: ('REL_RF_A', 101)

    # Test batch absolute name lookup.
    print("\n\nBatch lookup:")
    print(
        mapping.get(["ABS_RF_1", "ABS_RF_2"])
    )  # Output: [('REL_RF_A', 101), ('REL_RF_B', 102)]

    # Test convenience methods for getting just the relative name or ID.
    print("\n\nConvenience methods:")
    print("\n\nGet relative names for absolute names:")
    print(mapping.get_rf_name_for_abs_name("ABS_RF_2"))  # Output: 'REL_RF_B'
    print("\n\nGet IDs for absolute names:")
    print(mapping.get_rf_id_for_abs_name("ABS_RF_3"))  # Output: 103
    print("\n\nGet mapping as DataFrame:")
    print(mapping.to_df())

    # Property accessors for the same data.
    print("\n\nProperty accessors:")
    print("\n\nAbsolute to relative name mapping:")
    print(mapping.abs_to_rf_name)  # Output: {'ABS_RF_1': 'REL_RF_A', ...}
    print("\n\nAbsolute to relative ID mapping:")
    print(mapping.abs_to_rf_id)  # Output: {'ABS_RF_1':   101, ...}
    print("\n\nRelative name to relative ID mapping:")
    print(mapping.rf_name_to_rf_id)  # Output: {'REL_RF_A': 101, ...}

    # Print success message to confirm that the test ran without errors.
    print_success_message()
