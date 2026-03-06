# src/riskquant/config/resolution.py
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date


from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import overload

import numpy as np
import pandas as pd

# Ensure these utility functions are available in riskquant.utils.tenors
from riskquant.config.configs import RiskStaticConfig
from riskquant.utils.tenors import (
    add_months_to_date_and_return_yyyymm01,
    fmt_date_as_str_yyyymm,
    get_spot_year,
    parse_bracket_token,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RiskFactorAbsToRelNameIdMap(Mapping[str, tuple[str, int]]):
    as_of_date: date
    abs_to_rel_name_and_id: Mapping[str, tuple[str, int]]  # abs -> (rel, id)

    # NO INIT

    # Inverted mapping for convenient reverse lookups: rel -> (abs, id)
    _rel_to_abs_name_and_id: Mapping[str, tuple[str, int]] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:

        normalized = dict(self.abs_to_rel_name_and_id)

        seen_ids: dict[int, str] = {}
        seen_rel_names: dict[str, str] = {}

        for abs_name, (rel_name, rf_id) in normalized.items():
            if not isinstance(abs_name, str):
                raise ValueError(
                    f"absolute name must be str, got {type(abs_name).__name__}"
                )

            if not isinstance(rel_name, str):
                raise ValueError(
                    f"rel_name for '{abs_name}' must be str, got {type(rel_name).__name__}"
                )

            if not isinstance(rf_id, (int, np.integer)):
                raise ValueError(
                    f"id for '{abs_name}' must be an integer, got {type(rf_id).__name__}={rf_id!r}"
                )

            iv = int(rf_id)

            prev_abs_for_id = seen_ids.get(iv)
            if prev_abs_for_id is not None and prev_abs_for_id != abs_name:
                raise ValueError(
                    "Mapping contains duplicate ID assignment:\n"
                    f"  '{prev_abs_for_id}' and '{abs_name}' both map to ID={iv}\n"
                    "Each absolute risk factor must map to a unique risk_factor_name_id."
                )
            seen_ids[iv] = abs_name

            prev_abs_for_rel = seen_rel_names.get(rel_name)
            if prev_abs_for_rel is not None and prev_abs_for_rel != abs_name:
                raise ValueError(
                    "Mapping contains duplicate relative risk factor name assignment:\n"
                    f"  '{prev_abs_for_rel}' and '{abs_name}' both map to rel_name='{rel_name}'\n"
                    "Each absolute risk factor must map to a unique relative risk factor name."
                )
            seen_rel_names[rel_name] = abs_name

        # Construct the reverse mapping for rel_name -> (abs_name, id)
        object.__setattr__(
            self,
            "abs_to_rel_name_and_id",
            MappingProxyType(normalized),
        )

        # Construct the reverse mapping for rel_name -> (abs_name, id)
        reverse = {
            rel_name: (abs_name, int(rf_id))
            for abs_name, (rel_name, rf_id) in normalized.items()
        }
        object.__setattr__(
            self,
            "_rel_to_abs_name_and_id",
            MappingProxyType(reverse),
        )

    @staticmethod
    def from_config(
        config: RiskStaticConfig,
        as_of_date: date,
    ) -> "RiskFactorAbsToRelNameIdMap":
        return build_risk_factor_name_mapping_for_as_of_date(
            config=config,
            as_of_date=as_of_date,
        )

    def __getitem__(self, abs_name: str) -> tuple[str, int]:
        return self.abs_to_rel_name_and_id[abs_name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.abs_to_rel_name_and_id)

    def __len__(self) -> int:
        return len(self.abs_to_rel_name_and_id)

    @overload
    def lookup(self, abs_name: str) -> tuple[str, int]: ...
    @overload
    def lookup(self, abs_name: Iterable[str]) -> list[tuple[str, int]]: ...

    def lookup(
        self,
        abs_name: str | Iterable[str],
    ) -> tuple[str, int] | list[tuple[str, int]]:
        m = self.abs_to_rel_name_and_id
        if isinstance(abs_name, str):
            return m[abs_name]
        return [m[a] for a in abs_name]

    @overload
    def get_rel_name_for_abs_name(self, abs_name: str) -> str: ...
    @overload
    def get_rel_name_for_abs_name(
        self, abs_name: Iterable[str]
    ) -> list[str]: ...

    def get_rel_name_for_abs_name(
        self,
        abs_name: str | Iterable[str],
    ) -> str | list[str]:
        m = self.abs_to_rel_name_and_id
        if isinstance(abs_name, str):
            return m[abs_name][0]
        return [m[a][0] for a in abs_name]

    @overload
    def get_rel_id_for_abs_name(self, abs_name: str) -> int: ...
    @overload
    def get_rel_id_for_abs_name(self, abs_name: Iterable[str]) -> list[int]: ...

    def get_rel_id_for_abs_name(
        self,
        abs_name: str | Iterable[str],
    ) -> int | list[int]:
        m = self.abs_to_rel_name_and_id
        if isinstance(abs_name, str):
            return int(m[abs_name][1])
        return [int(m[a][1]) for a in abs_name]

    def to_df(self) -> pd.DataFrame:
        items = sorted(self.abs_to_rel_name_and_id.items(), key=lambda x: x[0])
        abs_names = [k for k, _ in items]
        rel_names = [v[0] for _, v in items]
        ids = np.fromiter(
            (int(v[1]) for _, v in items),
            dtype=np.int32,
            count=len(items),
        )

        return pd.DataFrame(
            {
                "risk_factor_name": rel_names,
                "risk_factor_id": ids,
            },
            index=pd.Index(abs_names, name="absolute_risk_factor_name"),
        )

    @property
    def df(self) -> pd.DataFrame:
        """Alias for to_df(), for convenient access."""
        return self.to_df()

    @property
    def abs_to_rel_name(self) -> dict[str, str]:
        return {
            abs_name: rf_name
            for abs_name, (rf_name, _) in self.abs_to_rel_name_and_id.items()
        }

    @property
    def abs_to_rel_id(self) -> dict[str, int]:
        return {
            abs_name: int(rf_id)
            for abs_name, (_, rf_id) in self.abs_to_rel_name_and_id.items()
        }

    @property
    def rel_to_abs_name(self) -> dict[str, str]:
        return {
            rf_name: abs_name
            for abs_name, (rf_name, _) in self.abs_to_rel_name_and_id.items()
        }

    @property
    def rel_name_to_id(self) -> dict[str, int]:
        return {
            rf_name: int(rf_id)
            for _, (rf_name, rf_id) in self.abs_to_rel_name_and_id.items()
        }

    # --- INVERTED MAPPINGS FOR CONVENIENCE --- #

    # rel -> (abs, id)

    @property
    def rel_to_abs_name_and_id(self) -> Mapping[str, tuple[str, int]]:
        return self._rel_to_abs_name_and_id

    # Scalar lookups: rel_name -> (abs_name, id)
    @overload
    def get_abs_name_and_id_for_rel_name(
        self, rel_name: str
    ) -> tuple[str, int]: ...
    @overload
    def get_abs_name_and_id_for_rel_name(
        self, rel_name: Iterable[str]
    ) -> list[tuple[str, int]]: ...

    def get_abs_name_and_id_for_rel_name(
        self,
        rel_name: str | Iterable[str],
    ) -> tuple[str, int] | list[tuple[str, int]]:
        m = self.rel_to_abs_name_and_id
        if isinstance(rel_name, str):
            return m[rel_name]
        return [m[r] for r in rel_name]

    # Scalar lookups: rel_name -> abs_name
    @overload
    def get_abs_name_for_rel_name(
        self,
        rel_name: str,
    ) -> str: ...
    @overload
    def get_abs_name_for_rel_name(
        self,
        rel_name: Iterable[str],
    ) -> list[str]: ...

    def get_abs_name_for_rel_name(
        self,
        rel_name: str | Iterable[str],
    ) -> str | list[str]:
        m = self.rel_to_abs_name_and_id
        if isinstance(rel_name, str):
            return m[rel_name][0]
        return [m[r][0] for r in rel_name]

    # Scalar lookups: rel_name -> id
    @overload
    def get_id_for_rel_name(
        self,
        rel_name: str,
    ) -> int: ...
    @overload
    def get_id_for_rel_name(
        self,
        rel_name: Iterable[str],
    ) -> list[int]: ...

    def get_id_for_rel_name(
        self,
        rel_name: str | Iterable[str],
    ) -> int | list[int]:
        m = self.rel_to_abs_name_and_id
        if isinstance(rel_name, str):
            return int(m[rel_name][1])
        return [int(m[r][1]) for r in rel_name]


def build_risk_factor_name_mapping_for_as_of_date(
    *,
    config: RiskStaticConfig,
    as_of_date: date,
) -> RiskFactorAbsToRelNameIdMap:
    """
    Resolve relative risk factor names into absolute names for a given as_of_date.

    Design:
        - Interrogates `config.risk_factors_dict` to define the universe of valid factors.
        - Translates each relative factor to its absolute name based on spot roll rules.
    """
    abs_to_rel_name_and_id: dict[str, tuple[str, int]] = {}

    for rf in config.risk_factors_dict.values():
        rf_rel_name = rf.risk_factor_name
        rf_rel_name_id = rf.risk_factor_name_id
        rf_container_name = rf.container_name
        rf_tenor_token = rf.tenor_token

        # Get container meta
        rf_cm = config.container_meta_dict.get(rf_container_name)
        if rf_cm is None:
            raise ValueError(
                "risk_factors_dict references unknown container_name.\n"
                f"container_name={rf_container_name!r}\n"
                f"relative={rf_rel_name!r}"
            )

        # ======================================================== #
        # === HANDLE SCALAR CONTAINERS (FX SPOT, EQUITIES, ETC) == #
        # ======================================================== #
        if rf_cm.alias_type_name is None:
            if rf_tenor_token is not None:
                raise ValueError(
                    f"Risk factor ID tenor_token ({rf_tenor_token}) must be None when alias_type_name is None.\n"
                    f"relative={rf_rel_name!r}\n"
                    "container_name={rf_container_name!r}"
                )
            # Scalar factors just map to themselves (no term structure to roll)
            abs_to_rel_name_and_id[rf_rel_name] = (rf_rel_name, rf_rel_name_id)
            continue

        # Get alias schema
        rf_alias = config.alias_schema_dict.get(rf_cm.alias_type_name)
        if rf_alias is None:
            raise ValueError(
                "container_meta_dict references unknown alias_type_name.\n"
                f"container_name={rf_cm.container_name!r}\n"
                "alias_type_name={rf_cm.alias_type_name!r}"
            )

        # ================================ #
        # === RESOLVE TO ABSOLUTE NAME === #
        # ================================ #

        if rf_alias.alias_kind == "NONE":
            if rf_tenor_token is not None:
                raise ValueError(
                    f"Risk factor ID tenor_token ({rf_tenor_token}) must be None for alias_kind = 'NONE'.\n"
                    f"relative={rf_rel_name!r}\ncurve_name={rf_container_name!r}"
                )
            abs_to_rel_name_and_id[rf_rel_name] = (rf_rel_name, rf_rel_name_id)
            continue

        elif rf_alias.alias_kind == "LABEL":
            if rf_tenor_token is None:
                raise ValueError(
                    "Risk factor missing tenor_token for alias_kind = 'LABEL'.\n"
                    f"relative={rf_rel_name!r}\ncurve_name={rf_container_name!r}"
                )
            label, signed_offset = parse_bracket_token(rf_tenor_token)
            if signed_offset is not None:
                raise ValueError(
                    f"alias_kind = 'LABEL' tenor_token ('{rf_tenor_token}') must not have +/- offset.\n"
                    f"relative={rf_rel_name!r}\ncurve_name={rf_container_name!r}"
                )
            rf_abs_name = f"{rf_container_name}_{label}"
            abs_to_rel_name_and_id[rf_abs_name] = (rf_rel_name, rf_rel_name_id)
            continue

        elif rf_alias.alias_kind == "OFFSET":
            if rf_tenor_token is None:
                raise ValueError(
                    "alias_kind = OFFSET risk factor missing tenor_token.\n"
                    f"relative={rf_rel_name!r}\ncurve_name={rf_container_name!r}"
                )

            if (
                not rf_cm.spot_roll_rule_name
                or rf_cm.spot_roll_rule_name == "None"
            ):
                raise ValueError(
                    "alias_kind = OFFSET curve missing spot_roll_rule_name in container_meta.\n"
                    f"relative={rf_rel_name!r}\ncurve_name={rf_container_name!r}"
                )

            tenor_token_unit, tenor_token_signed_offset = parse_bracket_token(
                rf_tenor_token
            )
            if tenor_token_signed_offset is None:
                raise ValueError(
                    "alias_kind = OFFSET tenor_token must have +/- integer.\n"
                    f"relative={rf_rel_name!r}\ncurve_name={rf_container_name!r}\ntenor_token={rf_tenor_token!r}\n"
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
                    f"relative={rf_rel_name!r}\ncurve_name={rf_container_name!r}"
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
                rf_abs_name = f"{rf_container_name}_{fmt_date_as_str_yyyymm(rf_abs_date_yyyymm01)}"
                abs_to_rel_name_and_id[rf_abs_name] = (
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
                rf_abs_name = f"{rf_container_name}_{year:04d}"
                abs_to_rel_name_and_id[rf_abs_name] = (
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
                        f"relative={rf_rel_name!r}\ncurve_name={rf_container_name!r}"
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
                        f"curve_name={rf_container_name!r}\nrelative={rf_rel_name!r}\n"
                    )

                abs_dt = boundaries[idx_target]
                yyyymmdd_str = (
                    f"{abs_dt.year:04d}{abs_dt.month:02d}{abs_dt.day:02d}"
                )
                rf_abs_name = f"{rf_container_name}_{yyyymmdd_str}"

                abs_to_rel_name_and_id[rf_abs_name] = (
                    rf_rel_name,
                    rf_rel_name_id,
                )
                continue

            else:
                raise ValueError(
                    "Unsupported OFFSET resolution combination: "
                    f"(tenor_type_name, rf_spot_roll_method) = ('{rf_tenor_type_name}', '{rf_spot_roll_method}')\n"
                    f"curve_name={rf_container_name!r}\nrelative={rf_rel_name!r}\n"
                )

    return RiskFactorAbsToRelNameIdMap(
        as_of_date=as_of_date,
        abs_to_rel_name_and_id=abs_to_rel_name_and_id,
    )


if __name__ == "__main__":
    pass
