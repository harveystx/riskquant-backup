# riskquant/config/configs.py
from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from typing import (
    Dict,
    Any,
    Optional,
    Tuple,
    Pattern,
    Mapping,
    Sequence,
    overload,
    Callable,
)
from typing import NamedTuple

import numpy as np
import pandas as pd

from riskquant.exceptions import PrettyKeyError
from riskquant.market_data.risk_factors.shift_types import validate_shift_types_dict
from riskquant.utils.tenors import (
    parse_bracket_token,
    get_spot_year,
    add_months_to_date_and_return_yyyymm01,
    fmt_date_as_str_yyyymm,
)


# ===================== #
# --- SMALL HELPERS --- #
# ===================== #


def _require_cols(df: pd.DataFrame, cols: tuple[str, ...], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} sheet missing required columns: {missing}")


def _build_str_to_int_dict(
    *,
    df: pd.DataFrame,
    key_col: str,
    id_col: str,
    sheet_name: str,
    key_transform: Optional[Callable[[str], str]] = None,
    require_unique_keys: bool = True,
    require_non_null_keys: bool = True,
) -> Dict[str, int]:
    """
    Build Dict[str,int] from df[key_col] -> df[id_col], fail-fast on NaN IDs.

    - key_transform: optional callable applied to key strings (e.g. str.upper)
    - require_unique_keys: if True, fail-fast if key_col has duplicates
    - require_non_null_keys: if True, fail-fast if key_col has NaN/None
    """
    _require_cols(df, (key_col, id_col), sheet_name)

    key_s = df[key_col]
    id_s = df[id_col]

    # Fail-fast on missing keys (recommended for config)
    if require_non_null_keys and key_s.isna().any():
        bad_rows = df.loc[key_s.isna(), :].head(10)
        raise ValueError(
            f"{sheet_name} sheet has missing {key_col} values. Example rows:\n{bad_rows}"
        )

    # Keys as strings (copy=False where possible)
    keys = key_s.astype(str, copy=False)

    # Apply transform if provided
    if key_transform is not None:
        # .map is fine here; config tables are small and it preserves index alignment
        keys = keys.map(key_transform)

    # Fail-fast on NaN IDs
    if id_s.isna().any():
        bad = keys[id_s.isna()].unique().tolist()
        raise ValueError(f"{sheet_name} sheet has missing {id_col} for: {bad}")

    # Duplicate key check
    if require_unique_keys and not keys.is_unique:
        dup = keys[keys.duplicated()].unique().tolist()
        raise ValueError(f"{sheet_name} sheet has duplicate {key_col} values: {dup}")

    # Safe integer cast (pandas -> int64), then Python int in dict
    ids = id_s.astype("int64", copy=False)

    return {str(k): int(v) for k, v in zip(keys, ids)}


def _is_na(x: Any) -> bool:
    # Fast NA check for typical pandas scalar values
    return x is None or (isinstance(x, float) and np.isnan(x))


# ========================================================== #
# --- RISK FACTOR ABSOLUTE NAME TO RELATIVE NAME MAPPING --- #
# ========================================================== #


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

    @overload
    def get(self, abs_name: str) -> tuple[str, int]:
        ...

    @overload
    def get(self, abs_name: Sequence[str]) -> list[tuple[str, int]]:
        ...

    def get(
        self,
        abs_name: str | Sequence[str],
    ) -> tuple[str, int] | list[tuple[str, int]]:
        m = self.abs_to_rf_name_and_id
        if isinstance(abs_name, str):
            return m[abs_name]
        return [m[a] for a in abs_name]

    @overload
    def get_rf_name_for_abs_name(self, abs_name: str) -> str:
        ...

    @overload
    def get_rf_name_for_abs_name(self, abs_name: Sequence[str]) -> list[str]:
        ...

    def get_rf_name_for_abs_name(
        self,
        abs_name: str | Sequence[str],
    ) -> str | list[str]:
        m = self.abs_to_rf_name_and_id
        if isinstance(abs_name, str):
            return m[abs_name][0]
        return [m[a][0] for a in abs_name]

    @overload
    def get_rf_id_for_abs_name(self, abs_name: str) -> int:
        ...

    @overload
    def get_rf_id_for_abs_name(self, abs_name: Sequence[str]) -> list[int]:
        ...

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
          [as_of_date, absolute_name, risk_factor_name, risk_factor_id]
        Rows are deterministically ordered by absolute_name.
        """
        items = sorted(self.abs_to_rf_name_and_id.items(), key=lambda x: x[0])
        abs_names = [k for k, _ in items]
        rel_names = [v[0] for _, v in items]
        ids = np.fromiter(
            (int(v[1]) for _, v in items), dtype=np.int32, count=len(items)
        )

        df = pd.DataFrame(
            {
                # "as_of_date": self.as_of_date,
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
        # Materialize a simple abs -> rel mapping (small; created on demand)
        return {
            abs_name: str(rf_name_and_id[0])
            for abs_name, rf_name_and_id in self.abs_to_rf_name_and_id.items()
        }

    @property
    def abs_to_rf_id(self) -> Dict[str, int]:
        # Materialize a simple abs -> id mapping (small; created on demand)
        return {
            abs_name: int(rf_name_and_id[1])
            for abs_name, rf_name_and_id in self.abs_to_rf_name_and_id.items()
        }

    @property
    def rf_name_to_rf_id(self) -> Dict[str, int]:
        # Materialize a simple abs -> id mapping (small; created on demand)
        return {
            rf_name_and_id[0]: int(rf_name_and_id[1])
            for _, rf_name_and_id in self.abs_to_rf_name_and_id.items()
        }


class CurveMapping(NamedTuple):
    curve_name: str
    curve_name_id: int
    risk_factor_group: str
    tenor_type_name: str
    spot_roll_rule_name: str
    alias_type_name: str
    quote_ccy: str
    shift_type_name: str


@dataclass(frozen=True, slots=True)
class RiskFactor:
    risk_factor_name: str
    risk_factor_name_id: int
    tenor_token: Optional[str]
    curve_name: str

    def __post_init__(self) -> None:
        # Remove tenor token as a suffix only (safer than replace)
        rf_name_prefix = (
            self.risk_factor_name.removesuffix(self.tenor_token)
            if self.tenor_token
            else self.risk_factor_name
        )

        if rf_name_prefix != self.curve_name:
            raise ValueError(
                f"risk_factor_name ('{self.risk_factor_name}') must match "
                f"curve_name ('{self.curve_name}') "
                f"after removing tenor_token ('{self.tenor_token}')!"
            )


class TenorSchema(NamedTuple):
    tenor_type_name: str
    tenor_units: str
    digits: int
    regex: Pattern[str]


class AliasSchema(NamedTuple):
    alias_type_name: str
    alias_kind: str
    alias_unit: Optional[str]


class SpotRollRule(NamedTuple):
    spot_roll_rule_name: str
    tenor_type_name: str
    spot_roll_method: str
    offset_in_tenor_units: Optional[int]
    roll_month: Optional[int]
    roll_day: Optional[int]
    fh_bh_ddmm: Optional[str]
    post_roll_shift: Optional[int]


class RiskDesk(NamedTuple):
    pass


class RiskBookStructure(NamedTuple):
    risk_desk: str
    risk_portfolio: str
    risk_book: str
    risk_sub_book: str
    risk_product: Optional[str]
    query: str


class CurveProxyMap(NamedTuple):
    curve_name: str
    proxy_curve_name: str
    proxy_curve_vol_scaling_factor: Optional[float]


class RiskFactorProxyMap(NamedTuple):
    risk_factor_name: str
    proxy_risk_factor_name: str
    proxy_risk_factor_vol_scaling_factor: Optional[float]


class HolidayCalendar(NamedTuple):
    calendar_name: str
    date: Optional[Tuple[date, ...]]


@dataclass(frozen=True, slots=True)
class RiskStaticConfig:
    """
    Container for risk static configuration tables.

    """

    curve_mapping_dict: Dict[str, CurveMapping] = field(
        init=True,
        default_factory=dict,
        repr=False,
    )
    risk_factors_dict: Dict[str, RiskFactor] = field(
        init=True,
        default_factory=dict,
        repr=False,
    )
    tenor_schema_dict: Dict[str, TenorSchema] = field(
        init=True,
        default_factory=dict,
        repr=False,
    )
    alias_schema_dict: Dict[str, AliasSchema] = field(
        init=True,
        default_factory=dict,
        repr=False,
    )
    spot_roll_rules_dict: Dict[tuple[str, str], SpotRollRule] = field(
        init=True,
        default_factory=dict,
        repr=False,
    )
    ccy_dict: Dict[str, int] = field(
        init=True,
        default_factory=dict,
        repr=False,
    )
    risk_factor_groups_dict: Dict[str, int] = field(
        init=True,
        default_factory=dict,
        repr=False,
    )
    shift_types_dict: Dict[str, int] = field(
        init=True,
        default_factory=dict,
        repr=False,
    )
    instruments_dict: Dict[str, int] = field(
        init=True,
        default_factory=dict,
        repr=False,
    )
    risk_desks: Tuple[RiskDesk] = field(
        init=True,
        default_factory=tuple,
        repr=False,
    )
    risk_book_structure: Tuple[RiskBookStructure] = field(
        init=True,
        default_factory=tuple,
        repr=False,
    )
    curve_proxies_dict: Dict[str, CurveProxyMap] = field(
        init=True,
        default_factory=dict,
        repr=False,
    )
    risk_factor_proxies_dict: Dict[str, RiskFactorProxyMap] = field(
        init=True,
        default_factory=dict,
        repr=False,
    )
    holiday_calendars_dict: Dict[str, HolidayCalendar] = field(
        init=True,
        default_factory=dict,
        repr=False,
    )

    # --- INTERNAL CACHES --- #

    # internal cache
    _rf_name_to_id: Optional[Dict[str, int]] = None
    _rf_id_to_proxy_rf_id: Optional[Dict[int, Tuple[int, float]]] = None
    _rf_id_to_risk_factor: Optional[Dict[int, RiskFactor]] = None
    _rf_id_to_shift_type_id: Optional[Dict[int, int]] = None
    _curve_tenor_index: Optional[
        Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, int]]]
    ] = None

    def __post_init__(self):
        if not self.curve_mapping_dict:
            raise ValueError(
                "curve_mapping_dict is empty. Check your curve_mapping sheet and 'active' flags."
            )
        if not self.risk_factors_dict:
            raise ValueError(
                "risk_factors_dict is empty. Check your risk_factor_ids sheet and 'active' flags."
            )

    def get_instrument_id(self, instrument_name: str) -> int:
        return self.instruments_dict[instrument_name.upper()]

    def get_risk_factor_id(self, risk_factor_name: str) -> int:
        return self.risk_factors_dict[risk_factor_name].risk_factor_name_id

    @property
    def rf_name_to_id(self) -> Dict[str, int]:
        """
        A built-once cache of risk factor name to risk factor name ID:
            risk_factor_name -> risk_factor_name_id
        """
        if self._rf_name_to_id is None:
            object.__setattr__(
                self,
                "_rf_name_to_id",
                {k: v.risk_factor_name_id for k, v in self.risk_factors_dict.items()},
            )
        return self._rf_name_to_id

    @property
    def rf_id_to_proxy_rf_id(self) -> Dict[int, Tuple[int, float]]:
        """
        A build-once cache of RF ID-level proxy overrides:
            required_rf_id -> (proxy_rf_id, vol_scale)
        """
        m = self._rf_id_to_proxy_rf_id
        if m is None:
            # risk_factor_name -> risk_factor_id
            rf_name_to_id = self.rf_name_to_id  # already cached
            out: Dict[int, Tuple[int, float]] = {}
            for p in self.risk_factor_proxies_dict.values():
                rf_id = int(rf_name_to_id[p.risk_factor_name])
                proxy_rf_id = int(rf_name_to_id[p.proxy_risk_factor_name])
                scale = float(p.proxy_risk_factor_vol_scaling_factor or 1.0)
                out[rf_id] = (proxy_rf_id, scale)

            object.__setattr__(self, "_rf_id_to_proxy_rf_id", out)
            m = out
        return m

    @property
    def rf_id_to_risk_factor(self) -> Dict[int, RiskFactor]:
        """
        Cached mapping:
            risk_factor_name_id (int) -> RiskFactor

        Built once. Used for fast ID-based access during:
          - proxy resolution
          - curve lookups
          - shift-type resolution
          - simulation kernels

        This avoids repeated scans over risk_factors_dict.values().
        """
        m = self._rf_id_to_risk_factor
        if m is None:
            # Init output.
            out: Dict[int, RiskFactor] = {}

            # Loop over all RiskFactors
            for rf in self.risk_factors_dict.values():
                # RF ID.
                rf_id = int(rf.risk_factor_name_id)

                # Fail-fast on duplicate IDs (should never happen if config validated)
                if rf_id in out:
                    raise ValueError(
                        "Duplicate risk_factor_name_id detected in risk_factors_dict:\n"
                        f"risk_factor_name_id={rf_id}\n"
                        f"existing={out[rf_id].risk_factor_name!r}\n"
                        f"new={rf.risk_factor_name!r}"
                    )

                # Store
                out[rf_id] = rf

            # Set attribute.
            object.__setattr__(self, "_rf_id_to_risk_factor", out)

            # Bind to m
            m = out

        # Return
        return m

    @property
    def rf_id_to_shift_type_id(self) -> Dict[int, int]:
        """
        Cached mapping:
            risk_factor_name_id (int) -> shift_type_id (int)

        Built once. Used for fast shift type ID access in kernels.
        """
        m = self._rf_id_to_shift_type_id
        if m is not None:
            return m

        # Local binds for speed
        curve_meta_get = self.curve_mapping_dict.get
        shift_types_get = self.shift_types_dict.get

        # Init out.
        out: Dict[int, int] = {}

        # Loop over all RiskFactors
        for rf in self.risk_factors_dict.values():
            # RF ID.
            rf_id = int(rf.risk_factor_name_id)

            # Duplicate detection (should not happen if config validated)
            if rf_id in out:
                raise ValueError(
                    "Duplicate risk_factor_name_id detected in risk_factors_dict:\n"
                    f"risk_factor_name_id={rf_id}\n"
                    f"existing_shift_type_id={int(out[rf_id])}\n"
                    f"new_risk_factor_name={rf.risk_factor_name!r}"
                )

            # Get curve meta for risk factor.
            cm = curve_meta_get(rf.curve_name)
            if cm is None:
                raise PrettyKeyError(
                    f"curve_name={rf.curve_name!r} referenced by risk_factor_name="
                    f"{rf.risk_factor_name!r} not found in curve_mapping_dict."
                )

            # Get the shift type name for risk factor.
            st_name = cm.shift_type_name
            st_id = shift_types_get(st_name)
            if st_id is None:
                raise PrettyKeyError(
                    "shift_type_name from curve_mapping not found in shift_types_dict:\n"
                    f"curve_name={rf.curve_name!r}\n"
                    f"shift_type_name={st_name!r}\n"
                    f"risk_factor_name={rf.risk_factor_name!r}\n"
                    f"risk_factor_name_id={rf_id}\n"
                )

            # Store the shift type ID of the risk factor ID.
            out[rf_id] = int(st_id)

        # Set.
        object.__setattr__(self, "_rf_id_to_shift_type_id", out)

        # Return.
        return out

    @property
    def curve_tenor_index(
        self,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, int]]]:
        """
        Cached per-curve tenor lookup used by curve-level proxy resolution.

        This structure supports resolving a required risk factor on curve A
        to the nearest appropriate tenor on a proxy curve B.

        Returns
        -------
        dict:
            curve_name -> (offsets_sorted, rf_ids_sorted, label_to_rf_id)

        offsets_sorted : (Kc,) int32
            Sorted signed offsets for OFFSET-style tenors on the curve
            (e.g. extracted from [Y+2], [M-1]).

        rf_ids_sorted : (Kc,) int32
            Risk factor IDs aligned to offsets_sorted.

        label_to_rf_id : dict[str, int]
            Mapping for LABEL or NONE tenors:
              - LABEL (e.g. [ON], [1W], [1M]) => label -> rf_id
              - NONE (tenor_token is None)    => "" -> rf_id
        """

        # --------------------------------------------------- #
        # --- FAST PATH: RETURN CACHED STRUCTURE IF BUILT --- #
        # --------------------------------------------------- #

        cached = self._curve_tenor_index
        if cached is not None:
            return cached

        # ------------------------------------------------------- #
        # --- TEMPORARY CONTAINERS FOR CURVE TENOR EXTRACTION --- #
        # ------------------------------------------------------- #
        #
        # offsets     : curve_name -> list of signed OFFSET integers
        # offset_ids  : curve_name -> list of rf_ids aligned to offsets
        # labels      : curve_name -> dict(label -> rf_id) for LABEL/NONE tenors
        #

        offsets: Dict[str, list[int]] = {}
        offset_ids: Dict[str, list[int]] = {}
        labels: Dict[str, Dict[str, int]] = {}

        # Local binding for speed (integer-key dict)
        rf_id_to_rf = self.rf_id_to_risk_factor

        # ------------------------------------------------ #
        # --- SINGLE PASS OVER CONFIGURED RISK FACTORS --- #
        # ------------------------------------------------ #
        #
        # For each risk factor:
        #   - group by curve_name
        #   - classify tenor as NONE, LABEL, or OFFSET
        #   - collect data needed for proxy resolution
        #

        for rf_id, rf in rf_id_to_rf.items():
            rf_curve_name = rf.curve_name
            rf_tenor_token = rf.tenor_token

            # --- NONE TENOR (spot-like) --- #
            #
            # No tenor token: treated as a label with empty key "".
            #
            if rf_tenor_token is None:
                labels.setdefault(rf_curve_name, {})[""] = int(rf_id)
                continue

            # Parse bracket token: [UNIT±OFFSET] or [LABEL]
            rf_unit, rf_offset = parse_bracket_token(rf_tenor_token)

            # --- LABEL TENOR (e.g. [ON], [1W], [1M]) --- #
            #
            # No signed offset => exact label match required.
            #
            if rf_offset is None:
                labels.setdefault(rf_curve_name, {})[rf_unit] = int(rf_id)

            # --- OFFSET TENOR (e.g. [Y+2], [M-1], [D+4]) --- #
            #
            # Store signed offset for nearest-neighbor matching.
            #
            else:
                offsets.setdefault(rf_curve_name, []).append(int(rf_offset))
                offset_ids.setdefault(rf_curve_name, []).append(int(rf_id))

        # ------------------------------------------------------- #
        # --- BUILD FINAL PER-CURVE LOOKUP STRUCTURE (OUTPUT) --- #
        # ------------------------------------------------------- #

        # Init dict to store data of property.
        #
        # out: curve_name -> (offsets_sorted, rf_ids_sorted, label_to_rf_id)
        #
        out: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, int]]] = {}

        # --- CURVES WITH OFFSET TENORS --- #
        #
        # Convert lists to NumPy arrays and sort by offset
        # using a stable sort for deterministic behavior.
        #
        for rf_curve_name, rf_offset_list in offsets.items():
            # RF IDs corresponding to offsets.
            rf_ids_list = offset_ids[rf_curve_name]

            # To arrays.
            rf_offsets_arr = np.asarray(rf_offset_list, dtype=np.int32)
            rf_ids_arr = np.asarray(rf_ids_list, dtype=np.int32)

            # Sort the offsets.
            rf_offset_idx = np.argsort(
                rf_offsets_arr, kind="mergesort"
            )  # stable + deterministic

            # Store in dict.
            out[rf_curve_name] = (
                rf_offsets_arr[rf_offset_idx],  # sorted offsets
                rf_ids_arr[rf_offset_idx],  # rf_ids aligned to sorted offsets
                labels.get(rf_curve_name, {}),  # LABEL/NONE lookup (may be empty)
            )

        # --- CURVES WITH ONLY LABEL / NONE TENORS --- #
        #
        # Ensure all curves appear in output, even if they
        # have no OFFSET tenors.
        #
        for rf_curve_name, rf_curve_label_to_rf_id in labels.items():
            if rf_curve_name not in out:
                out[rf_curve_name] = (
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.int32),
                    rf_curve_label_to_rf_id,
                )

        # -------------------------------- #
        # --- CACHE AND RETURN RESULT --- #
        # -------------------------------- #

        # Cache.
        object.__setattr__(self, "_curve_tenor_index", out)

        # Return.
        return out

    def build_risk_factor_name_mapping_for_as_of_date(
        self,
        as_of_date: date,
    ) -> RiskFactorNameAbsToRelMapping:
        """
        Resolve relative risk factor names (risk_factors_dict keys) into absolute names for as_of_date.

        Design:
            - risk_factor_ids defines the authoritative whitelist of valid factors (relative name + id).
            - For the given as-of date, resolve each relative factor to its absolute column name.
            - Fail fast on:
                * missing config rows (unknown curve, missing roll rule)
                * token/kind mismatches (LABEL vs OFFSET)
                * absolute-name collisions (two relatives resolve to same absolute)

        Output:
          abs_to_rel: absolute_name -> relative_name
          abs_to_id:  absolute_name -> risk_factor_name_id
        """

        # --- INIT --- #

        # Init the dicts to store the mappings.
        abs_to_rf_name_and_id: Dict[str, Tuple[str, int]] = {}

        # # cache HY boundary grids per fh_bh_ddmm string
        # hy_cache: Dict[str, list[date]] = {}

        # --- MAIN LOOP --- #

        # Loop over white list of risk factor IDs.
        for rf in self.risk_factors_dict.values():
            # --- EXTRACT RF ID INFORMATION --- #
            rf_rel_name = rf.risk_factor_name
            rf_rel_name_id = rf.risk_factor_name_id
            rf_curve_name = rf.curve_name
            rf_tenor_token = rf.tenor_token

            # --- GET CURVE MAPPING FOR RISK FACTOR ID --- #

            # Get curve.
            rf_cm = self.curve_mapping_dict.get(rf_curve_name)

            # Smoke - need a curve mapping.
            if rf_cm is None:
                raise ValueError(
                    "risk_factor_ids references unknown curve_name.\n"
                    f"curve_name={rf_curve_name!r}\nrelative={rf_rel_name!r}"
                )

            # --- GET ALIAS SCHEMA

            # Get alias schema for RF from RF curve mapping
            rf_alias = self.alias_schema_dict.get(rf_cm.alias_type_name)

            # Smoke - need and alias schema
            if rf_alias is None:
                raise ValueError(
                    "curve_mapping references unknown alias_type_name.\n"
                    f"curve_name={rf_cm.curve_name!r}\nalias_type_name={rf_cm.alias_type_name!r}"
                )

            # ================================ #
            # === RESOLVE TO ABSOLUTE NAME === #
            # ================================ #

            # --- ALIAS KIND == 'NONE' --- #
            elif rf_alias.alias_kind == "NONE":
                # Tenor token must be None or empty.
                if rf_tenor_token is not None:
                    raise ValueError(
                        f"Risk factor ID tenor_token ({rf_tenor_token}) must be None for alias_kind = 'NONE'.\n"
                        f"relative={rf_rel_name!r}\ncurve_name={rf_curve_name!r}"
                    )

                # Absolute name is same as rf relative name.
                rf_abs_name = rf_rel_name

                # Store.
                abs_to_rf_name_and_id[rf_abs_name] = (rf_rel_name, rf_rel_name_id)

                # Continue.
                continue

            # --- ALIAS KIND == 'LABEL'
            elif rf_alias.alias_kind == "LABEL":
                # If the tenor token is missing.
                if rf_tenor_token is None:
                    # For LABEL curves the tenor token must exist (e.g., [ON], [1W])
                    raise ValueError(
                        "Risk factor missing tenor_token for alias_kind = 'LABEL'.\n"
                        f"relative={rf_rel_name!r}\ncurve_name={rf_curve_name!r}"
                    )

                # Parse the tenor token.
                label, signed_offset = parse_bracket_token(rf_tenor_token)

                # Error is signed offset is not none.
                if signed_offset is not None:
                    raise ValueError(
                        f"alias_kind = 'LABEL' tenor_token ('{rf_tenor_token}') must not have +/- offset.\n"
                        f"relative={rf_rel_name!r}\ncurve_name={rf_curve_name!r}"
                    )

                # Create absolute name.
                rf_abs_name = f"{rf_curve_name}_{label}"

                # Store.
                abs_to_rf_name_and_id[rf_abs_name] = (rf_rel_name, rf_rel_name_id)

                # Continue.
                continue

            # --- ALIAS KIND == 'OFFSET'
            elif rf_alias.alias_kind == "OFFSET":
                # Token cannot be None.
                if rf_tenor_token is None:
                    raise ValueError(
                        "alias_kind = OFFSET risk factor missing tenor_token.\n"
                        f"relative={rf_rel_name!r}\ncurve_name={rf_curve_name!r}"
                    )

                # spot_roll_rule_name cannot be None or missing for OFFSET.
                if (
                    rf_cm.spot_roll_rule_name is None
                    or rf_cm.spot_roll_rule_name == "None"
                ):
                    raise ValueError(
                        "alias_kind = OFFSET curve missing spot_roll_rule_name in curve_mapping.\n"
                        f"relative={rf_rel_name!r}\ncurve_name={rf_curve_name!r}"
                    )

                # Parse RF tenor token
                tenor_token_unit, tenor_token_signed_offset = parse_bracket_token(
                    rf_tenor_token
                )

                # Need +/- offset.
                if tenor_token_signed_offset is None:
                    raise ValueError(
                        "alias_kind = OFFSET tenor_token must have +/- integer.\n"
                        f"relative={rf_rel_name!r}\n"
                        f"curve_name={rf_curve_name!r}"
                        f"tenor_token={rf_tenor_token!r}\n"
                    )

                # Get spot roll rule
                rf_srr = self.spot_roll_rules_dict.get(
                    (rf_cm.spot_roll_rule_name, rf_cm.tenor_type_name),
                    None,
                )

                # The RF spot roll rule (SRR) cannot be None for alias_kind = 'OFFSET'.
                if rf_srr is None:
                    raise ValueError(
                        "Missing spot roll rule in spot_roll_rules_dict! \n"
                        "For alias_kind = OFFSET, the spot roll cannot be none! \n"
                        f"spot_roll_rule_name={rf_cm.spot_roll_rule_name!r}\n"
                        f"tenor_type_name={rf_cm.tenor_type_name!r}\n"
                        f"relative={rf_rel_name!r}\n"
                        f"curve_name={rf_curve_name!r}"
                    )

                # Extract.
                rf_spot_roll_method = rf_srr.spot_roll_method
                rf_tenor_type_name = rf_cm.tenor_type_name

                # Normalize optional ints (SpotRollRule stores np.int32 or None)
                rf_offset = int(rf_srr.offset_in_tenor_units or 0)
                rf_roll_month = int(rf_srr.roll_month or 0)
                rf_roll_day = int(rf_srr.roll_day or 0)
                rf_post_shift = int(rf_srr.post_roll_shift or 0)

                # Get the relative name.
                if (
                    rf_tenor_type_name == "TENOR_YM"
                    and rf_spot_roll_method == "CALENDAR_PERIOD"
                ):
                    # M+0 anchored to first day of asof month, shifted by rule offset
                    spot_month_yyyymm01 = add_months_to_date_and_return_yyyymm01(
                        d=date(as_of_date.year, as_of_date.month, 1),
                        months_to_add=rf_offset,
                    )

                    # Get RF absolute date.
                    rf_abs_date_yyyymm01 = add_months_to_date_and_return_yyyymm01(
                        d=spot_month_yyyymm01,
                        months_to_add=int(tenor_token_signed_offset),
                    )

                    # Create absolute name.
                    rf_abs_name = f"{rf_curve_name}_{fmt_date_as_str_yyyymm(rf_abs_date_yyyymm01)}"

                    # Store.
                    abs_to_rf_name_and_id[rf_abs_name] = (rf_rel_name, rf_rel_name_id)

                    # Continue.
                    continue

                # Get the relative name.
                elif (
                    rf_tenor_type_name == "TENOR_Y"
                    and rf_spot_roll_method == "ROLL_ON_MMDD"
                ):
                    # Get the spot year for the as-of-date.
                    spot_year = get_spot_year(
                        as_of_date=as_of_date,
                        roll_month=rf_roll_month,
                        roll_day=rf_roll_day,
                        post_roll_shift=rf_post_shift,
                    )

                    # Compute absolute year from as-of-date.
                    year = spot_year + int(tenor_token_signed_offset)

                    # Create absolute name.
                    rf_abs_name = f"{rf_curve_name}_{year:04d}"

                    # Store.
                    abs_to_rf_name_and_id[rf_abs_name] = (rf_rel_name, rf_rel_name_id)

                    # Continue.
                    continue

                # Get the relative name.
                elif (
                    rf_tenor_type_name == "TENOR_YMD"
                    and rf_spot_roll_method == "HALF_YEAR_BY_DATE"
                ):
                    # Get FH and BH MM-DD dates.
                    fh_bh_ddmm = rf_srr.fh_bh_ddmm

                    # Cannot be None.
                    if fh_bh_ddmm is None:
                        raise ValueError(
                            f"spot_roll_method = 'HALF_YEAR_BY_DATE' missing fh_bh_ddmm!\n"
                            f"spot_roll_rule_name={rf_cm.spot_roll_rule_name!r}\n"
                            f"tenor_type_name={rf_cm.tenor_type_name!r}\n"
                            f"relative={rf_rel_name!r}\n"
                            f"curve_name={rf_curve_name!r}"
                        )

                    # Split FH and BH
                    fh_mmdd, bh_mmdd = fh_bh_ddmm.split("|")

                    # Sanity.
                    if len(fh_mmdd) != 4 or len(bh_mmdd) != 4:
                        raise ValueError(
                            f"fh_bh_ddmm ('{fh_bh_ddmm}) must be of form 'ddmm|ddmm'!"
                        )

                    # Get year of as of date.
                    as_of_year = as_of_date.year

                    # Business rule for HALF_YEAR_BY_DATE:
                    #   - Build a chronological sequence of FH/BH boundary dates around the as-of date
                    #   - Define HY+0 as the first boundary date that is >= as_of_date
                    #   - Apply the signed offset from the token (HY+k moves k boundaries forward/backward)
                    #   - The resulting boundary date becomes the absolute YYYYMMDD suffix

                    # Choose a tight year window so we never allocate more than necessary.
                    # We need enough boundaries to cover the requested +/- offset.
                    look_years = max(3, abs(int(tenor_token_signed_offset)) + 1)

                    # Build a sorted list of boundary dates:
                    #   [ ... , YYYY-<FH>, YYYY-<BH>, (YYYY+1)-<FH>, (YYYY+1)-<BH>, ... ]
                    boundaries: list[date] = []
                    for y in range(
                        as_of_year - look_years, as_of_year + look_years + 1
                    ):
                        boundaries.append(
                            date(year=y, month=int(fh_mmdd[:2]), day=int(fh_mmdd[2:]))
                        )
                        boundaries.append(
                            date(year=y, month=int(bh_mmdd[:2]), day=int(bh_mmdd[2:]))
                        )
                    boundaries.sort()
                    # print("boundaries:", boundaries)

                    # HY+0 anchor: first boundary >= as_of_date
                    idx_hy0 = bisect_left(boundaries, as_of_date)

                    # Apply signed offset from token, e.g. HY+1 => next boundary, HY-2 => two boundaries back
                    idx_target = idx_hy0 + int(tenor_token_signed_offset)

                    # Fail fast if the offset pushes outside our constructed grid
                    if idx_target < 0 or idx_target >= len(boundaries):
                        raise ValueError(
                            "HALF_YEAR_BY_DATE offset out of range for constructed boundary grid.\n"
                            f"curve_name={rf_curve_name!r}\nrelative={rf_rel_name!r}\n"
                            f"tenor_token={rf_tenor_token!r}\nas_of_date={as_of_date}\n"
                            f"fh_bh_ddmm={fh_bh_ddmm!r}\n"
                            f"grid_years=[{as_of_year - look_years}, {as_of_year + look_years}]"
                        )

                    # Absolute delivery date is the chosen boundary
                    abs_dt = boundaries[idx_target]

                    # Convert to YYYYMMDD string suffix for the absolute name
                    yyyymmdd_str = (
                        f"{abs_dt.year:04d}{abs_dt.month:02d}{abs_dt.day:02d}"
                    )

                    # Create absolute name.
                    rf_abs_name = f"{rf_curve_name}_{yyyymmdd_str}"

                    # Store.
                    abs_to_rf_name_and_id[rf_abs_name] = (rf_rel_name, rf_rel_name_id)

                    # Continue.
                    continue

                # Error as not implemented or invalid.
                else:
                    raise ValueError(
                        "Unsupported OFFSET resolution combination: "
                        f"(tenor_type_name, rf_spot_roll_method) = ('{rf_tenor_type_name}', '{rf_spot_roll_method}')\n"
                        f"curve_name={rf_curve_name!r}\n"
                        f"relative={rf_rel_name!r}\n"
                        f"relative_id={rf_rel_name_id!r}\n"
                        f"tenor_token={rf_tenor_token!r}"
                    )

        return RiskFactorNameAbsToRelMapping(
            as_of_date=as_of_date,
            abs_to_rf_name_and_id=abs_to_rf_name_and_id,
        )


@dataclass(frozen=True, slots=True)
class RiskStaticConfigFrames:
    """
    Temporary container that holds pandas DataFrames loaded from Excel.
    Call .compile() to create a runtime RiskStaticConfig that holds no DataFrames.
    """

    curve_mapping: pd.DataFrame
    risk_factors: pd.DataFrame
    tenor_schema: pd.DataFrame
    alias_schema: pd.DataFrame
    spot_roll_rules: pd.DataFrame
    ccy: pd.DataFrame
    risk_factor_groups: pd.DataFrame
    shift_types: pd.DataFrame
    instruments: pd.DataFrame
    risk_desks: pd.DataFrame
    risk_book_structure: pd.DataFrame
    curve_proxies: pd.DataFrame
    risk_factor_proxies: pd.DataFrame
    holiday_calendars: pd.DataFrame

    def __post_init__(self):
        # (1) Curve names must be unique.
        if not self.curve_mapping["curve_name"].is_unique:
            raise ValueError(
                f"self.curve_mapping.curve_name must be unique: "
                f"{self.curve_mapping['curve_name'].unique()}"
            )
        # (2) Curve name IDs must be unique.
        if not self.curve_mapping["curve_name_id"].is_unique:
            raise ValueError(
                f"self.curve_mapping.curve_name_id must be unique: "
                f"{self.curve_mapping['curve_name_id'].unique()}"
            )
        # (3) Risk factor names must be unique.
        if not self.risk_factors["risk_factor_name"].is_unique:
            raise ValueError(
                f"self.risk_factors.risk_factor_name must be unique: "
                f"{self.risk_factors['risk_factor_name'].unique()}"
            )
        # (4) Risk factor name IDs must be unique.
        if not self.risk_factors["risk_factor_name_id"].is_unique:
            raise ValueError(
                f"self.risk_factors.risk_factor_name_id must be unique: "
                f"{self.risk_factors['risk_factor_name_id'].unique()}"
            )

        # Curve proxies.
        if not self.curve_proxies["curve_name"].is_unique:
            raise ValueError(
                f"self.curve_proxies.curve_name must be unique: "
                f"{self.curve_proxies['curve_name'].unique()}"
            )
        if not self.risk_factor_proxies["risk_factor_name"].is_unique:
            raise ValueError(
                f"self.risk_factor_proxies.risk_factor_name must be unique: "
                f"{self.risk_factor_proxies['risk_factor_name'].unique()}"
            )

    def compile(self) -> RiskStaticConfig:
        """
        Compile DataFrames into compact Python structures and return runtime config.
        """

        # --- CURVE MAPPING --- #

        # Init.
        curve_mapping_dict: Dict[str, CurveMapping] = {}

        # Populate.
        for r in self.curve_mapping.to_dict("records"):
            # Is active.
            is_active = r.get("active", 1)

            # Smoke.
            if is_active is None:
                raise ValueError(f"is_active cannot be None!")
            if not isinstance(is_active, (bool, int)):
                raise TypeError(
                    f"is_active must be bool or int, got {type(is_active).__name__}"
                )

            # Continue if not active.
            if not bool(is_active):
                continue

            # Extract.
            curve_name = str(r["curve_name"])
            curve_name_id = int(r["curve_name_id"])
            risk_factor_group = str(r["risk_factor_group"])
            tenor_type_name = str(r["tenor_type_name"])
            spot_roll_rule_name = str(r["spot_roll_rule_name"])
            alias_type_name = str(r["alias_type_name"])
            quote_ccy = str(r["quote_ccy"])
            shift_type_name = str(r["shift_type_name"])

            # Create CurveMapping
            if curve_name not in curve_mapping_dict:
                curve_mapping_dict[curve_name] = CurveMapping(
                    curve_name=curve_name,
                    curve_name_id=curve_name_id,
                    risk_factor_group=risk_factor_group,
                    tenor_type_name=tenor_type_name,
                    spot_roll_rule_name=spot_roll_rule_name,
                    alias_type_name=alias_type_name,
                    quote_ccy=quote_ccy,
                    shift_type_name=shift_type_name,
                )
            else:
                raise ValueError(
                    f"CureMapping already exists for curve_name: '{curve_name}'!"
                )

        # --- RISK FACTOR IDS --- #

        # Init.
        risk_factors_dict: Dict[str, RiskFactor] = {}

        # Populate.
        for r in self.risk_factors.to_dict("records"):
            # Is active.
            is_active = r.get("active", 1)

            # Smoke.
            if is_active is None:
                raise ValueError(f"is_active cannot be None!")
            if not isinstance(is_active, (bool, int)):
                raise TypeError(
                    f"is_active must be bool or int, got {type(is_active).__name__}"
                )

            # Continue if not active.
            if not bool(is_active):
                continue

            # Extract.
            risk_factor_name = str(r["risk_factor_name"])
            risk_factor_name_id = int(r["risk_factor_name_id"])
            tenor_token = None if _is_na(r["tenor_token"]) else str(r["tenor_token"])
            curve_name = str(r["curve_name"])

            # Create CurveMapping
            if risk_factor_name not in risk_factors_dict:
                risk_factors_dict[risk_factor_name] = RiskFactor(
                    risk_factor_name=risk_factor_name,
                    risk_factor_name_id=risk_factor_name_id,
                    tenor_token=tenor_token,
                    curve_name=curve_name,
                )
            else:
                raise ValueError(
                    f"RiskFactor already exists for risk_factor_name: '{risk_factor_name}'!"
                )

        # --- TENOR SCHEMA --- #

        # Init dict.
        tenor_schema_dict: Dict[str, TenorSchema] = {}

        # Loop over records.
        for r in self.tenor_schema.to_dict("records"):
            # Tenor type.
            tenor_type_name = str(r["tenor_type_name"])
            if tenor_type_name in tenor_schema_dict:
                raise ValueError(
                    f"TenorSchema already exists for tenor_type_name: {tenor_type_name!r}"
                )

            # Regex
            regex_raw = r.get("regex")
            if _is_na(regex_raw):
                raise ValueError(
                    f"TenorSchema.regex is missing for tenor_type_name: {tenor_type_name!r}"
                )

            # Create.
            tenor_schema_dict[tenor_type_name] = TenorSchema(
                tenor_type_name=tenor_type_name,
                tenor_units=str(r["tenor_units"]),
                digits=int(r["digits"]),
                regex=re.compile(str(regex_raw)),
            )

        # --- ALIAS SCHEMA --- #

        # Init dict.
        alias_schema_dict: Dict[str, AliasSchema] = {}

        # Loop over records.
        for r in self.alias_schema.to_dict("records"):
            # Alias type name
            alias_type_name = str(r["alias_type_name"])
            if alias_type_name in alias_schema_dict:
                raise ValueError(
                    f"AliasSchema already exists for alias_type_name: {alias_type_name!r}"
                )

            # Alias unit.
            unit = r.get("alias_unit")

            # Create.
            alias_schema_dict[alias_type_name] = AliasSchema(
                alias_type_name=alias_type_name,
                alias_kind=str(r["alias_kind"]),
                alias_unit=None if _is_na(unit) else str(unit),
            )

        # --- SPOT ROLL RULES --- #

        # NOTE: dict key should be (spot_roll_rule_name, tenor_type_name) to avoid collisions across tenor types.
        spot_roll_rules_dict: Dict[tuple[str, str], SpotRollRule] = {}

        # Loop over records.
        for r in self.spot_roll_rules.to_dict("records"):
            spot_roll_rule_name = str(r["spot_roll_rule_name"])
            tenor_type_name = str(r["tenor_type_name"])
            key = (spot_roll_rule_name, tenor_type_name)

            if key in spot_roll_rules_dict:
                raise ValueError(
                    "SpotRollRule already exists for key:\n"
                    f"  spot_roll_rule_name={spot_roll_rule_name!r}\n"
                    f"  tenor_type_name={tenor_type_name!r}"
                )

            # Extract other.
            srm = r.get("spot_roll_method")
            off = r.get("offset_in_tenor_units")
            rmm = r.get("roll_month")
            rdd = r.get("roll_day")
            fbh = r.get("fh_bh_ddmm")
            prs = r.get("post_roll_shift")

            # Validate

            # Create.
            spot_roll_rules_dict[key] = SpotRollRule(
                spot_roll_rule_name=spot_roll_rule_name,
                tenor_type_name=tenor_type_name,
                spot_roll_method="" if _is_na(srm) else str(srm),
                offset_in_tenor_units=None if _is_na(off) else int(off),
                roll_month=None if _is_na(rmm) else int(rmm),
                roll_day=None if _is_na(rdd) else int(rdd),
                fh_bh_ddmm=None if _is_na(fbh) else str(fbh),
                post_roll_shift=None if _is_na(prs) else int(prs),
            )

        # =========== #
        # --- CCY --- #
        # =========== #

        ccy_dict: Dict[str, int] = _build_str_to_int_dict(
            df=self.ccy,
            key_col="ccy",
            id_col="ccy_id",
            sheet_name="ccy",
            key_transform=str.upper,  # currencies should be uppercase
            require_unique_keys=True,
        )

        # ========================== #
        # --- RISK FACTOR GROUPS --- #
        # ========================== #

        risk_factor_groups_dict: Dict[str, int] = _build_str_to_int_dict(
            df=self.risk_factor_groups,
            key_col="risk_factor_group",
            id_col="risk_factor_group_id",
            sheet_name="risk_factor_groups",
            key_transform=str,  # keep as-is (or use str.upper if desired)
            require_unique_keys=True,
        )

        # ================== #
        # --- SHIFT TYPE --- #
        # ================== #

        shift_types_dict: Dict[str, int] = _build_str_to_int_dict(
            df=self.shift_types,
            key_col="shift_type",
            id_col="shift_type_id",
            sheet_name="shift_types",
            key_transform=str.upper,  # your config uses LOG_RETURN style
            require_unique_keys=True,
        )

        # Validate shift types.
        validate_shift_types_dict(shift_types_dict)

        # =================== #
        # --- INSTRUMENTS --- #
        # =================== #

        instruments_dict: Dict[str, int] = _build_str_to_int_dict(
            df=self.instruments,
            key_col="instrument",
            id_col="instrument_id",
            sheet_name="instruments",
            key_transform=str.upper,  # your code calls instrument_name.upper()
            require_unique_keys=True,
        )

        # --- CURVE PROXIES DICT --- #

        # Init.
        curve_proxies_dict: Dict[str, CurveProxyMap] = {}

        # Loop.
        for r in self.curve_proxies.to_dict("records"):
            # Is active.
            is_active = r.get("active", 1)

            # Smoke.
            if is_active is None:
                raise ValueError(f"is_active cannot be None!")
            if not isinstance(is_active, (bool, int)):
                raise TypeError(
                    f"is_active must be bool or int, got {type(is_active).__name__}"
                )

            # Continue if not active.
            if not bool(is_active):
                continue

            # Curve name.
            curve_name = str(r["curve_name"])

            # Duplicate detection (explicit, fail-fast)
            if curve_name in curve_proxies_dict:
                raise ValueError(
                    f"CurveProxyMap already exists for curve_name: '{curve_name}'!"
                )

            # Vol scaling factor: default only on missing/NaN (do NOT override 0.0)
            scaling_factor = r["proxy_curve_vol_scaling_factor"]

            # Set to 1.0 if None or NaN.
            if scaling_factor is None or (
                isinstance(scaling_factor, float) and np.isnan(scaling_factor)
            ):
                scaling_factor = 1.0

            # Cannot be < 0.
            if scaling_factor < 0:
                raise ValueError("proxy_curve_vol_scaling_factor must be positive!")

            # Create and store.
            curve_proxies_dict[curve_name] = CurveProxyMap(
                curve_name=curve_name,
                proxy_curve_name=str(r["proxy_curve_name"]),
                proxy_curve_vol_scaling_factor=float(scaling_factor),
            )

        # --- RISK FACTOR PROXIES DICT --- #

        # Init.
        risk_factor_proxies_dict: Dict[str, RiskFactorProxyMap] = {}

        # Loop.
        for r in self.risk_factor_proxies.to_dict("records"):
            # Is active.
            is_active = r.get("active", 1)

            # Smoke.
            if is_active is None:
                raise ValueError(f"is_active cannot be None!")
            if not isinstance(is_active, (bool, int)):
                raise TypeError(
                    f"is_active must be bool or int, got {type(is_active).__name__}"
                )

            # Continue if not active.
            if not bool(is_active):
                continue

            # Curve name.
            risk_factor_name = str(r["risk_factor_name"])

            # Duplicate detection (explicit, fail-fast)
            if risk_factor_name in risk_factor_proxies_dict:
                raise ValueError(
                    f"RiskFactorProxyMap already exists for risk_factor_name: '{risk_factor_name}'!"
                )

            # Vol scaling factor: default only on missing/NaN (do NOT override 0.0)
            scaling_factor = r["proxy_risk_factor_vol_scaling_factor"]

            # Set to 1.0 if None or NaN.
            if scaling_factor is None or (
                isinstance(scaling_factor, float) and np.isnan(scaling_factor)
            ):
                scaling_factor = 1.0

            # Create and store.
            risk_factor_proxies_dict[risk_factor_name] = RiskFactorProxyMap(
                risk_factor_name=risk_factor_name,
                proxy_risk_factor_name=str(r["proxy_risk_factor_name"]),
                proxy_risk_factor_vol_scaling_factor=float(scaling_factor),
            )

        # --- FINISH --- #

        # Create and return
        return RiskStaticConfig(
            curve_mapping_dict=curve_mapping_dict,
            risk_factors_dict=risk_factors_dict,
            tenor_schema_dict=tenor_schema_dict,
            alias_schema_dict=alias_schema_dict,
            spot_roll_rules_dict=spot_roll_rules_dict,
            ccy_dict=ccy_dict,
            risk_factor_groups_dict=risk_factor_groups_dict,
            shift_types_dict=shift_types_dict,
            instruments_dict=instruments_dict,
            curve_proxies_dict=curve_proxies_dict,
            risk_factor_proxies_dict=risk_factor_proxies_dict,
            # TODO: OTHERS
        )
