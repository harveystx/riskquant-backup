# src/riskquant/market_data/curves/label_maturity.py
"""
LABEL tenor token -> maturity ordinal for constructed discount curves (D080).

This is the last place a string exists on the discount-curve path. Everything
downstream of `resolve_label_maturity_ords` is float64 coordinates and int32
ordinals; the kernel never learns what `[3M]`, `MOD_FOLLOWING` or `US` mean.

D080 vocabulary
---------------
Exactly `ON, 1W, 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y`. `XAxisKind.LABEL` also admits
`SN`, `TN` and `nD`, for which D080 defines no maturity. Those are REJECTED
here, never inferred: guessing a settlement convention would be inventing
economics (D080 clause: absence of semantics is resolved by rejection).

Date rule (D080)
----------------
    spot = business_day_add(as_of, spot_lag_days, calendar)
    ON   = next_business_day(as_of, calendar)          # bypasses spot, never T+3
    1W   = adjust(spot + 7 calendar days)
    nM   = adjust(add_months(spot, n))
    nY   = adjust(add_years(spot, n))
    adjust = the curve's business-day convention, with the end-of-month rule:
             if spot is its month's LAST BUSINESS day, the maturity is the target
             month's last business day.

The result depends only on `(as_of, calendar, spot_lag, convention, eom, label)`.
It is independent of rates, shocks and scenarios, so maturities are built once
per run, not per scenario.
"""

from __future__ import annotations

from riskquant.config.static_config.schema_objects import CurveMeta
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.utils.business_days import (
    add_months_ord,
    add_years_ord,
    adjust_ord,
    business_day_add_ord,
    is_last_business_day_of_month_ord,
    last_business_day_of_month_ord,
    next_business_day_ord,
    ord_to_date,
)
from riskquant.utils.tenors import parse_bracket_token

#: The only labels D080 gives a maturity. Anything else fails closed.
D080_SUPPORTED_LABELS: frozenset[str] = frozenset({
    "ON",
    "1W",
    "1M",
    "3M",
    "6M",
    "1Y",
    "2Y",
    "5Y",
    "10Y",
})


def resolve_label_maturity_ords(
    *,
    curve: CurveMeta,
    tenor_tokens: tuple[str, ...],
    as_of_ord32: int,
    static_config: RiskStaticConfig,
) -> tuple[int, ...]:
    """
    Resolve one curve's LABEL pillars to maturity ordinals.

    Parameters
    ----------
    curve
        Must declare `maturity_calendar`, `spot_lag_days`,
        `business_day_convention` and `end_of_month`; a missing field is a
        configuration error, never a default.
    tenor_tokens
        Bracketed tokens as stored on `CurvePointMeta.tenor_token`, e.g. `"[3M]"`.

    Returns
    -------
    tuple[int, ...]
        Maturity ordinals in the same order as `tenor_tokens`. Canonical
        ordering by maturity is the caller's separate step.
    """
    calendar_name, spot_lag, convention, eom = _require_construction_metadata(curve)
    calendar = static_config.holiday_calendars_dict.get(calendar_name)
    if calendar is None:
        raise ValueError(
            f"curve {curve.curve_name!r}: maturity_calendar {calendar_name!r} is not "
            f"a configured holiday calendar "
            f"{sorted(static_config.holiday_calendars_dict)}."
        )
    holidays = calendar.holiday_dates_ord

    spot_ord = business_day_add_ord(
        date_ord=int(as_of_ord32), n_days=spot_lag, holidays_ord=holidays
    )
    spot_is_month_end = eom and is_last_business_day_of_month_ord(
        date_ord=spot_ord, holidays_ord=holidays
    )

    out: list[int] = []
    for token in tenor_tokens:
        label = _require_supported_label(curve_name=curve.curve_name, token=token)
        if label == "ON":
            # ON is an as-of-to-next-business-day interval: it does NOT start at
            # spot, so a T+2 curve must not report a T+3 overnight pillar.
            out.append(
                next_business_day_ord(date_ord=int(as_of_ord32), holidays_ord=holidays)
            )
            continue
        unit = label[-1]
        count = int(label[:-1])
        if unit == "W":
            unadjusted = spot_ord + 7 * count
        elif unit == "M":
            unadjusted = add_months_ord(date_ord=spot_ord, n_months=count)
        else:  # "Y" — the vocabulary gate above admits nothing else
            unadjusted = add_years_ord(date_ord=spot_ord, n_years=count)

        if spot_is_month_end and unit in ("M", "Y"):
            target = ord_to_date(unadjusted)
            out.append(
                last_business_day_of_month_ord(
                    year=target.year, month=target.month, holidays_ord=holidays
                )
            )
        else:
            out.append(
                adjust_ord(
                    date_ord=unadjusted, convention=convention, holidays_ord=holidays
                )
            )
    return tuple(out)


def _require_supported_label(*, curve_name: str, token: str) -> str:
    if not (token.startswith("[") and token.endswith("]")):
        raise ValueError(
            f"curve {curve_name!r}: malformed tenor token {token!r}; a LABEL axis "
            "stores bracketed tokens such as '[3M]' (D080)."
        )
    label, numeric = parse_bracket_token(token)
    if numeric is not None:
        raise ValueError(
            f"curve {curve_name!r}: tenor token {token!r} is numeric; a constructed "
            "discount curve requires a LABEL token (D080)."
        )
    if label not in D080_SUPPORTED_LABELS:
        raise ValueError(
            f"curve {curve_name!r}: tenor label {label!r} has no D080 maturity "
            f"semantics. Supported: {sorted(D080_SUPPORTED_LABELS)}. Labels such as "
            "SN, TN and nD are admitted by the generic LABEL axis but are rejected "
            "here rather than inferred."
        )
    return label


def _require_construction_metadata(curve: CurveMeta) -> tuple[str, int, str, bool]:
    """Every field is mandatory for construction; absence fails, never defaults."""
    missing = [
        name
        for name, value in (
            ("maturity_calendar", curve.maturity_calendar),
            ("spot_lag_days", curve.spot_lag_days),
            ("business_day_convention", curve.business_day_convention),
            ("end_of_month", curve.end_of_month),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"curve {curve.curve_name!r}: maturity construction requires "
            f"{sorted(missing)} on curve_meta (D080)."
        )
    return (
        str(curve.maturity_calendar),
        int(curve.spot_lag_days),  # type: ignore[arg-type]
        str(curve.business_day_convention),
        bool(curve.end_of_month),
    )
