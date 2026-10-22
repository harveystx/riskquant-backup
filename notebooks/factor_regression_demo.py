# notebooks/factor_regression_demo.py
"""
Factor-regression library demo over the canonical fixture.

Run: ``marimo edit notebooks/factor_regression_demo.py`` (interactive) or
``marimo run notebooks/factor_regression_demo.py`` (app view).

The notebook is a terminal: it reads the canonical fixture, calls the public
facade ``riskquant.solutions.factor_regression.api`` and shows the results.
It holds no estimator. Every number comes from the facade. The notebook
computes only display quantities (FX-converted levels and spreads for plots).
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="riskquant factor regression")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from riskquant.ingest_fixtures.factor_regression import analyses as fr_analyses
    from riskquant.ingest_fixtures.factor_regression import catalog as fr_catalog
    from riskquant.ingest_fixtures.factor_regression.observed import (
        aligned_levels_for_case,
    )
    from riskquant.ingest_fixtures.factor_regression.proxy_calibration import (
        population_loadings,
    )
    from riskquant.solutions.factor_regression import api

    return (
        aligned_levels_for_case,
        api,
        fr_analyses,
        fr_catalog,
        mo,
        np,
        pl,
        plt,
        population_loadings,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Factor regression in riskquant

    The library answers one kind of question: **how does one market series
    move with others?** It has four public functions in
    `riskquant.solutions.factor_regression.api`:

    | Function | What it does |
    |---|---|
    | `align_levels` | Keeps the dates common to all series in a window. Records every dropped observation. |
    | `fit_ols` | Builds return or level rows, checks rank, fits OLS, reports diagnostics and digests. |
    | `wald_test` | Tests a joint linear hypothesis `R @ coef = r` on one fit. |
    | `unit_root_test` | ADF test of a declared linear combination of levels (cointegration with a declared vector). |

    The library reports statistics. It does not decide. A diagnostic never
    stops a fit. A real fault (wrong domain, rank loss, exact identity) raises
    a named error.

    The data come from the canonical fixture
    `riskquant.ingest_fixtures.factor_regression`: five cases, each one a
    business question that the risk engine asks.
    """)
    return


@app.cell
def _(aligned_levels_for_case, fr_catalog):
    cases = fr_catalog.CANONICAL_FACTOR_REGRESSION_CASES
    levels_by_case = {
        name: aligned_levels_for_case(case) for name, case in cases.items()
    }
    return cases, levels_by_case


@app.cell
def _(api, np, pl):
    def coef_table(result: api.OlsResult) -> pl.DataFrame:
        """One row per design column: estimate, standard error, t, p."""
        return pl.DataFrame({
            "column": list(result.column_names),
            "coef": result.coef,
            "se": result.se,
            "t": result.t_stats,
            "p": result.p_values,
        })

    def fit_summary(label: str, result: api.OlsResult) -> dict[str, object]:
        """Scalar facts of one fit, for side-by-side tables."""
        return {
            "fit": label,
            "rows": result.n_obs,
            "step": result.step,
            "stride": result.stride,
            "overlap_q": result.overlap_order,
            "covariance": result.covariance.kind.name,
            "r_squared": result.r_squared,
            "durbin_watson": result.durbin_watson,
            "kappa": result.condition_number,
            "diagnostics": ", ".join(d.code for d in result.diagnostics) or "-",
        }

    def unit_root_row(label: str, result: api.UnitRootResult) -> dict[str, object]:
        """Statistic against the MacKinnon (2010) N = 1 critical values."""
        cv = result.critical_values
        return {
            "test": label,
            "deterministic": result.deterministic.name,
            "lag": result.lag,
            "rows": result.n_obs,
            "tau": result.statistic,
            "cv_1%": float(cv[0]),
            "cv_5%": float(cv[1]),
            "cv_10%": float(cv[2]),
            "tau_below_10%_cv": bool(result.statistic < cv[2]),
        }

    def dates_of(levels: api.AlignedLevels) -> np.ndarray:
        """Aligned int32 ordinals as numpy datetime64 for plot axes only."""
        return levels.date_ord.astype("datetime64[D]")

    def level_of(levels: api.AlignedLevels, name: str) -> np.ndarray:
        """One aligned level row by series name."""
        return levels.levels[levels.names.index(name)]

    def term(series: str, transform: api.Transform) -> api.Term:
        return api.Term(series=series, transform=transform)

    def try_call(fn, *args):
        """Return (result, None) or (None, facade error). Other errors raise."""
        facade_errors = (
            api.AlignmentError,
            api.DegenerateFitError,
            api.InsufficientObservationsError,
            api.ModelSpecError,
            api.RankDeficiencyError,
            api.RestrictionError,
            api.ScaleDomainError,
            api.TransformDomainError,
        )
        try:
            return fn(*args), None
        except facade_errors as error:
            return None, error

    return (
        coef_table,
        dates_of,
        fit_summary,
        level_of,
        term,
        try_call,
        unit_root_row,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The data and the alignment

    Each case is a Polars long frame: one row per series and observed date.
    A missing observation is an absent row. The series trade on different
    calendars (TARGET, NYSE, FX, SIX), so `align_levels` keeps only common
    dates and records each dropped observation with a reason:

    * `OUTSIDE_WINDOW`: the date is outside `[start, end)`.
    * `NOT_COMMON_DATE`: another series has no observation on that date.

    Returns are built **after** alignment. All terms of one row cover the same
    interval, also across a holiday gap.
    """)
    return


@app.cell
def _(cases, mo):
    case_picker = mo.ui.dropdown(
        options=list(cases), value=next(iter(cases)), label="Case"
    )
    mo.hstack([case_picker], justify="start")
    return (case_picker,)


@app.cell
def _(cases, levels_by_case, case_picker, dates_of, mo, np, pl, plt):
    _case = cases[case_picker.value]
    _levels = levels_by_case[case_picker.value]
    _summary = _case.frame.group_by("series_name", "unit", maintain_order=True).agg(
        pl.len().alias("observations"),
        pl.col("date").min().alias("first"),
        pl.col("date").max().alias("last"),
    )
    _exclusions = (
        pl
        .DataFrame(
            {
                "series_name": [e.series for e in _levels.exclusions],
                "reason": [e.reason for e in _levels.exclusions],
            },
            schema={"series_name": pl.Utf8, "reason": pl.Utf8},
        )
        .group_by("series_name", "reason")
        .len()
        .sort("series_name", "reason")
    )
    _fig, _ax = plt.subplots(figsize=(9, 3.2))
    for _j, _name in enumerate(_levels.names):
        _row = _levels.levels[_j]
        # Rebase positive series to 100 for one shared axis; rates stay raw.
        _y = 100.0 * _row / _row[0] if bool(np.all(_row > 0.0)) else _row
        _ax.plot(dates_of(_levels), _y, lw=0.9, label=_name)
    _ax.set_title("Aligned levels (positive series rebased to 100)")
    _ax.legend(fontsize=7, ncol=3)
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"**Question.** {_case.question}"),
        mo.md(
            f"Window `[{_case.window_start}, {_case.window_end})`, "
            f"aligned dates: **{_levels.date_ord.shape[0]}**."
        ),
        mo.hstack([_summary, _exclusions], widths=[3, 2]),
        _fig,
        mo.md("**Generating parameters (truth)**"),
        pl.DataFrame({
            "parameter": list(_case.truth),
            "value": list(_case.truth.values()),
        }),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Is a EUR asset tied to a USD asset converted to EUR?

    `XLIST_EUR` and `XLIST_USD` are two listings of one claim. The test
    object is the log combination

    $$z_t = \ln XLIST\_EUR_t - \ln XLIST\_USD_t - \ln USDEUR_t .$$

    The `LOG` scale converts USD to EUR inside the combination: no
    pre-converted series is necessary. The vector $(1, -1, -1)$ is
    **declared**, not estimated, so the N = 1 MacKinnon table applies.
    `PEER_EUR` shares 70 % of the return driver but has no long-run tie.
    """)
    return


@app.cell
def _(mo, levels_by_case, dates_of, level_of, np, plt):
    _lv = levels_by_case["fx_converted_cointegration"]
    _d = dates_of(_lv)
    _usd_in_eur = level_of(_lv, "XLIST_USD") * level_of(_lv, "USDEUR")
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(10, 3.2))
    _a.plot(_d, level_of(_lv, "XLIST_EUR"), lw=0.9, label="XLIST_EUR")
    _a.plot(_d, _usd_in_eur, lw=0.9, label="XLIST_USD x USDEUR")
    _a.plot(_d, level_of(_lv, "PEER_EUR"), lw=0.9, label="PEER_EUR")
    _a.set_title("Levels in EUR")
    _a.legend(fontsize=7)
    _b.plot(_d, np.log(level_of(_lv, "XLIST_EUR") / _usd_in_eur), lw=0.8, label="XLIST")
    _b.plot(_d, np.log(level_of(_lv, "PEER_EUR") / _usd_in_eur), lw=0.8, label="PEER")
    _b.set_title("Log spread to the converted USD asset")
    _b.legend(fontsize=7)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell
def _(mo):
    fx_lag = mo.ui.slider(0, 10, value=1, label="ADF lag p", show_value=True)
    fx_det = mo.ui.radio(
        options=["CONSTANT", "CONSTANT_TREND"], value="CONSTANT", label="Deterministic"
    )
    mo.hstack([fx_lag, fx_det], justify="start")
    return fx_det, fx_lag


@app.cell
def _(levels_by_case, api, fx_det, fx_lag, fr_analyses, pl, unit_root_row):
    def _spec(first: str) -> api.UnitRootSpec:
        return api.UnitRootSpec(
            combination=api.LinearCombination(
                terms=(
                    api.CombinationTerm(series=first, weight=1.0),
                    api.CombinationTerm(series="XLIST_USD", weight=-1.0),
                    api.CombinationTerm(series="USDEUR", weight=-1.0),
                ),
                scale=api.Scale.LOG,
            ),
            deterministic=api.Deterministic[fx_det.value],
            lag=fx_lag.value,
            rank_rtol=fr_analyses.RANK_RTOL,
        )

    _lv = levels_by_case["fx_converted_cointegration"]
    pl.DataFrame([
        unit_root_row(
            f"{_first} - XLIST_USD - USDEUR", api.unit_root_test(_lv, _spec(_first))
        )
        for _first in ("XLIST_EUR", "PEER_EUR")
    ])
    return


@app.cell
def _(levels_by_case, api, coef_table, fit_summary, fr_analyses, mo, np, pl, try_call):
    _lv = levels_by_case["fx_converted_cointegration"]
    _level_fit = fr_analyses.run_ols_analysis(
        fr_analyses.ols_analysis("fx_level_peer"), _lv
    )
    _ret_fit = fr_analyses.run_ols_analysis(
        fr_analyses.ols_analysis("fx_return_conversion"), _lv
    )
    _wald = api.wald_test(
        _ret_fit,
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([1.0, 1.0]),
    )
    _, _tri_error = try_call(
        fr_analyses.run_ols_analysis,
        fr_analyses.ols_analysis("fx_triangle_identity"),
        _lv,
    )
    mo.vstack([
        mo.md("**Level regression `XLIST_EUR ~ PEER_EUR` (the trap).**"),
        pl.DataFrame([fit_summary("levels on peer", _level_fit)]),
        mo.callout(
            mo.md(
                "R² is high and Durbin-Watson is near 0. The diagnostics "
                "flag the fit; the unit-root test above shows why."
            ),
            kind="warn",
        ),
        mo.md("**Daily log returns `XLIST_EUR ~ XLIST_USD + USDEUR` (HC0).**"),
        coef_table(_ret_fit),
        mo.md(
            f"Joint Wald H0: both slopes = 1. "
            f"W = {_wald.w:.3f}, chi2 p = {_wald.chi2_pvalue:.3f}, "
            f"F p = {_wald.f_pvalue:.3f}."
        ),
        mo.md("**Exact cross `EURGBP ~ USDGBP + USDEUR`.**"),
        mo.callout(
            mo.md(f"`{type(_tri_error).__name__}`: {_tri_error}"),
            kind="danger",
        ),
        mo.md(
            "EURGBP is `USDGBP / USDEUR` on every date. The regression has "
            "no error term, so the library refuses to report standard "
            "errors. This is the Proxy V1 cross identity: author it, do "
            "not estimate it."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Calibrate Proxy V1 loadings for a short-history factor

    `SMALLCAP_EQ_EU` listed on 2024-06-03 and misses about 6 % of its trading
    days. The engine needs a Proxy V1 row (D088), without intercept:

    $$X_{target} = \beta_{EU}\,Y(EQ\_INDEX\_EU,\ LOG,\ h) + \beta_{US}\,Y(EQ\_INDEX\_US,\ LOG,\ h).$$

    The US index closes one session after the target. A one-day return of the
    US index covers a shifted window, so the daily US loading is too low.
    The shift is one session out of $2h$, so the bias falls like $1/h$. It
    does not vanish: the population US loading is still 0.28 at $h = 20$.
    `population_loadings` in the fixture gives the exact biased loadings at
    every $h$ (closed form, no simulation). Overlapping $h$-day rows have
    MA($h-1$) errors: use HAC with lag $\ge h - 1$, or set `stride = step`.

    A longer $h$ is not free. The count of independent $h$-day intervals is
    about $R / h$, so the $\pm 2$ se band widens with $h$. A longer $h$ trades
    bias for a wider band. The plot shows the population curve (what the
    regression measures on average), the fitted loadings of this sample path
    with their $\pm 2$ se band, and the long-run truth.

    On this sample path the fitted US loading lies about 0.2 above the
    population curve for $h = 2$ to $20$ with overlapping rows. That is the
    sampling deviation of one path, not a property of the process. With the
    HAC band the truth stays inside $\pm 2$ se from $h = 2$.

    Truth: $\beta_{EU} = 0.90$, $\beta_{US} = 0.30$.
    """)
    return


@app.cell
def _(mo):
    proxy_step = mo.ui.slider(1, 20, value=10, label="step h (days)", show_value=True)
    proxy_cov = mo.ui.radio(
        options=["IID", "HAC lag h-1", "non-overlapping (stride = h)"],
        value="HAC lag h-1",
        label="Rows and covariance",
    )
    mo.hstack([proxy_step, proxy_cov], justify="start")
    return proxy_cov, proxy_step


@app.cell
def _(api, fr_analyses, term):
    def proxy_spec(step: int, mode: str) -> api.ModelSpec:
        """The Proxy V1 regression at horizon ``step`` in one of three modes."""
        if mode == "IID":
            stride, cov = 1, api.CovarianceSpec(kind=api.CovarianceKind.IID, lag=None)
        elif mode == "HAC lag h-1":
            stride = 1
            cov = api.CovarianceSpec(kind=api.CovarianceKind.HAC_BARTLETT, lag=step - 1)
        else:
            stride, cov = (
                step,
                api.CovarianceSpec(kind=api.CovarianceKind.IID, lag=None),
            )
        log = api.Transform.LOG_RETURN
        return api.ModelSpec(
            response=term("SMALLCAP_EQ_EU", log),
            regressors=(term("EQ_INDEX_EU", log), term("EQ_INDEX_US", log)),
            intercept=False,
            step=step,
            stride=stride,
            covariance=cov,
            rank_rtol=fr_analyses.RANK_RTOL,
            condition_threshold=None,
        )

    return (proxy_spec,)


@app.cell
def _(
    mo,
    cases,
    levels_by_case,
    api,
    np,
    plt,
    population_loadings,
    proxy_cov,
    proxy_spec,
    proxy_step,
):
    _lv = levels_by_case["proxy_calibration"]
    _truth = cases["proxy_calibration"].truth
    _steps = np.arange(1, 21, dtype=np.int32)
    _population = population_loadings(_steps)
    _fits = [api.fit_ols(_lv, proxy_spec(int(_h), proxy_cov.value)) for _h in _steps]
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.2), sharex=True)
    for _k, (_ax, _name) in enumerate(
        zip(_axes, ("EQ_INDEX_EU", "EQ_INDEX_US"), strict=True)
    ):
        _b = np.array([f.coef[_k] for f in _fits])
        _s = np.array([f.se[_k] for f in _fits])
        _ax.fill_between(_steps, _b - 2 * _s, _b + 2 * _s, alpha=0.25, label="+/- 2 se")
        _ax.plot(_steps, _b, marker="o", ms=3, label="estimate (this path)")
        _ax.plot(
            _steps, _population[:, _k], color="C3", lw=1.2, label="population beta(h)"
        )
        _ax.axhline(_truth[f"beta_{_name}"], color="k", lw=0.8, ls="--", label="truth")
        _ax.axvline(proxy_step.value, color="grey", lw=0.6)
        _ax.set_title(f"beta {_name}  ({proxy_cov.value})")
        _ax.set_xlabel("step h")
        _ax.legend(fontsize=7)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell
def _(
    cases,
    levels_by_case,
    api,
    coef_table,
    fit_summary,
    mo,
    np,
    pl,
    proxy_cov,
    proxy_spec,
    proxy_step,
):
    _lv = levels_by_case["proxy_calibration"]
    _truth = cases["proxy_calibration"].truth
    proxy_fit = api.fit_ols(_lv, proxy_spec(proxy_step.value, proxy_cov.value))
    _wald = api.wald_test(
        proxy_fit,
        np.eye(2),
        np.array([_truth["beta_EQ_INDEX_EU"], _truth["beta_EQ_INDEX_US"]]),
    )
    mo.vstack([
        pl.DataFrame([fit_summary("proxy", proxy_fit)]),
        coef_table(proxy_fit),
        mo.md(
            f"Joint Wald H0: loadings = truth (0.90, 0.30). "
            f"chi2 p = **{_wald.chi2_pvalue:.4f}**. With IID on overlapping "
            "rows the standard errors are too small and a true H0 is "
            "rejected; the diagnostic `OVERLAP_NOT_CORRECTED` says so."
        ),
    ])
    return (proxy_fit,)


@app.cell
def _(mo, pl, proxy_fit):
    # The authored-term frame shape of the canonical proxy model fixture
    # (RiskProxyModelTermData). The target and drivers are fixture names, not
    # static-config nodes: this shows the hand-off, it does not compile.
    _rows = pl.DataFrame({
        "term_kind": ["NODE", "NODE"],
        "target_name": ["SMALLCAP_EQ_EU", "SMALLCAP_EQ_EU"],
        "driver_name": list(proxy_fit.column_names),
        "driver_chart": ["LOG_RETURN", "LOG_RETURN"],
        "loading": proxy_fit.coef.tolist(),
    })
    mo.vstack([
        mo.md(
            "**Proxy V1 authored terms from this fit.** Provenance: "
            f"`spec_digest={proxy_fit.spec_digest[:16]}...`, "
            f"`sample_digest={proxy_fit.sample_digest[:16]}...`, "
            f"`{proxy_fit.engine_version}`."
        ),
        _rows,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Factor loadings for the credit VaR copula

    The multi-factor Gaussian copula needs, per obligor, the loadings on the
    systematic factors and the R² (the copula correlation). Equity log
    returns are the asset-return proxy. Rows are weekly and non-overlapping
    (`step = stride = 5`). A volatility regime (August to October 2024, volatility times 2.5) makes
    the residual variance move with the regressors, so IID and HC0 standard
    errors differ.

    The window starts on 2023-07-03, half a year after the first
    observation: see the `OUTSIDE_WINDOW` exclusions in section 1.
    """)
    return


@app.cell
def _(cases, levels_by_case, api, fr_analyses, np, pl, term):
    _lv = levels_by_case["credit_factor_loadings"]
    _truth = cases["credit_factor_loadings"].truth
    _log = api.Transform.LOG_RETURN
    _rows = []
    for _obligor in ("OBLIGOR_EU_BANK", "OBLIGOR_EU_UTILITY", "OBLIGOR_US_INDUSTRIAL"):
        for _kind in (api.CovarianceKind.IID, api.CovarianceKind.HC0):
            _fit = api.fit_ols(
                _lv,
                api.ModelSpec(
                    response=term(_obligor, _log),
                    regressors=(
                        term("CREDIT_FACTOR_GLOBAL", _log),
                        term("CREDIT_FACTOR_EUROPE", _log),
                    ),
                    intercept=True,
                    step=5,
                    stride=5,
                    covariance=api.CovarianceSpec(kind=_kind, lag=None),
                    rank_rtol=fr_analyses.RANK_RTOL,
                    condition_threshold=None,
                ),
            )
            _no_europe = api.wald_test(
                _fit, np.array([[0.0, 0.0, 1.0]]), np.array([0.0])
            )
            _rows.append({
                "obligor": _obligor,
                "covariance": _kind.name,
                "beta_global": float(_fit.coef[1]),
                "true_beta_global": _truth[f"beta_CREDIT_FACTOR_GLOBAL_{_obligor}"],
                "beta_europe": float(_fit.coef[2]),
                "true_beta_europe": _truth[f"beta_CREDIT_FACTOR_EUROPE_{_obligor}"],
                "se_europe": float(_fit.se[2]),
                "r_squared": _fit.r_squared,
                "true_r_squared": _truth[f"r_squared_{_obligor}"],
                "p(europe = 0)": _no_europe.chi2_pvalue,
            })
    pl.DataFrame(_rows)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Read the table as follows. The slopes are close to the generating slopes.
    The US industrial has no European loading, and the test does not reject
    that. R² is a noisy estimate of the copula correlation on 126 weekly
    rows: about 13 of them fall in the stress regime and carry most of the
    variance.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Two drivers that carry one signal

    German power on two gas hubs (TTF, THE) whose returns correlate about
    0.998, and on TTF quoted per MWh and per therm. The condition number
    $\kappa$ is computed on the column-scaled design. The declared threshold
    is 30 (Belsley, Kuh and Welsch).
    """)
    return


@app.cell
def _(levels_by_case, api, coef_table, fit_summary, fr_analyses, mo, np, pl, try_call):
    _lv = levels_by_case["collinear_drivers"]
    _one = fr_analyses.run_ols_analysis(fr_analyses.ols_analysis("gas_single_hub"), _lv)
    _two = fr_analyses.run_ols_analysis(fr_analyses.ols_analysis("gas_two_hubs"), _lv)
    _sum = api.wald_test(_two, np.array([[0.0, 1.0, 1.0]]), np.array([0.85]))
    _, _units_error = try_call(
        fr_analyses.run_ols_analysis,
        fr_analyses.ols_analysis("gas_one_price_two_units"),
        _lv,
    )
    mo.vstack([
        pl.DataFrame([fit_summary("TTF", _one), fit_summary("TTF + THE", _two)]),
        mo.hstack([coef_table(_one), coef_table(_two)]),
        mo.md(
            "The standard error of each hub slope grows about 17 times. "
            f"The **sum** of the slopes stays well measured: Wald H0 "
            f"sum = 0.85, p = {_sum.chi2_pvalue:.3f}."
        ),
        mo.callout(
            mo.md(
                f"TTF per MWh + TTF per therm: `{type(_units_error).__name__}`: {_units_error}"
            ),
            kind="danger",
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Rates that cross zero: spreads and trends

    CHF two-year rates fall through zero. A log return is undefined, so rates
    use `DIFFERENCE` rows and `LEVEL`-scale combinations. Two spreads:

    * swap minus government: stationary around a constant;
    * covered minus swap: stationary around a trend of 25 bp per year.

    Only the matching deterministic case detects the second one.
    """)
    return


@app.cell
def _(mo, levels_by_case, dates_of, level_of, plt):
    _lv = levels_by_case["rates_spread"]
    _d = dates_of(_lv)
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(10, 3.2))
    for _name in _lv.names:
        _a.plot(_d, level_of(_lv, _name), lw=0.8, label=_name)
    _a.axhline(0.0, color="k", lw=0.6)
    _a.set_title("Levels (percent)")
    _a.legend(fontsize=7)
    _b.plot(
        _d,
        level_of(_lv, "CHF_SWAP_2Y") - level_of(_lv, "CHF_GOVT_2Y"),
        lw=0.8,
        label="swap - govt",
    )
    _b.plot(
        _d,
        level_of(_lv, "CHF_COVERED_2Y") - level_of(_lv, "CHF_SWAP_2Y"),
        lw=0.8,
        label="covered - swap",
    )
    _b.set_title("Spreads (percent)")
    _b.legend(fontsize=7)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell
def _(levels_by_case, api, fr_analyses, mo, pl, try_call, unit_root_row):
    _lv = levels_by_case["rates_spread"]
    _pairs = {
        "swap level": (("CHF_SWAP_2Y", 1.0),),
        "swap - govt": (("CHF_SWAP_2Y", 1.0), ("CHF_GOVT_2Y", -1.0)),
        "covered - swap": (("CHF_COVERED_2Y", 1.0), ("CHF_SWAP_2Y", -1.0)),
    }
    _rows = []
    for _label, _terms in _pairs.items():
        for _det in api.Deterministic:
            _spec = api.UnitRootSpec(
                combination=api.LinearCombination(
                    terms=tuple(
                        api.CombinationTerm(series=s, weight=w) for s, w in _terms
                    ),
                    scale=api.Scale.LEVEL,
                ),
                deterministic=_det,
                lag=1,
                rank_rtol=fr_analyses.RANK_RTOL,
            )
            _rows.append(unit_root_row(_label, api.unit_root_test(_lv, _spec)))
    _, _log_error = try_call(
        fr_analyses.run_unit_root_analysis,
        next(
            a
            for a in fr_analyses.CANONICAL_UNIT_ROOT_ANALYSES
            if a.name == "rates_swap_spread_log_scale"
        ),
        _lv,
    )
    _, _ret_error = try_call(
        fr_analyses.run_ols_analysis,
        fr_analyses.ols_analysis("rates_log_return_through_zero"),
        _lv,
    )
    mo.vstack([
        pl.DataFrame(_rows),
        mo.callout(
            mo.md(f"LOG-scale spread: `{type(_log_error).__name__}`: {_log_error}"),
            kind="danger",
        ),
        mo.callout(
            mo.md(f"LOG_RETURN rows: `{type(_ret_error).__name__}`: {_ret_error}"),
            kind="danger",
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Canonical catalog self-check

    The fixture records the outcome of every canonical analysis. This table
    runs them all through the facade and compares. The fixture tests assert
    the same table.
    """)
    return


@app.cell
def _(levels_by_case, fr_analyses, pl, try_call):
    _rows = []
    _fits = {}
    for _a in fr_analyses.CANONICAL_OLS_ANALYSES:
        _r, _e = try_call(fr_analyses.run_ols_analysis, _a, levels_by_case[_a.case])
        _fits[_a.name] = _r
        _expected = (
            _a.expected_error.__name__
            if _a.expected_error
            else ", ".join(sorted(_a.expected_diagnostics)) or "-"
        )
        _observed = (
            type(_e).__name__
            if _e
            else ", ".join(sorted(d.code for d in _r.diagnostics)) or "-"
        )
        _rows.append({
            "kind": "OLS",
            "name": _a.name,
            "expected": _expected,
            "observed": _observed,
        })
    for _h in fr_analyses.CANONICAL_WALD_HYPOTHESES:
        _w = fr_analyses.run_wald_hypothesis(_h, _fits[_h.analysis])
        _rel = "<" if _h.expect_pvalue_below else ">="
        _obs = "<" if _w.chi2_pvalue < _h.pvalue_level else ">="
        _rows.append({
            "kind": "WALD",
            "name": _h.name,
            "expected": f"p {_rel} {_h.pvalue_level}",
            "observed": f"p {_obs} {_h.pvalue_level} ({_w.chi2_pvalue:.4f})",
        })
    for _u in fr_analyses.CANONICAL_UNIT_ROOT_ANALYSES:
        _r, _e = try_call(
            fr_analyses.run_unit_root_analysis, _u, levels_by_case[_u.case]
        )
        _expected = (
            _u.expected_error.__name__
            if _u.expected_error
            else f"tau below cv at {_u.expected_tau_below_level}"
        )
        _observed = (
            type(_e).__name__
            if _e
            else f"tau below cv at {fr_analyses.tau_below_level(_r)} ({_r.statistic:.3f})"
        )
        _rows.append({
            "kind": "ADF",
            "name": _u.name,
            "expected": _expected,
            "observed": _observed,
        })
    _table = pl.DataFrame(_rows)
    _table.with_columns(
        pl
        .col("observed")
        .str.split(" (")
        .list.first()
        .eq(pl.col("expected"))
        .alias("match")
    )
    return


if __name__ == "__main__":
    app.run()
