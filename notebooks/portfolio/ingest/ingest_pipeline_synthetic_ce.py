import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full", app_title="Ingest Pipeline")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RiskQuant: Ingest Pipeline for Synthetic Current Exposure (CE) Instruments
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell
def _():
    from riskquant.portfolio.ingest.schema import validate_schema
    from riskquant.portfolio.ingest.canonicalize import canonicalize_raw_ingest_df
    from riskquant.portfolio.ingest.materialize import materialize_canonicalized_ingest_df
    from riskquant.portfolio.ingest.validate import validate_materialized_ingest_df

    return (
        canonicalize_raw_ingest_df,
        materialize_canonicalized_ingest_df,
        validate_materialized_ingest_df,
        validate_schema,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test Data
    """)
    return


@app.cell
def _(pd):
    data = {
        "uid": [1_000, 1_005, 1_010],
        "instrument": [
            "synthetic_ce_t_copula",
            "SYNTHETIC_CE_T_COPULA",
            "SYNTHETIC_CE_T_COPULA",
        ],
        "ead": [5_000_000.0, 12_000_000.0, 8_500_000.0],
        "pd": [0.02, 0.05, 0.01],
        "lgd": [0.60, 0.40, 0.50],
        "beta": [0.25, 0.35, 0.15],
        "systemic_factor": ["SYS_FACTOR", "SYS_FACTOR", "SYS_FACTOR"],
        "idiosyncratic_factor": ["IDIO_1", "IDIO_2", "IDIO_3"],
        "chi_squared_factor": ["CHI_FACTOR", "CHI_FACTOR", "CHI_FACTOR"],
    }
    raw_ingest_df = pd.DataFrame(data)
    raw_ingest_df
    return (raw_ingest_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Schema
    """)
    return


@app.cell
def _(raw_ingest_df, validate_schema):
    validate_schema(df=raw_ingest_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Canonicalize
    """)
    return


@app.cell
def _(canonicalize_raw_ingest_df, pd, raw_ingest_df):
    canonicalized_ingest_df: pd.DataFrame = canonicalize_raw_ingest_df(df=raw_ingest_df)
    canonicalized_ingest_df
    return (canonicalized_ingest_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Materialize
    """)
    return


@app.cell
def _(
    canonicalized_ingest_df: "pd.DataFrame",
    materialize_canonicalized_ingest_df,
    pd,
):
    materialized_ingest_df: pd.DataFrame = materialize_canonicalized_ingest_df(df=canonicalized_ingest_df)
    materialized_ingest_df
    return (materialized_ingest_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validate
    """)
    return


@app.cell
def _(materialized_ingest_df: "pd.DataFrame", validate_materialized_ingest_df):
    validate_materialized_ingest_df(df=materialized_ingest_df)
    materialized_ingest_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
