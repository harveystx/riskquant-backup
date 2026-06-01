# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(
    width="full",
    app_title="Market Risk Ingest Pipeline",
    layout_file="layouts/ingest_pipeline_market_risk.slides.json",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RiskQuant: Ingest Pipeline for Market Risk Instruments
    """)
    return


@app.cell
def _():
    import warnings
    from pathlib import Path

    import marimo as mo

    # Filter out the openpyxl data validation warning specifically
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
    return Path, mo


@app.cell
def _():
    from riskquant.utils.printer import print_dict, print_df

    return (print_dict,)


@app.cell
def _():
    from riskquant.utils.dates import get_previous_business_date_as_date

    return (get_previous_business_date_as_date,)


@app.cell
def _():
    from riskquant.config.static_config.compile import (
        compile_risk_static_config_from_xlsx,
    )

    return (compile_risk_static_config_from_xlsx,)


@app.cell
def _():
    from riskquant.ingest_fixtures.portfolio.exhaustive import (
        CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA,
    )

    return (CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA,)


@app.cell
def _(CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA, print_dict):
    print_dict(CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA)
    return


@app.cell
def _():

    from riskquant.portfolio.ingest.canonicalize import canonicalize_raw_ingest_df
    from riskquant.portfolio.ingest.materialize import materialize_canonicalized_ingest_df
    from riskquant.portfolio.ingest.validate import validate_materialized_ingest_df

    return (
        canonicalize_raw_ingest_df,
        materialize_canonicalized_ingest_df,
        validate_materialized_ingest_df,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## As-of Date
    """)
    return


@app.cell
def _(get_previous_business_date_as_date):
    # Set an As-Of Date
    as_of_date = get_previous_business_date_as_date()

    # Print.
    print(f"As-of date: {as_of_date}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Configuration Data - Risk Static Data
    """)
    return


@app.cell
def _(Path, compile_risk_static_config_from_xlsx):
    # Import Risk Static Config from XLSX
    risk_static_config = compile_risk_static_config_from_xlsx(
        path=Path(
            "C:/Users/mharvey/Documents/code/riskquant/data/dev",
            "dev_risk_static_data_config_v01.xlsx",
        )
    )
    return (risk_static_config,)


@app.cell
def _(risk_static_config):
    risk_static_config
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Positions Data
    """)
    return


app._unparsable_cell(
    r"""
    raw_ingest_df: pd.DataFrame =
    """,
    name="_"
)


@app.cell
def _(ingest_df):
    ingest_df.head(n=30)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Canonicalize
    """)
    return


@app.cell
def _(canonicalize_raw_ingest_df, ingest_df):
    canonicalize_raw_ingest_df(df=ingest_df)
    ingest_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Materialize
    """)
    return


@app.cell
def _(ingest_df, materialize_canonicalized_ingest_df):
    materialize_canonicalized_ingest_df(df=ingest_df)
    ingest_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validate
    """)
    return


@app.cell
def _(ingest_df, validate_materialized_ingest_df):
    validate_materialized_ingest_df(df=ingest_df)
    ingest_df
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
