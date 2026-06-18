import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Portfolio Ingestation: Vanilla IRS
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    return mo, pd


@app.cell
def _():
    from riskquant.utils import printer

    return (printer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import Canonical RiskQuant Ingest Fixtures
    """)
    return


@app.cell
def _():
    from riskquant.ingest_fixtures.as_of import CANONICAL_AS_OF_DATE
    from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS
    from riskquant.ingest_fixtures.portfolio.vanilla_irs import CANONICAL_VANILLA_IRS_MARKET_RISK_PORTFOLIO_DATA

    return (
        CANONICAL_AS_OF_DATE,
        CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
        CANONICAL_VANILLA_IRS_MARKET_RISK_PORTFOLIO_DATA,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Canonical As-Of
    """)
    return


@app.cell
def _(CANONICAL_AS_OF_DATE, printer):
    printer.print_text(f"Canonical As-Of Date: {CANONICAL_AS_OF_DATE}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Canonical Risk Static Config
    """)
    return


@app.cell
def _(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS, printer):
    printer.print_dict_of_dfs(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS, title="Canonical Risk Static Config")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Canonical Vanilla IRS Ingest Portfolio Data
    """)
    return


@app.cell
def _(CANONICAL_VANILLA_IRS_MARKET_RISK_PORTFOLIO_DATA, pd):
    ingest_df = pd.DataFrame(CANONICAL_VANILLA_IRS_MARKET_RISK_PORTFOLIO_DATA)
    return (ingest_df,)


@app.cell
def _(ingest_df, printer):
    printer.print_df(ingest_df, title="Canonical Vanilla IRS Ingest Data")
    return


@app.cell
def _():
    return


@app.cell
def _():
    ## Load int 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compile the SoA from the DataFrame
    """)
    return


@app.cell
def _():
    from riskquant.portfolio.ingest.compile import compile_ingest_soa_payload_from_df

    return (compile_ingest_soa_payload_from_df,)


@app.cell
def _(compile_ingest_soa_payload_from_df, ingest_df):
    compile_ingest_soa_payload_from_df(
        df=ingest_df,
    
    )
    return


@app.cell
def _(CANONICAL_RISK_STATIC_CONFIG_DATA):
    CANONICAL_RISK_STATIC_CONFIG_DATA
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
