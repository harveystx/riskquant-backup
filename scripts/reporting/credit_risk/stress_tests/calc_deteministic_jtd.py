# scripts/credit_risk/stress_tests/calc_deterministic_jtd.py
"""
This script calculates a deterministic worst-case JTD loss for a given set of exposures.
"""

# TODO: Mapping tables still required to map from raw positions data to the required exposure data for JTD calculations, specifically:
#  - COUNTERPARTY -> CREDIT RATING -> PD
#    Maps each counterparty to a credit rating (e.g. using an internal ratings map or external ratings data) and
#    then maps the credit rating to a PD (e.g. using a standard ratings-to-PD mapping table).
#  - COUNTERPARTY -> LGD
#    Maps each counterparty to a loss given default (LGD) value (e.g. using internal LGD estimates or external data).
#  - NETTING SETS
#    Maps each counterparty to a netting set (e.g. using internal netting set definitions or external data).
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import sqlalchemy
from dotenv import load_dotenv
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import text

from riskquant.ingest_fixtures.jtd.deterministic_jtd.minimal_deterministic_jtd import (
    CANONICAL_DETERMINISTIC_JTD_INGEST_DATA,
)

_DWH_DB_NAME_FOR_EOD_POSITIONS = "stxdwh"
_DWH_TABLE_NAME_FOR_EOD_POSITIONS = "eod.positions_eod_history"


_SQL_MAX_RETRIES: int = 5
_SQL_RETRY_DELAY: float = 2  # seconds
_SQL_CHUNK_SIZE: int | None = 10_000


_COLUMN_TO_IMPORT: tuple[str, ...] = (
    "client_name",
    "client_id",
    "trade_id",
    "position_id",
    "risk_mid_eur",
)


def load_env() -> None:
    """
    Load environment variables from .env into os.environ.
    """
    load_dotenv()


def get_as_of_date(business_day_offset: int = 0) -> str:
    """
    Returns the as-of date string in YYYY-MM-DD format based on the current date and a business day offset.
    (In this example, we will return a hardcoded date for simplicity.)
    """
    # In a real implementation, this function would calculate the as-of date based on the current date and business day offset.
    # For this example, we will return a hardcoded date.
    today = pd.Timestamp.date(pd.Timestamp.today())
    as_of_date = today - pd.tseries.offsets.BDay(business_day_offset + 1)
    as_of_date = as_of_date.strftime("%Y-%m-%d")
    return as_of_date


class DeterministicJTDConfig:
    NUM_EXPOSURES: int = int(
        os.getenv(
            "CREDIT_RISK_DETERMINISTIC_JTD_NUM_EXPOSURES",
            5,
        )
    )
    SUB_IG_PD_THRESHOLD: float = float(
        os.getenv(
            "CREDIT_RISK_DETERMINISTIC_JTD_SUB_IG_PD_THRESHOLD",
            0.005,
        )
    )
    DATA_SOURCE: str = os.getenv(
        "CREDIT_RISK_DETERMINISTIC_JTD_DATA_SOURCE",
        "canonical",
    )
    DB_HOST: str = os.getenv(
        "DWH_HOST",
        "dwh-prod-replica.stxaws.net",
    )
    DB_PORT: int = int(
        os.getenv(
            "DWH_PORT",
            5432,
        )
    )
    DB_USER: str = os.getenv(
        "DWH_USER",
        "UNKNOWN_USER",
    )
    DB_PASSWORD: str = os.getenv(
        "DWH_PASSWORD",
        "UNKNOWN_PASSWORD",
    )


def load_canonical_test_data() -> pd.DataFrame:
    """
    Loads the canonical test data for deterministic JTD calculations.
    """
    return pd.DataFrame(CANONICAL_DETERMINISTIC_JTD_INGEST_DATA)


def create_sql_query_for_positions_data_for_as_of_date(as_of_date: str) -> str:
    """
    Creates a SQL query to retrieve positions data for a given as-of date.
    (In this example, we will use the canonical test data instead of an actual SQL query.)
    """
    columns_str = ", ".join(_COLUMN_TO_IMPORT)
    # sql_str = f"SELECT {columns_str} FROM {_DWH_TABLE_NAME_FOR_EOD_POSITIONS}"
    # sql_str += " WHERE"
    # sql_str += f" (dwh_snapshot_ts_utc = '{as_of_date}')"
    # sql_str += " AND (risk_mid_eur IS NOT NULL)"
    # sql_str += " AND ((recognition_num is NULL) OR (recognition_num = 1))"
    # sql_str += " AND (dwh_current_flg = 1)"
    # sql_str += " AND (realisation_date IS NULL)"
    sql_str = f"""
        SELECT {columns_str}
        FROM {_DWH_TABLE_NAME_FOR_EOD_POSITIONS}
        WHERE dwh_snapshot_ts_utc = '{as_of_date}'
            AND risk_mid_eur IS NOT NULL
            AND ((recognition_num IS NULL) OR (recognition_num = 1))
            AND dwh_current_flg = 1
            AND realisation_date IS NULL
        """
    return sql_str


def init_sql_alchemy_engine(
    username: str,
    password: str,
    host: str,
    port: int,
    db_name: str,
):
    """
    Create an SQLAlchemy engine.

    Parameters:
    - username: str, the database username
    - password: str, the database password
    - host: str, the database host
    - port: int, the database port
    - dbname: str, the name of the database

    Returns:
    - engine: SQLAlchemy engine instance
    """
    connection_string = (
        f"postgresql+psycopg://{username}:{password}@{host}:{port}/{db_name}"
    )

    # Create and return the engine.
    return sqlalchemy.create_engine(
        connection_string,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )


def init_engine_for_data_warehouse(db_name: str):
    # Connect to the database.
    db_engine = init_sql_alchemy_engine(
        username=DeterministicJTDConfig.DB_USER,
        password=DeterministicJTDConfig.DB_PASSWORD,
        host=DeterministicJTDConfig.DB_HOST,
        port=DeterministicJTDConfig.DB_PORT,
        db_name=db_name,
    )

    # Print.
    print(
        f"NOTE! An SQL Alchemy Engine has been initialised for the '{db_name}' database in Data Warehouse:\n".upper()
    )
    print(f"- Host:     {DeterministicJTDConfig.DB_HOST}")
    print(f"- Port:     {DeterministicJTDConfig.DB_PORT}")
    print(f"- Database: {db_name}")

    # Returns the connection to the specified database.
    return db_engine


def execute_sql_with_retry(
    *,
    engine: Engine,
    sql_query: str,
    max_retries: int = _SQL_MAX_RETRIES,
    retry_delay: float = _SQL_RETRY_DELAY,
    chunksize: int | None = _SQL_CHUNK_SIZE,
) -> pd.DataFrame:
    def _run_query() -> pd.DataFrame:
        if chunksize:
            chunks = list(
                pd.read_sql(
                    sql=text(sql_query),
                    con=engine,
                    chunksize=chunksize,
                )
            )

            if not chunks:
                return pd.DataFrame()

            return pd.concat(chunks, ignore_index=True)
        return pd.read_sql(sql=text(sql_query), con=engine)

    def _is_retryable(e: Exception) -> bool:
        msg = str(e).lower()
        return any(
            keyword in msg
            for keyword in (
                "serialization",
                "deadlock",
                "could not serialize",
                "connection reset",
                "connection aborted",
                "timeout",
            )
        )

    last_exception: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            df = _run_query()
            return df

        except OperationalError as e:
            last_exception = e

            if attempt < max_retries and _is_retryable(e):
                wait_time = retry_delay * (2 ** (attempt - 1))
                print(
                    f"Retryable SQL error (attempt {attempt}/{max_retries}). "
                    f"Retrying in {wait_time:.1f}s...\n"
                )
                time.sleep(wait_time)
                continue

            break

        except Exception as e:
            # Non-operational errors → do not retry
            last_exception = e
            break

    if last_exception is not None:
        raise last_exception

    raise RuntimeError("SQL execution failed but no exception was captured")


def import_data_from_sql_query(
    *,
    sql_query: str,
    max_retries: int = _SQL_MAX_RETRIES,
    retry_delay: float = _SQL_RETRY_DELAY,
    chunksize: int | None = _SQL_CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Executes the given SQL query and returns the results as a DataFrame.
    (In this example, we will use the canonical test data instead of executing a SQL query.)
    """
    # Heading
    heading = "ATTEMPING TO IMPORT DATA FROM DB WITH SQL QUERY"
    print("\n" + "=" * len(heading))
    print(heading)
    print("=" * len(heading) + "\n")

    # In a real implementation, this function would execute the SQL query against a database and return the results as a DataFrame.
    # For this example, we will return the canonical test data instead.
    # Connect to DB.
    db_engine = init_engine_for_data_warehouse(db_name=_DWH_DB_NAME_FOR_EOD_POSITIONS)

    # Download the internal_ppts_df into a pandas internal_ppts_df.
    # eod_positions_df = pd.read_sql(sql=sql_query_str, con=db_engine)
    eod_positions_df = execute_sql_with_retry(
        engine=db_engine,
        sql_query=sql_query,
        max_retries=max_retries,
        retry_delay=retry_delay,
        chunksize=chunksize,
    )

    # Print summary of results.
    n_pos = len(eod_positions_df)
    print()
    print(f"SQL query executed successfully. Retrieved {n_pos:,} rows.")
    print()
    print(f"Columns retrieved: {', '.join(eod_positions_df.columns)}")
    print()
    print(f"Sample data:\n\n{eod_positions_df.head().to_string(index=False)}")

    heading = f"IMPORT COMPLETED! IMPORTED {n_pos:,.0f} ROWS SUCCESSFULLY!"
    print("\n" + "=" * len(heading))
    print(heading)
    print("=" * len(heading) + "\n")

    # Return the resulting DataFrame.
    return eod_positions_df


def transform_raw_positions_data_to_exposures_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps raw DWH positions into exposure schema required for JTD model.
    """
    df = df.copy()

    df["client_name"] = (
        df["client_name"]
        .fillna("UNKNOWN_CLIENT")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "_", regex=True)
    )

    uid = (
        df["client_name"]
        + "_"
        + df["trade_id"].astype(str)
        + "_"
        + df["position_id"].astype(str)
    )

    # EAD is postion unrealised PnL (risk_mid_eur) which can be negative → set negative EADs to 0
    # This is the so-called replacement cost approach to calculating EAD for trading book positions, which is a common approach in stress testing and JTD calculations.
    rc_ead = np.maximum(df["risk_mid_eur"].to_numpy(), 0.0)

    exposures_df = pd.DataFrame(
        {
            "uid": uid,
            "ead": rc_ead,
            "lgd": 0.6,  # placeholder → must be sourced properly
            "pd": 0.01,  # placeholder → must be sourced properly
        }
    )

    return exposures_df


def load_and_transform_positions_data(
    *,
    as_of_date: str | None = None,
    data_source: str = DeterministicJTDConfig.DATA_SOURCE,
) -> pd.DataFrame:
    """
    Loads position data based on configured data source.
    """
    source = data_source

    if source == "canonical":
        return load_canonical_test_data()

    elif source == "sql":
        if as_of_date is None:
            raise ValueError("as_of_date is required for SQL data source")

        sql_query = create_sql_query_for_positions_data_for_as_of_date(as_of_date)
        df = import_data_from_sql_query(sql_query=sql_query)
        exposures_df = transform_raw_positions_data_to_exposures_data(df)
        return exposures_df

    else:
        raise ValueError(f"Unknown data source: {source}")


def validate_exposures_data(df: pd.DataFrame) -> None:
    """
    Validates that the input DataFrame has the expected columns and that they have consistent lengths.
    """
    expected_columns = {"uid", "ead", "lgd", "pd"}
    actual_columns = set(df.columns)

    if not expected_columns.issubset(actual_columns):
        missing_cols = expected_columns - actual_columns
        raise ValueError(f"Input DataFrame is missing expected columns: {missing_cols}")


def calculate_aggregate_jtd_loss_for_num_exposures(
    *,
    df: pd.DataFrame,
    num_exposures: int,
    sub_ig_threshold: float,
) -> pd.DataFrame:
    """
    Calculates the deterministic worst-case JTD loss for the top N exposures.
    Filters for Sub-Investment Grade using a PD threshold.
    """
    sub_ig_mask = df["pd"] > sub_ig_threshold
    stress_df = df[sub_ig_mask].copy()

    if stress_df.empty:
        return pd.DataFrame()

    # Deterministic JTD (assumes 100% default)
    stress_df["jtd_loss"] = stress_df["ead"] * stress_df["lgd"]

    top_n_df = stress_df.sort_values(by="jtd_loss", ascending=False).head(num_exposures)

    return top_n_df


def main(
    *,
    num_exposures: int | None = None,
    sub_ig_threshold: float | None = None,
    business_day_offset: int = 0,
    data_source: str | None = None,
) -> None:
    # 1. Load environment
    load_env()

    # 2. Load config (after env is loaded)
    num_exposures = num_exposures or DeterministicJTDConfig.NUM_EXPOSURES
    sub_ig_threshold = sub_ig_threshold or DeterministicJTDConfig.SUB_IG_PD_THRESHOLD
    data_source = data_source or DeterministicJTDConfig.DATA_SOURCE

    # 2.5 As-of date (for data retrieval, if needed)
    as_of_date = get_as_of_date(business_day_offset=business_day_offset)

    # 3. Load data and validate
    df = load_and_transform_positions_data(
        as_of_date=as_of_date,
        data_source=data_source,
    )

    # 3.5 Validate the exposures data to ensure it has the required structure and columns
    validate_exposures_data(df)

    # 4. Run stress
    top_exposures = calculate_aggregate_jtd_loss_for_num_exposures(
        df=df,
        num_exposures=num_exposures,
        sub_ig_threshold=sub_ig_threshold,
    )

    total_stress_loss = top_exposures["jtd_loss"].sum()

    # 5. Output
    print("=" * 65)
    print(f" STRESS TEST: TOP {num_exposures} SUB-IG DEFAULTS ".center(65, "="))
    print("=" * 65)

    print("Run Configuration:")
    print(f"  NUM_EXPOSURES = {num_exposures}")
    print(f"  SUB_IG_PD_THRESHOLD = {sub_ig_threshold}\n")

    print(f"Total Worst-Case Concentration Loss: {total_stress_loss:,.2f}\n")

    print("Driver Breakdown:")
    display_df = top_exposures[["uid", "ead", "lgd", "pd", "jtd_loss"]].copy()

    display_df["ead"] = display_df["ead"].map("{:,.0f}".format)
    display_df["jtd_loss"] = display_df["jtd_loss"].map("{:,.0f}".format)

    print(display_df.to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main(data_source="sql")
