# src/riskquant/credit/netting_set/loader.py
"""
Netting-set master loader entry points.

`loader.py` is the external-source bridge — CSVs, the DWH, S3, whatever
concrete production format is wired in. Each loader reads the raw source
into a canonical `pd.DataFrame` with columns `uid`, `obligor_uid`,
`net_collateral`, then hands off to `ingest.compile_netting_set_master_from_df`
for dtype coercion, validation and SoA materialization.

This module is intentionally a thin shell: it owns I/O only, never
validation logic. Validation lives in `validate.py`; SoA materialization
lives in `ingest.py`. Adding a loader is one function that returns a
DataFrame and one call site that hands the frame to the ingest.
"""

from __future__ import annotations

# TODO: implement concrete loaders once production sources are concrete.
# Examples:
#   def load_netting_set_master_from_csv(path: str | pathlib.Path) -> pd.DataFrame: ...
#   def load_netting_set_master_from_dwh(*, as_of_date: str, engine) -> pd.DataFrame: ...
