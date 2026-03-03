# riskquant/simlab/resolve/proxy_audit.py
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

from riskquant.config.deprecated_configs import RiskStaticConfig
from riskquant.exceptions import PrettyKeyError
from riskquant.rq_logging import LogContext, get_logger, with_context
from riskquant.simlab.resolve.proxy import (
    ProxyResolution,
    RES_NOT_SIMULATED,
    RES_CURVE_PROXY,
    RES_DIRECT,
    RES_ID_PROXY,
    RES_TENOR_FALLBACK,
)

# TYPE ALIAS
LoggerLike = Union[logging.Logger, logging.LoggerAdapter]


# --- LABELS FOR AUDIT OUTPUT --- #
_SOURCE_CODE_TO_LABEL: Dict[int, str] = {
    int(RES_NOT_SIMULATED): "NOT_SIMULATED",
    int(RES_DIRECT): "DIRECT",
    int(RES_TENOR_FALLBACK): "TENOR_FALLBACK",
    int(RES_ID_PROXY): "ID_PROXY",
    int(RES_CURVE_PROXY): "CURVE_PROXY",
}


def _smoke_test_proxy_resolution(proxy_resolution: ProxyResolution) -> None:
    """
    Enforce dtypes expected by downstream vectorized resolution code.
    """
    if proxy_resolution.required_factor_ids.dtype != np.int32:
        raise TypeError("proxy_resolution.required_factor_ids must be int32")
    if proxy_resolution.source_factor_ids.dtype != np.int32:
        raise TypeError("proxy_resolution.source_factor_ids must be int32")
    if proxy_resolution.resolution_code.dtype != np.int8:
        raise TypeError("proxy_resolution.resolution_code must be int8")
    if proxy_resolution.shock_scale.dtype != np.float32:
        raise TypeError("proxy_resolution.shock_scale must be float32")


def _build_audit_df(
    *,
    cfg: RiskStaticConfig,
    proxy_resolution: ProxyResolution,
) -> pd.DataFrame:
    """
    Internal implementation. Intentionally materializes Python strings / DataFrame.
    Audit-only; not kernel-friendly.
    """
    _smoke_test_proxy_resolution(proxy_resolution)

    rf_by_id = cfg.rf_id_to_risk_factor
    num_required_factors = int(proxy_resolution.required_factor_ids.shape[0])

    # Avoid copies where possible.
    req_id = proxy_resolution.required_factor_ids.astype(np.int32, copy=False)
    src_id = proxy_resolution.source_factor_ids.astype(np.int32, copy=False)
    code = proxy_resolution.resolution_code

    required_curve = [""] * num_required_factors
    required_tok = [""] * num_required_factors
    required_name = [""] * num_required_factors

    source_curve = [""] * num_required_factors
    source_tok = [""] * num_required_factors
    source_name = [""] * num_required_factors

    source_code = [""] * num_required_factors

    id_proxy_target = [""] * num_required_factors
    curve_proxy_target = [""] * num_required_factors

    for i in range(num_required_factors):
        rid = int(req_id[i])
        sid = int(src_id[i])

        rrf = rf_by_id.get(rid)
        srf = rf_by_id.get(sid)

        if rrf is None:
            raise PrettyKeyError(
                f"required_factor_id not found in config: {rid}"
            )
        if srf is None:
            raise PrettyKeyError(f"source_factor_id not found in config: {sid}")

        required_curve[i] = rrf.container_name
        required_tok[i] = "" if rrf.tenor_token is None else rrf.tenor_token
        required_name[i] = rrf.risk_factor_name

        source_curve[i] = srf.container_name
        source_tok[i] = "" if srf.tenor_token is None else srf.tenor_token
        source_name[i] = srf.risk_factor_name

        c = int(code[i])
        source_code[i] = _SOURCE_CODE_TO_LABEL.get(c, f"UNKNOWN_{c}")

        if c == int(RES_ID_PROXY):
            # ID proxy resolves directly to the proxy rf (sid), so source_name is the target rf name.
            id_proxy_target[i] = source_name[i]
        elif c == int(RES_CURVE_PROXY):
            # Curve proxy: show the proxy curve name if present.
            cp = cfg.curve_proxies_dict.get(required_curve[i])
            curve_proxy_target[i] = "" if cp is None else cp.proxy_curve_name

    return pd.DataFrame(
        {
            "required_factor_id": req_id,
            "required_risk_factor_name": required_name,
            "required_curve_name": required_curve,
            "required_tenor_token": required_tok,
            "source_factor_id": src_id,
            "source_risk_factor_name": source_name,
            "source_curve_name": source_curve,
            "source_tenor_token": source_tok,
            "source_code": source_code,
            "shock_scale": proxy_resolution.shock_scale,
            "id_proxy_target_risk_factor_name": id_proxy_target,
            "curve_proxy_target_curve_name": curve_proxy_target,
        }
    )


def _jsonable(v: Any) -> Any:
    """
    Convert non-JSON-native scalars to JSON-friendly Python types.
    """
    # numpy scalar -> python scalar
    if isinstance(v, np.generic):
        return v.item()
    # handle NaN -> None (safer for downstream ingestion)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


@dataclass(slots=True)
class ProxyResolutionAudit:
    """
    Handle for producing and logging an audit of a ProxyResolution.

    - Lazy: DataFrame is built only when needed.
    - Cached: repeated calls reuse the same DataFrame.
    - Logging: supports summary logging and NDJSON row logging.
    """

    cfg: RiskStaticConfig
    proxy_resolution: ProxyResolution
    name: str = "proxy_resolution_audit"
    component: str = "simlab.resolve.proxy_audit"

    # Tie all row-lines to one emission/run for easy filtering/grouping.
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # Cached df - build once.
    _df: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    @property
    def df(self) -> pd.DataFrame:
        return self.to_frame()

    def to_frame(self) -> pd.DataFrame:
        if self._df is None:
            self._df = _build_audit_df(
                cfg=self.cfg, proxy_resolution=self.proxy_resolution
            )
        return self._df

    def to_csv(self, path: str, *, index: bool = False, **kwargs) -> str:
        self.to_frame().to_csv(path, index=index, **kwargs)
        return path

    def _logger(self, logger: Optional[LoggerLike]) -> logging.LoggerAdapter:
        # Normalize to a base Logger, regardless of whether caller passed a LoggerAdapter.
        base: Optional[logging.Logger]
        if isinstance(logger, logging.LoggerAdapter):
            base = logger.logger
        else:
            base = logger

        if base is None:
            base = get_logger(__name__)

        ctx = LogContext(
            component=self.component,
            object_name=self.name,
            object_id=self.run_id,
        )
        return with_context(base, ctx)

    def log_summary(
        self,
        *,
        logger: Optional[LoggerLike] = None,
        level: int = logging.INFO,
    ) -> None:
        """
        Human-friendly headline summary: row count + source_code counts.
        """
        log = self._logger(logger)
        df = self.to_frame()

        counts = df["source_code"].value_counts(dropna=False)
        counts_str = ", ".join(f"{k}={int(v)}" for k, v in counts.items())

        log.log(
            level,
            "audit_summary rows=%d source_code_counts=[%s]",
            int(df.shape[0]),
            counts_str,
        )

    def log_rows_ndjson(
        self,
        *,
        logger: Optional[LoggerLike] = None,
        level: int = logging.INFO,
        event: str = "proxy_resolution_audit_row",
        columns: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Emit the entire audit table as NDJSON: one log line per row.

        This is designed to be:
          - easy to search (grep/ripgrep)
          - easy to reconstruct (pandas.read_json(..., lines=True)) provided the log line is JSON.

        NOTE:
        - For best ingestion, write these lines to a dedicated handler/file using a formatter of '%(message)s'
          so each line is pure JSON (no timestamp/level prefix).
        """
        log = self._logger(logger)
        df = self.to_frame()

        if columns is not None:
            cols = list(columns)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise KeyError(
                    f"Requested audit columns not in DataFrame: {missing}"
                )
            df = df.loc[:, cols]

        col_names = list(df.columns)
        audit_name = self.name
        run_id = self.run_id

        # Iterate without materializing a big list of dicts.
        for i, row in enumerate(df.itertuples(index=False, name=None)):
            row_dict = {
                col_names[j]: _jsonable(row[j]) for j in range(len(col_names))
            }
            payload = {
                "event": event,
                "audit_name": audit_name,
                "run_id": run_id,
                "row_idx": i,
                **row_dict,
            }
            msg = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            log.log(level, msg)


def build_proxy_resolution_audit_df(
    *,
    cfg: RiskStaticConfig,
    proxy_resolution: ProxyResolution,
) -> pd.DataFrame:
    """
    Backwards-compatible functional entrypoint.
    """
    return _build_audit_df(cfg=cfg, proxy_resolution=proxy_resolution)
