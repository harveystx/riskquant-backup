# riskquant/rq_logging.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"


def configure_logging(
    *,
    level: str | int = "INFO",
    fmt: str = _DEFAULT_FORMAT,
) -> None:
    """
    Configure root logging (intended for scripts / CLIs / notebooks).

    Library code should NOT call this automatically.
    """
    logging.basicConfig(level=level, format=fmt)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@dataclass(frozen=True, slots=True)
class LogContext:
    """
    Structured context you want consistently attached to log records.

    Keep this small; it flows into every record for an object.
    """

    component: str
    object_name: Optional[str] = None
    object_id: Optional[str] = None

    def to_extra(self) -> Dict[str, Any]:
        # Put under a single key to avoid clobbering LogRecord attrs.
        return {
            "rq": {
                "component": self.component,
                "object_name": self.object_name,
                "object_id": self.object_id,
            }
        }


class ContextLoggerAdapter(logging.LoggerAdapter):
    """
    Keeps contextual fields attached to log records.
    """

    def process(self, msg, kwargs):
        extra = kwargs.get("extra")
        base = dict(self.extra)  # adapter's extra
        if extra:
            # merge; nested dicts are simplest (rq namespace)
            base.update(extra)
        kwargs["extra"] = base
        return msg, kwargs


def with_context(logger: logging.Logger, ctx: LogContext) -> logging.LoggerAdapter:
    return ContextLoggerAdapter(logger, ctx.to_extra())
