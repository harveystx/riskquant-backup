# riskquant/exceptions.py


class PrettyKeyError(KeyError):
    """
    KeyError variant that prints the message using str(...) instead of repr(...),
    allowing multi-line, human-readable diagnostics while preserving KeyError
    semantics.
    """

    def __str__(self) -> str:
        return str(self.args[0])


class RiskQuantError(Exception):
    """Base exception for all custom riskquant errors."""

    pass


class ConfigurationError(RiskQuantError):
    """
    Raised when static configuration, ID namespaces, or
    market data mappings violate the engine's strict contracts.
    """

    pass
