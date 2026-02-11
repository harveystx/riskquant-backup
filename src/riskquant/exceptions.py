# riskquant/exceptions.py


class PrettyKeyError(KeyError):
    """
    KeyError variant that prints the message using str(...) instead of repr(...),
    allowing multi-line, human-readable diagnostics while preserving KeyError
    semantics.
    """

    def __str__(self) -> str:
        return str(self.args[0])
