from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class Payload:
    """
    Abstract base class for all trade payloads.
    """

    pass
