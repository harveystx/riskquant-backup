# riskquant/portfolio/payloads/base.py
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Payload:
    """
    Marker base class for instrument payloads.
    No methods; payloads are compiled into blocks.
    """

    def __repr__(self) -> str:
        return self.__str__()
