# src/riskquant/config/enums/credit_ratings.py


from enum import Enum


class GenericCreditRating(str, Enum):
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    DEFAULT = "D"


VALID_GENERIC_CREDIT_RATINGS: frozenset[str] = frozenset(
    e.value for e in GenericCreditRating
)
