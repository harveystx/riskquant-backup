# riskquant/utils/printer.py
from typing import Iterable


def as_list_str(x: Iterable) -> str:
    items = (str(i) for i in x)  # generator avoids extra allocations
    return "- " + ",\n- ".join(items) + "."
