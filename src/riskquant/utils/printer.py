# riskquant/utils/printer.py
from typing import Iterable


def as_list_str(x: Iterable) -> str:
    # Convert an iterable of items to a nicely formatted string with each
    # item on its own line, prefixed by "- ".
    items = (str(i) for i in x)  # generator avoids extra allocations
    return "- " + ",\n- ".join(items) + "."
