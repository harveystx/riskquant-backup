# riskquant/utils/printer.py
from typing import Any, Iterable, Mapping

import pandas as pd


def as_list_str(x: Iterable) -> str:
    # Convert an iterable of items to a nicely formatted string with each
    # item on its own line, prefixed by "- ".
    items = (str(i) for i in x)  # generator avoids extra allocations
    return "- " + ",\n- ".join(items) + "."


def print_as_list_str(x: Iterable) -> None:
    print(as_list_str(x))


def print_wrapped_text_box(
    text: str,
    *,
    width: int = 70,
    padding: int = 1,
    corner_char: str = "+",
    top_border: str = "-",
    bottom_border: str = "-",
    horiz_border: str = "|",
    add_space_before: bool = True,
    add_space_after: bool = True,
) -> None:
    """Word‑wrap `text` to `width` characters and display it in an ASCII box.

    The box will have a single‑character border and the given number of spaces
    between the text and the vertical edges.

    Example:

        +-------------------------------+
        | This is a long sentence that  |
        | will be wrapped automatically |
        | and displayed in a neat box.  |
        +-------------------------------+

    Args:
        text: string to render.
        width: max characters per line inside the box.
        padding: spaces between text and left/right borders.
    """
    import textwrap  # local import keeps top‑level imports unchanged

    lines = textwrap.wrap(text, width=width)
    if not lines:
        lines = [""]
    max_len = max(len(line) for line in lines)

    inner_width = max_len + padding * 2
    horiz = corner_char + top_border * inner_width + corner_char

    if add_space_before:
        print()
    print(horiz)
    for line in lines:
        print(
            horiz_border
            + " " * padding
            + line.ljust(max_len)
            + " " * padding
            + horiz_border
        )
    print(corner_char + bottom_border * inner_width + corner_char)
    if add_space_after:
        print()


def print_success_message() -> None:
    print_wrapped_text_box(
        "SUCCESS! CODE RAN SUCCESSFULLY!",
        width=60,
        padding=2,
        corner_char="*",
        top_border="=",
        bottom_border="=",
    )


def print_dict(
    d: Mapping[str, Any],
    *,
    indent: int = 0,
    indent_step: int = 4,
    sort_keys: bool = False,
    limit: int | None = None,
) -> None:
    """
    Pretty-print a dictionary with aligned keys and nested support.

    Args:
        d: Dictionary (or mapping) to print.
        indent: Initial indentation level.
        indent_step: Spaces added per nesting level.
        sort_keys: If True, keys are printed alphabetically.
        limit: Maximum number of keys to print (None = all).
    """
    if not isinstance(d, Mapping):
        raise TypeError("print_dict expects a mapping type.")

    items = sorted(d.items()) if sort_keys else list(d.items())
    if limit is not None:
        items = items[:limit]

    # Compute key width once for alignment
    try:
        max_key_len = max(len(str(k)) for k in d.keys())
    except ValueError:  # empty dict
        max_key_len = 0

    pad = " " * indent

    print(pad + "{")
    for key, value in items:
        key_str = str(key)
        key_pad = key_str.ljust(max_key_len)

        if isinstance(value, Mapping):
            print(pad + " " * indent_step + f"{key_pad}:")
            print_dict(
                value,
                indent=indent + indent_step * 2,
                indent_step=indent_step,
                sort_keys=sort_keys,
                limit=None,  # Don't limit nested dicts
            )
        else:
            print(pad + " " * indent_step + f"{key_pad}: {value}")

    print(pad + "}")


class _FloatFormatter:
    __slots__ = ("_fmt", "_zero_tol")

    def __init__(self, spec: str, zero_tol: float = 0.0) -> None:
        # spec like ".2f", "_.2f", ",.3f"
        self._fmt = "{:" + spec + "}"
        self._zero_tol = float(zero_tol)

    def __call__(self, x: float) -> str:
        # pandas will only call this for float entries
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "nan"
        if self._zero_tol and abs(x) < self._zero_tol:
            x = 0.0
        return self._fmt.format(x)


def print_df(
    obj: pd.Series | pd.DataFrame,
    *,
    index: bool = True,
    max_rows: int | None = None,
    max_cols: int | None = None,
    float_format: str | None = "_.2f",
    add_space_before: bool = True,
    add_space_after: bool = True,
    zero_tol: float = 0.0,
) -> None:
    """
    Print a Series/DataFrame with optional display controls.

    float_format:
        Format specifier like ".2f", "_.2f", ",.3f", etc.
        Use None to disable float formatting.
    zero_tol:
        If > 0, values with abs(x) < zero_tol are printed as 0.0.
        Useful to avoid "-0.00".
    """
    if isinstance(obj, pd.Series):
        name = obj.name if obj.name is not None else "value"
        df = obj.to_frame(name=name)
    else:
        df = obj

    if add_space_before:
        print()

    if float_format is None:
        print(df.to_string(index=index, max_rows=max_rows, max_cols=max_cols))
    else:
        formatter = _FloatFormatter(float_format, zero_tol=zero_tol)
        print(
            df.to_string(
                index=index,
                max_rows=max_rows,
                max_cols=max_cols,
                float_format=formatter,
            )
        )

    if add_space_after:
        print()
