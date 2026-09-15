# src/riskquant/utils/printer.py
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# ==========================================
# DEFAULT STYLING & ANSI COLORS
# ==========================================
ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_FOREST_GREEN = "\033[38;5;28m"  # A rich, medium-dark green (#008700)
ANSI_DEEP_GREEN = "\033[38;5;22m"  # A very dark, subtle green (#005f00)
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"
ANSI_MAGENTA = "\033[95m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"
ANSI_BLUE = "\033[34m"  # Standard Dark/Deep Blue
ANSI_BRIGHT_BLUE = (
    "\033[94m"  # High-intensity Blue (matches the brightness of your other variables)
)
ANSI_MIDNIGHT_BLUE = "\033[38;5;17m"  # Deepest dark blue (#00005f)
ANSI_NAVY_BLUE = "\033[38;5;18m"  # Slightly brighter navy (#000087)

ANSI_BLACK = "\033[30m"
ANSI_BRIGHT_BLACK = "\033[90m"  # Dark gray, great for subdued metadata
ANSI_BLACK_BG = "\033[40m"


# Default colors for title hierarchy (same blue palette)
DEFAULT_TITLE_COLOR = ANSI_BRIGHT_BLUE  # H0 – most prominent
DEFAULT_HEADER_COLOR = ANSI_BLUE  # H1 – standard
DEFAULT_SUBHEADER_COLOR = ANSI_NAVY_BLUE  # H2 – subdued


def print_title(
    title: str,
    *,
    step_delta: int | None = None,
    add_space_before: bool = True,
    add_space_after: bool = True,
    color: str | None = None,
) -> None:
    """
    Prints a highly prominent, double-line boxed title (H0).
    Acts as the top-level title renderer, above header (H1) and subheader (H2).
    """
    if step_delta is not None:
        title_ = f"STEP N - {abs(step_delta):1.0f}: {title}"
    else:
        title_ = title

    c = color or DEFAULT_TITLE_COLOR

    if add_space_before:
        print()

    # Double-line box with extra padding and blank lines for emphasis
    inner_width = len(title_) + 4  # two spaces on each side
    top_border = f"╔{'═' * inner_width}╗"
    empty_line = f"║{' ' * inner_width}║"
    title_line = f"║  {title_}  ║"
    bottom_border = f"╚{'═' * inner_width}╝"

    print(f"{c}{ANSI_BOLD}{top_border}{ANSI_RESET}")
    print(f"{c}{ANSI_BOLD}{empty_line}{ANSI_RESET}")
    print(f"{c}{ANSI_BOLD}{title_line}{ANSI_RESET}")
    print(f"{c}{ANSI_BOLD}{empty_line}{ANSI_RESET}")
    print(f"{c}{ANSI_BOLD}{bottom_border}{ANSI_RESET}")

    if add_space_after:
        print()


def print_header(
    title: str,
    *,
    step_delta: int | None = None,
    add_space_before: bool = True,
    add_space_after: bool = True,
    color: str | None = None,
) -> None:
    """
    Prints a stylized, rounded Unicode box containing the title.
    Acts as the master title renderer for all other print functions.
    """
    if step_delta is not None:
        title_ = f"STEP N - {abs(step_delta):1.0f}: {title}"
    else:
        title_ = title

    c = color or DEFAULT_HEADER_COLOR

    if add_space_before:
        print()

    width = len(title_) + 2
    # print(f"{c}{ANSI_BOLD}╭{'─' * width}╮{ANSI_RESET}")
    # print(f"{c}{ANSI_BOLD}│ {title_} │{ANSI_RESET}")
    # print(f"{c}{ANSI_BOLD}╰{'─' * width}╯{ANSI_RESET}")
    print(f"{c}{ANSI_BOLD}┏{'━' * width}┓{ANSI_RESET}")
    print(f"{c}{ANSI_BOLD}┃ {title_} ┃{ANSI_RESET}")
    print(f"{c}{ANSI_BOLD}┗{'━' * width}┛{ANSI_RESET}")

    if add_space_after:
        print()


def print_subheader(
    title: str,
    *,
    step_delta: int | None = None,
    add_space_before: bool = True,
    add_space_after: bool = True,
    color: str | None = None,
) -> None:
    """
    Prints a stylized, rounded Unicode box containing the title.
    Acts as the master title renderer for all other print functions.
    """
    if step_delta is not None:
        title_ = f"STEP N - {abs(step_delta):1.0f}: {title}"
    else:
        title_ = title

    c = color or DEFAULT_SUBHEADER_COLOR

    if add_space_before:
        print()

    width = len(title_) + 2
    print(f"{c}{ANSI_BOLD}  {title_}  {ANSI_RESET}")
    print(f"{c}{ANSI_BOLD} {'─' * width} {ANSI_RESET}")

    if add_space_after:
        print()


def print_text(
    text: str,
    *,
    width: int = 70,
    add_space_before: bool = True,
    add_space_after: bool = True,
) -> None:
    import textwrap

    if add_space_before:
        print()
    print(textwrap.fill(text, width=width))
    if add_space_after:
        print()


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
    color: str | None = None,
) -> None:
    """Word-wrap `text` to `width` characters and display it in an ASCII box."""
    import textwrap

    lines = textwrap.wrap(text, width=width)
    if not lines:
        lines = [""]
    max_len = max(len(line) for line in lines)

    inner_width = max_len + padding * 2
    horiz = corner_char + top_border * inner_width + corner_char

    c = color or ""
    reset = ANSI_RESET if color else ""
    bold = ANSI_BOLD if color else ""

    if add_space_before:
        print()

    print(f"{c}{bold}{horiz}{reset}")
    for line in lines:
        print(
            f"{c}{bold}{horiz_border}{reset}"
            + " " * padding
            + line.ljust(max_len)
            + " " * padding
            + f"{c}{bold}{horiz_border}{reset}"
        )
    print(f"{c}{bold}{corner_char + bottom_border * inner_width + corner_char}{reset}")

    if add_space_after:
        print()


def print_success_message(
    message: str | None = None,
    *,
    width: int = 70,
    padding: int = 2,
    add_space_before: bool = True,
    add_space_after: bool = True,
    uppercase: bool = True,
    color: str | None = ANSI_GREEN,
) -> None:
    """
    Print a success message inside a wrapped, rounded ANSI box (matching print_header).
    """
    import textwrap

    s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if message:
        s += f" | {message}"
    else:
        s += " | SUCCESS! CODE RAN SUCCESSFULLY!"

    text = s.upper() if uppercase else s.capitalize()

    # Handle text wrapping
    lines = textwrap.wrap(text, width=width)
    if not lines:
        lines = [""]
    max_len = max(len(line) for line in lines)
    inner_width = max_len + padding * 2

    c = color or ""
    reset = ANSI_RESET if color else ""
    bold = ANSI_BOLD if color else ""

    if add_space_before:
        print()

    # Top border
    print(f"{c}{bold}╭{'─' * inner_width}╮{reset}")

    # Body (Color applied to the entire line)
    for line in lines:
        print(
            f"{c}{bold}│"
            + " " * padding
            + line.ljust(max_len)
            + " " * padding
            + f"│{reset}"
        )

    # Bottom border
    print(f"{c}{bold}╰{'─' * inner_width}╯{reset}")

    if add_space_after:
        print()


def print_calling_file(
    *,
    width: int = 70,
    padding: int = 2,
    add_space_before: bool = True,
    add_space_after: bool = True,
    color: str | None = ANSI_BLACK,
) -> None:
    """
    Print the name of the Python file that called this function
    inside a wrapped, rounded ANSI box (matching print_header).
    """
    import inspect
    import os
    import textwrap

    frame = inspect.currentframe()
    try:
        caller_frame = frame.f_back if frame else None
        filename = caller_frame.f_code.co_filename if caller_frame else "<unknown>"
        filename = os.path.basename(filename)
    finally:
        # Explicitly delete frame references to avoid reference cycles
        del frame

    dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"{dt_str} | RUNNING FILE: {filename}"

    # Handle text wrapping
    lines = textwrap.wrap(text, width=width)
    if not lines:
        lines = [""]
    max_len = max(len(line) for line in lines)
    inner_width = max_len + padding * 2

    c = color or ""
    reset = ANSI_RESET if color else ""
    bold = ANSI_BOLD if color else ""

    if add_space_before:
        print()

    # Top border
    print(f"{c}{bold}╭{'─' * inner_width}╮{reset}")

    # Body (Color applied to the entire line)
    for line in lines:
        print(
            f"{c}{bold}│"
            + " " * padding
            + line.ljust(max_len)
            + " " * padding
            + f"│{reset}"
        )

    # Bottom border
    print(f"{c}{bold}╰{'─' * inner_width}╯{reset}")

    if add_space_after:
        print()


def print_dict(
    d: dict[Any, Any | np.integer | np.floating]
    | Mapping[Any, Any | np.integer | np.floating],
    *,
    title: str | None = None,
    indent: int = 0,
    indent_step: int = 4,
    sort_keys: bool = False,
    limit: int | None = None,
    add_space_before: bool = True,
    add_space_after: bool = True,
    title_color: str | None = None,
) -> None:
    """Pretty-print a dictionary with aligned keys and nested support."""
    items = sorted(d.items()) if sort_keys else list(d.items())
    if limit is not None:
        items = items[:limit]

    try:
        max_key_len = max(len(str(k)) for k in d)
    except ValueError:
        max_key_len = 0

    pad = " " * indent

    if add_space_before:
        print()

    if title:
        print_header(
            title=f"Dictionary: {title}",
            color=title_color,
            add_space_before=False,
            add_space_after=False,
        )

    print(pad + "{")
    for key, value in items:
        key_str = str(key)
        key_pad = key_str.ljust(max_key_len)

        if isinstance(value, (Mapping, dict)):
            print(pad + " " * indent_step + f"{key_pad}:")
            print_dict(
                d=value,
                indent=indent + indent_step * 2,
                indent_step=indent_step,
                sort_keys=sort_keys,
                limit=None,
                add_space_before=False,
                add_space_after=False,
            )
        else:
            print(pad + " " * indent_step + f"{key_pad}: {value}")

    print(pad + "}")

    if add_space_after:
        print()


# class _FloatFormatter:
#     __slots__ = ("_fmt", "_spec", "_zero_tol")

#     def __init__(self, spec: str, zero_tol: float = 0.0) -> None:
#         self._spec = spec
#         if "_" in spec:
#             clean_spec = spec.replace("_", ",")
#             self._fmt = lambda x, s=clean_spec: f"{x:{s}}".replace(",", "_")
#         else:
#             self._fmt = lambda x, s=spec: f"{x:{s}}"
#         self._zero_tol = float(zero_tol)

#     def __call__(self, x: float) -> str:
#         if x is None or (isinstance(x, float) and math.isnan(x)):
#             # Handle NaN alignment for Numpy formats
#             try:
#                 width = int(self._spec.split(".")[0].replace(">", "").replace("<", ""))
#                 return "nan".rjust(width)
#             except (ValueError, TypeError, IndexError):
#                 return "nan"
#         if self._zero_tol and abs(x) < self._zero_tol:
#             x = 0.0
#         try:
#             return self._fmt(x)
#         except (ValueError, TypeError):
#             return str(x)


class _FloatFormatter:
    __slots__ = ("_spec", "_zero_tol")

    def __init__(self, spec: str, zero_tol: float = 0.0) -> None:
        spec = spec.strip().removeprefix("{").removesuffix("}")
        self._spec = spec.removeprefix(":")
        self._zero_tol = float(zero_tol)

    def __call__(self, x: float) -> str:
        if np.isnan(x):
            return "nan"

        if self._zero_tol and abs(x) < self._zero_tol:
            x = 0.0

        return format(x, self._spec)


def print_df(
    obj: pd.Series | pd.DataFrame,
    *,
    title: str | None = None,
    index: bool = False,
    max_rows: int | None = None,
    max_cols: int | None = None,
    float_format: str | None = "_.0f",
    add_space_before: bool = True,
    add_space_after: bool = True,
    zero_tol: float = 0.0,
    title_color: str | None = None,
) -> None:
    """Print a Series/DataFrame with optional display controls."""
    if isinstance(obj, pd.Series):
        name = obj.name if obj.name is not None else "value"
        df = obj.to_frame(name=name)
    else:
        df = obj

    if float_format is not None:
        numeric_cols = df.select_dtypes(
            include=["int64", "int32", "float64", "float32"]
        ).columns
        df = df.copy()
        df[numeric_cols] = df[numeric_cols].astype(float)

    if add_space_before:
        print()

    if title:
        df_attrs = f"(shape={df.shape})"
        print_header(
            title=f"DataFrame: {title} {df_attrs}",
            color=title_color,
            add_space_before=False,
            add_space_after=False,
        )

    if float_format is None:
        print(
            df.to_string(
                index=index,
                max_rows=max_rows,
                max_cols=max_cols,
            )
        )
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


def print_dict_of_dfs(
    d: dict[str, pd.DataFrame],
    *,
    title: str | None = None,
    index: bool = False,
    max_rows: int | None = None,
    max_cols: int | None = None,
    float_format: str | None = "_.0f",
    add_space_before: bool = True,
    add_space_after: bool = True,
    zero_tol: float = 0.0,
    cols_before_transpose: int = 20,
    title_color: str | None = ANSI_MIDNIGHT_BLUE,
    subtitle_color: str | None = ANSI_BLUE,
) -> None:
    """Print a dictionary of DataFrames with optional display controls."""
    if add_space_before:
        print()

    if title:
        # Use a slightly different color for the main dictionary wrapper if desired,
        # but defaulting to standard title behavior.
        print_header(
            title=f"Dictionary of DataFrames: {title}",
            color=title_color,
            add_space_before=False,
            add_space_after=True,
        )

    for key, df in d.items():
        if df.shape[1] > cols_before_transpose:
            df = df.T
            key += " (transposed)"
        print_df(
            df,
            title=key,
            index=index,
            max_rows=max_rows,
            max_cols=max_cols,
            float_format=float_format,
            add_space_before=False,
            add_space_after=True,
            zero_tol=zero_tol,
            title_color=subtitle_color,
        )

    if add_space_after:
        print()


def print_array(
    arr: np.ndarray,
    *,
    title: str | None = None,
    max_rows: int | None = None,
    max_cols: int | None = None,
    float_format: str | None = ">10.2f",
    add_space_before: bool = True,
    add_space_after: bool = True,
    zero_tol: float = 0.0,
    title_color: str | None = None,
) -> None:
    """Print a NumPy array with perfectly aligned tabular columns and ANSI colors."""
    _array_info = f"(shape={arr.shape}, dtype={arr.dtype})"

    if add_space_before:
        print()

    if title:
        full_title = f"Array: {title} {_array_info}"
        print_header(
            title=full_title,
            color=title_color,
            add_space_before=False,
            add_space_after=False,
        )

    if float_format is not None and np.issubdtype(arr.dtype, np.integer):
        arr_to_print = arr.astype(float)
    else:
        arr_to_print = arr

    print_opts = {
        "suppress": True,
        "linewidth": 200,
    }

    if max_rows is not None or max_cols is not None:
        r_edge = max_rows // 2 if max_rows is not None else 3
        c_edge = max_cols // 2 if max_cols is not None else 3
        print_opts["edgeitems"] = max(r_edge, c_edge)

        row_limit = max_rows if max_rows is not None else arr_to_print.shape[0]
        col_limit = (
            max_cols
            if max_cols is not None
            else (arr_to_print.shape[1] if arr_to_print.ndim > 1 else 1)
        )
        print_opts["threshold"] = row_limit * col_limit - 1

    if float_format is not None:
        formatter = _FloatFormatter(float_format, zero_tol=zero_tol)
        print_opts["formatter"] = {"float_kind": formatter}

    with np.printoptions(**print_opts):
        print(arr_to_print)

    if add_space_after:
        print()
