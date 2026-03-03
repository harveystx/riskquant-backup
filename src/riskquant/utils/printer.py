# riskquant/utils/printer.py
from typing import Iterable


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
