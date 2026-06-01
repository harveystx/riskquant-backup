from pandas import DataFrame
from pandas.api.types import is_string_dtype


def validate_string_cols(df: DataFrame, cols: tuple[str, ...] | list[str]) -> None:
    for col in cols:
        if col not in df.columns:
            continue

        s = df[col]

        if not is_string_dtype(s):
            raise ValueError(f"Column '{col}' must be of string dtype.")

        non_na = s.dropna()
        if non_na.empty:
            continue

        # Length
        max_len = non_na.str.len().max()
        if max_len > 255:
            raise ValueError(
                f"Column '{col}' contains strings longer than 255 characters "
                f"(max={max_len})."
            )

        # Canonical format
        pattern = r"^[A-Z0-9]+(?:_[A-Z0-9]+)*(?:\[(?:[A-Z]+[+-]\d+|(?:ON|TN|SN|\d+[DWMY]))\])?$"
        if not non_na.str.match(pattern).all():
            raise ValueError(
                f"Column '{col}' contains non-canonical strings. "
                f"Expected format: UPPERCASE tokens separated by underscores."
            )
