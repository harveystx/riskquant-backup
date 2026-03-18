# src/riskquant/portfolio/blocks/base_block.py


from dataclasses import dataclass
from dataclasses import fields
import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class Block:
    def print_block(self, max_items: int = 5) -> None:
        cls = type(self).__name__
        print(f"{cls}:")

        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                print(
                    f"  {f.name:<30} | shape: {v.shape!s:>10} | dtype: {v.dtype!s:>10} | values: {v[:max_items]}"
                )
            else:
                print(f"  {f.name:<30} {v}")
