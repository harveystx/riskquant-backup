# scripts/demos/simlab/scenarios/core_math/demo_sbb_path_construction_visualizer.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from riskquant.simlab.scenarios.core_math.sbb_block_generator import (
    generate_sbb_path_indices,
)


def compute_shifts(
    levels: np.ndarray,
    shift_type: str,
) -> np.ndarray:
    if shift_type == "difference":
        return np.diff(levels)

    if shift_type == "simple_return":
        return levels[1:] / levels[:-1] - 1.0

    if shift_type == "log_return":
        return np.log(levels[1:] / levels[:-1])

    raise ValueError(f"Unknown shift_type={shift_type}")


def reconstruct_levels(
    terminal_level: float,
    bootstrapped_shifts: np.ndarray,
    shift_type: str,
) -> np.ndarray:
    out = np.empty(len(bootstrapped_shifts) + 1)
    out[0] = terminal_level

    if shift_type == "difference":
        for i, shift in enumerate(bootstrapped_shifts):
            out[i + 1] = out[i] + shift

    elif shift_type == "simple_return":
        for i, shift in enumerate(bootstrapped_shifts):
            out[i + 1] = out[i] * (1.0 + shift)

    elif shift_type == "log_return":
        for i, shift in enumerate(bootstrapped_shifts):
            out[i + 1] = out[i] * np.exp(shift)

    else:
        raise ValueError(f"Unknown shift_type={shift_type}")

    return out


def identify_blocks(
    path_indices: np.ndarray,
    max_valid_index: int,
) -> list[tuple[int, int]]:
    """
    Reconstruct block boundaries from stitched indices.

    Returns
    -------
    list[(start_position_in_path, block_length)]
    """

    blocks: list[tuple[int, int]] = []

    block_start = 0

    for i in range(1, len(path_indices)):
        prev_idx = path_indices[i - 1]
        curr_idx = path_indices[i]

        expected_next = 1 + ((prev_idx - 1 + 1) % max_valid_index)

        if curr_idx != expected_next:
            blocks.append((block_start, i - block_start))
            block_start = i

    blocks.append((block_start, len(path_indices) - block_start))

    return blocks


def plot_sbb_demo(
    levels: np.ndarray,
    shift_type: str,
    *,
    horizon: int = 60,
    expected_block_size: float = 10.0,
    seed: int = 1234,
) -> None:
    rng = np.random.default_rng(seed)

    shifts = compute_shifts(levels, shift_type)

    path_indices = generate_sbb_path_indices(
        t_max=len(levels),
        h=horizon,
        total_scenarios=1,
        expected_block_size=expected_block_size,
        rng=rng,
    )[0]

    bootstrapped_shifts = shifts[path_indices - 1]

    simulated_levels = reconstruct_levels(
        terminal_level=float(levels[-1]),
        bootstrapped_shifts=bootstrapped_shifts,
        shift_type=shift_type,
    )

    blocks = identify_blocks(
        path_indices,
        max_valid_index=len(levels) - 1,
    )

    cmap = plt.get_cmap("tab20")

    _ = plt.figure(figsize=(16, 12))

    ax_levels = plt.subplot(4, 1, 1)
    ax_returns = plt.subplot(4, 1, 2)
    ax_stitched = plt.subplot(4, 1, 3)
    ax_sim = plt.subplot(4, 1, 4)

    # ------------------------------------------------------------------
    # Historical levels
    # ------------------------------------------------------------------

    ax_levels.plot(levels, lw=1.5)
    ax_levels.set_title(f"Historical Levels ({shift_type})")

    # ------------------------------------------------------------------
    # Historical shifts with highlighted selected blocks
    # ------------------------------------------------------------------

    ax_returns.plot(
        np.arange(1, len(levels)),
        shifts,
        color="lightgrey",
        lw=1,
        zorder=1,
    )

    for block_num, (path_start, block_len) in enumerate(blocks):
        colour = cmap(block_num % 20)

        hist_start = path_indices[path_start]
        hist_end = path_indices[path_start + block_len - 1]

        if hist_end >= hist_start:
            x = np.arange(hist_start, hist_end + 1)

            ax_returns.plot(
                x,
                shifts[x - 1],
                lw=4,
                color=colour,
                zorder=3,
            )
        else:
            # wrapped historical block
            x1 = np.arange(hist_start, len(levels))
            x2 = np.arange(1, hist_end + 1)

            ax_returns.plot(
                x1,
                shifts[x1 - 1],
                lw=4,
                color=colour,
                zorder=3,
            )

            ax_returns.plot(
                x2,
                shifts[x2 - 1],
                lw=4,
                color=colour,
                zorder=3,
            )

    ax_returns.set_title("Historical Shifts with Selected Blocks Highlighted")

    # ------------------------------------------------------------------
    # Stitched returns
    # ------------------------------------------------------------------

    boundary = 0

    for block_num, (_, block_len) in enumerate(blocks):
        colour = cmap(block_num % 20)

        x = np.arange(boundary, boundary + block_len)

        ax_stitched.plot(
            x,
            bootstrapped_shifts[boundary : boundary + block_len],
            lw=3,
            color=colour,
        )

        if boundary > 0:
            ax_stitched.axvline(
                boundary,
                ls="--",
                alpha=0.5,
            )

        boundary += block_len

    ax_stitched.set_title("Bootstrapped Shifts After Stitching")

    # ------------------------------------------------------------------
    # Simulated levels
    # ------------------------------------------------------------------

    ax_sim.plot(simulated_levels, lw=2)

    boundary = 0
    for _, block_len in blocks:
        if boundary > 0:
            ax_sim.axvline(
                boundary,
                ls="--",
                alpha=0.5,
            )
        boundary += block_len

    ax_sim.set_title("Simulated Levels (starting from historical terminal value)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    n = 260

    log_returns = 0.0005 + 0.01 * rng.standard_normal(n)

    levels = 100.0 * np.exp(np.cumsum(log_returns))

    for shift_type in (
        # "difference",
        # "simple_return",
        "log_return",
    ):
        plot_sbb_demo(
            levels,
            shift_type,
            horizon=100,
            expected_block_size=20,
            seed=123_123,
        )
