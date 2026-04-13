# src/riskquant/simlab/scenarios/core_math/sbb_block_generator.py


import numpy as np


def generate_sbb_path_indices(
    *,
    T: int,
    h: int,
    B: int,
    expected_block_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Vectorized generator for Stationary Block Bootstrap (SBB) path indices.

    Returns
    -------
    random_t_indices : np.ndarray
        A (B, h) matrix of int32 indices bounded strictly between [1, T-1].
        These represent the valid t1 indices to calculate 1-day historical returns.
    """
    # We need 1-day returns, so valid t1 indices are [1, T-1]
    max_valid_length = T - 1
    p = 1.0 / float(expected_block_size)

    # --- VECTORIZED RANDOM GENERATION ---
    # (B * h) is the absolute maximum number of blocks we could ever need
    max_possible_blocks = B * h

    if p >= 1.0:
        block_lengths = np.ones(max_possible_blocks, dtype=np.int32)
    else:
        block_lengths = rng.geometric(p, size=max_possible_blocks).astype(
            np.int32
        )

    # Draw unconstrained valid starting indices (1 to T-1)
    start_indices = rng.integers(
        low=1, high=T, size=max_possible_blocks, dtype=np.int32
    )

    # --- PATH STITCHING ---
    random_t_indices = np.empty((B, h), dtype=np.int32)

    block_idx = 0
    for b in range(B):
        step = 0
        while step < h:
            b_len = min(block_lengths[block_idx], h - step)
            start_t = start_indices[block_idx]

            # Generate raw sequence
            raw_indices = np.arange(start_t, start_t + b_len, dtype=np.int32)

            # Circular Wrap-Around: keeps indices bounded strictly between [1, T-1]
            wrapped = 1 + ((raw_indices - 1) % max_valid_length)

            random_t_indices[b, step : step + b_len] = wrapped

            step += b_len
            block_idx += 1

    return random_t_indices
