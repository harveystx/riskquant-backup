# src/riskquant/ingest_fixtures/rng.py
import numpy as np

from riskquant.ingest_fixtures.constants import CANONICAL_RNG_SEED


def get_canonical_rng() -> np.random.Generator:
    """Get a new RNG seeded with the canonical seed."""
    return np.random.default_rng(seed=CANONICAL_RNG_SEED)
