# src/riskquant_lab/fixtures/credit_exposure.py
"""
Canonical exposure book (shared fixture).

A small, hermetic, seeded trade universe for exercising and locking the credit-
exposure contracts end to end: trade -> CEM -> netting-set EAD -> obligor EAD ->
JTD. The book trades ONLY environmental commodities (RNG, biodiesel, RECs, RINs,
carbon allowances). The asset subclasses and their supervisory add-on factors are
HOUSE-DEFINED, not SA-CCR's taxonomy — the engine takes SA-CCR's structure but our
own asset classes and parameters.

The book is deliberately laid out so every contract edge is exercised: netting
sets with offsetting marks (netting benefit), a collateralised set, a fully-OTM
set (PFE floor), and more netting sets than obligors (so netting-set -> obligor
aggregation is a genuine many-to-one).
"""

from __future__ import annotations

__all__ = [
    "CanonicalExposureBook",
    "EnvironmentalSubclass",
    "generate_canonical_exposure_book",
]

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

# House-defined environmental subclasses:
# --> (name, uix, supervisory add-on factor).
# NOT SA-CCR asset classes — house universe and calibration.
EnvironmentalSubclass: Final = (
    ("RNG", 1, 0.20),  # renewable natural gas
    ("BIODIESEL", 2, 0.18),
    ("REC", 3, 0.15),  # renewable energy certificate
    ("RIN", 4, 0.25),  # renewable identification number ("tickets")
    ("CARBON_ALLOWANCE", 5, 0.22),  # emissions certificates
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CanonicalExposureBook:
    """Post-ingest, engine-ready SoA for the canonical book (Contract A inputs)."""

    # --- trade-level SoA (aligned, portfolio order) ---
    trade_uid: npt.NDArray[np.int64]  # (M,)
    trade_netting_set_id: npt.NDArray[np.int64]  # (M,)
    asset_subclass_uix: npt.NDArray[np.int32]  # (M,)
    volume: npt.NDArray[np.float64]  # (M,) contract volume
    current_price: npt.NDArray[np.float64]  # (M,) CURRENT market price (not inception)
    mtm: npt.NDArray[np.float64]  # (M,) current mark (valuation-engine stand-in)

    # --- netting-set master (aligned, ascending uid) ---
    netting_set_uid: npt.NDArray[np.int64]  # (K,)
    netting_set_obligor_uid: npt.NDArray[np.int64]  # (K,)
    netting_set_collateral: npt.NDArray[np.float64]  # (K,) net VM

    # --- supervisory add-on config (aligned, ascending uix) ---
    subclass_uix: npt.NDArray[np.int32]  # (C,)
    subclass_factor: npt.NDArray[np.float64]  # (C,)
    subclass_name: tuple[str, ...]  # (C,) display labels, aligned to subclass_uix

    @property
    def obligor_universe(self) -> npt.NDArray[np.int64]:
        """Distinct obligors, ascending — the JTD aggregation target order."""
        return np.unique(self.netting_set_obligor_uid)

    def trades_frame(self) -> pd.DataFrame:
        """Human-readable trade table (the authoring edge of Contract A)."""
        name_by_uix = dict(
            zip(
                self.subclass_uix.tolist(),
                self.subclass_name,
                strict=True,
            )
        )
        return pd.DataFrame(
            {
                "trade_uid": self.trade_uid,
                "netting_set_id": self.trade_netting_set_id,
                "asset_subclass": [
                    name_by_uix[u] for u in self.asset_subclass_uix.tolist()
                ],
                "volume": self.volume,
                "current_price": self.current_price,
                "mtm": self.mtm,
            }
        )


def generate_canonical_exposure_book(
    *,
    n_trades: int = 20,
    n_netting_sets: int = 7,
    n_obligors: int = 5,
    seed: int = 999_999,
) -> CanonicalExposureBook:
    """Build the canonical exposure book (deterministic, hermetic).

    Netting sets outnumber obligors, so netting-set -> obligor aggregation is a
    genuine many-to-one; marks are drawn to include offsetting, collateralised and
    fully-OTM sets.
    """
    if not (n_obligors <= n_netting_sets <= n_trades):
        raise ValueError("require n_obligors <= n_netting_sets <= n_trades.")
    rng = np.random.default_rng(seed)

    subclass_uix = np.array([u for _, u, _ in EnvironmentalSubclass], dtype=np.int32)
    subclass_factor = np.array(
        [f for _, _, f in EnvironmentalSubclass], dtype=np.float64
    )
    subclass_name = tuple(n for n, _, _ in EnvironmentalSubclass)

    trade_uid = np.arange(1, n_trades + 1, dtype=np.int64)

    # Each netting set gets >=1 trade (first K trades seed the sets), rest random.
    ns_ids = np.arange(1, n_netting_sets + 1, dtype=np.int64)
    trade_netting_set_id = np.empty(n_trades, dtype=np.int64)
    trade_netting_set_id[:n_netting_sets] = ns_ids
    trade_netting_set_id[n_netting_sets:] = rng.choice(
        ns_ids, size=n_trades - n_netting_sets
    )

    asset_subclass_uix = rng.choice(subclass_uix, size=n_trades).astype(np.int32)

    volume = np.round(rng.uniform(1_000.0, 100_000.0, n_trades), 0)
    volume = np.round(volume / 1_000.0, 0) * 1_000.0  # round to nearest 1k

    current_price = np.round(rng.uniform(5.0, 120.0, n_trades), 0)

    mtm = np.round(rng.uniform(-2_000_000.0, 2_000_000.0, n_trades), 0)
    mtm = np.round(mtm / 10_000.0, 0) * 10_000.0  # round to nearest 10k

    # Netting sets -> obligors (many-to-one). Each obligor owns >=1 netting set.
    obligor_ids = np.arange(1, n_obligors + 1, dtype=np.int64)
    ns_obligor = np.empty(n_netting_sets, dtype=np.int64)
    ns_obligor[:n_obligors] = obligor_ids
    ns_obligor[n_obligors:] = rng.choice(obligor_ids, size=n_netting_sets - n_obligors)

    # Master keyed ascending by netting-set uid.
    order = np.argsort(ns_ids, kind="stable")

    netting_set_collateral = np.round(rng.uniform(0.0, 1_500_000.0, n_netting_sets), 0)
    netting_set_collateral = (
        np.round(netting_set_collateral / 100_000.0, 0) * 100_000.0
    )  # round to nearest 100k

    return CanonicalExposureBook(
        trade_uid=trade_uid,
        trade_netting_set_id=trade_netting_set_id,
        asset_subclass_uix=asset_subclass_uix,
        volume=volume,
        current_price=current_price,
        mtm=mtm,
        netting_set_uid=np.ascontiguousarray(ns_ids[order]),
        netting_set_obligor_uid=np.ascontiguousarray(ns_obligor[order]),
        netting_set_collateral=np.ascontiguousarray(netting_set_collateral[order]),
        subclass_uix=subclass_uix,
        subclass_factor=subclass_factor,
        subclass_name=subclass_name,
    )
