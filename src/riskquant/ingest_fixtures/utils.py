# src/riskquant/ingest_fixtures/utils.py


from collections.abc import Mapping, Sequence
from typing import Any, cast

from riskquant.ingest_fixtures.schemas import (
    BetaBinomialJTDIngestData,
    CreditFactorUniverseIngestData,
    DeterministicJTDIngestData,
    GaussianCopulaJTDIngestData,
    IndependentBernoulliJTDIngestData,
    MarketRiskIngestData,
    NettingSetMasterIngestData,
    ObligorCreditIngestData,
    RiskProxyConfigData,
    RiskStaticConfigCcy,
    RiskStaticConfigCurveMeta,
    RiskStaticConfigCurvePointMeta,
    RiskStaticConfigData,
    RiskStaticConfigHolidayCalendar,
    RiskStaticConfigNodeMeta,
    RiskStaticConfigSpotRollRule,
    RiskStaticConfigSurfaceMeta,
    RiskStaticConfigSurfacePointMeta,
    StressRegimeBernoulliJTDIngestData,
    TCopulaJTDIngestData,
)

_DATA = (
    MarketRiskIngestData
    | GaussianCopulaJTDIngestData
    | TCopulaJTDIngestData
    | IndependentBernoulliJTDIngestData
    | StressRegimeBernoulliJTDIngestData
    | BetaBinomialJTDIngestData
    | DeterministicJTDIngestData
    | ObligorCreditIngestData
    | NettingSetMasterIngestData
    | CreditFactorUniverseIngestData
    # Risk Static Config
    | RiskStaticConfigSpotRollRule
    | RiskStaticConfigData
    | RiskStaticConfigCcy
    | RiskStaticConfigHolidayCalendar
    | RiskStaticConfigNodeMeta
    | RiskStaticConfigCurveMeta
    | RiskStaticConfigCurvePointMeta
    | RiskStaticConfigSurfaceMeta
    | RiskStaticConfigSurfacePointMeta
    # Risk Proxy Config
    | RiskProxyConfigData
)


def _validate_single_table(data: Mapping[str, Sequence[Any]]) -> None:
    lengths = {len(v) for v in data.values()}

    if len(lengths) > 1:
        raise ValueError(f"Inconsistent column lengths: {lengths}")


def validate_column_lengths(data: Any) -> None:
    """
    Validates:
    - flat columnar dicts: Dict[str, Sequence]
    - nested config: Dict[str, Dict[str, Sequence]]
    """

    # Case 1: Nested (RiskStaticConfigData)
    if all(isinstance(v, Mapping) for v in data.values()):
        for k, sub_table in data.items():
            if not isinstance(sub_table, Mapping):
                raise TypeError(f"{k} is not a valid table")

            _validate_single_table(cast(Mapping[str, Sequence[Any]], sub_table))
        return

    # Case 2: Flat table
    _validate_single_table(cast(Mapping[str, Sequence[Any]], data))


def generate_netting_set_ids(
    *,
    n: int,
    none_probability: float = 0.2,
) -> list[int | None]:

    import numpy as np

    from riskquant.ingest_fixtures.credit.netting_set.minimal_netting_set_master import (
        CANONICAL_NETTING_SET_UIDS,
    )
    from riskquant.ingest_fixtures.rng import get_canonical_rng

    _rng: np.random.Generator = get_canonical_rng()

    # 1. Generate the remaining IDs from valid options
    ns_ids = _rng.choice(
        a=CANONICAL_NETTING_SET_UIDS,
        size=n,
        replace=True,
    ).tolist()

    # 2. Randomly assign None based on the specified probability
    none_mask = _rng.random(size=n) < none_probability
    for i in range(n):
        if none_mask[i]:
            ns_ids[i] = None

    # 3. Combine and shuffle (shuffle works in-place)
    _rng.shuffle(ns_ids)

    return ns_ids
