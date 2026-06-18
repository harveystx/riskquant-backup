# src/riskquant/credit/jtd/blocks/type_alias.py
"""
JTD block type aliases.
"""

from __future__ import annotations

from typing import TypeAlias

from riskquant.credit.jtd.blocks.beta_binomial import (
    BetaBinomialJtdBlock,
)
from riskquant.credit.jtd.blocks.forced_default import (
    ForcedDefaultJtdBlock,
)
from riskquant.credit.jtd.blocks.independent_bernoulli import (
    IndependentBernoulliJtdBlock,
)
from riskquant.credit.jtd.blocks.mf_copula_gaussian import (
    MfGaussianCopulaJtdBlock,
)
from riskquant.credit.jtd.blocks.mf_copula_students_t import (
    MfStudentsTCopulaJtdBlock,
)
from riskquant.credit.jtd.blocks.sf_copula_gaussian import (
    SfGaussianCopulaJtdBlock,
)
from riskquant.credit.jtd.blocks.sf_copula_students_t import (
    SfStudentsTCopulaJtdBlock,
)
from riskquant.credit.jtd.blocks.stress_regime_bernoulli import (
    StressRegimeBernoulliJtdBlock,
)

JtdBlockTypeUnion: TypeAlias = (
    BetaBinomialJtdBlock
    | ForcedDefaultJtdBlock
    | IndependentBernoulliJtdBlock
    | MfGaussianCopulaJtdBlock
    | MfStudentsTCopulaJtdBlock
    | SfGaussianCopulaJtdBlock
    | SfStudentsTCopulaJtdBlock
    | StressRegimeBernoulliJtdBlock
)

SfCopulaJtdBlockTypeUnion: TypeAlias = (
    SfGaussianCopulaJtdBlock | SfStudentsTCopulaJtdBlock
)

MfCopulaJtdBlockTypeUnion: TypeAlias = (
    MfGaussianCopulaJtdBlock | MfStudentsTCopulaJtdBlock
)
