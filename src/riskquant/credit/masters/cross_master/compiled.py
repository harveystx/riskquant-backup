# src/riskquant/credit/masters/cross_master/compiled.py
"""
The validated credit master aggregate.

`CompiledCreditMasters` is the single value that represents "a set of credit
masters that are mutually consistent". It does NOT compile anything — each master
is compiled independently by its own ingest edge (`compile_obligor_master`,
`compile_factor_universe`, `compile_netting_set_master`) and handed in already
built, so every master remains separately inspectable. What this type adds is the
cross-master invariant: it runs `validate_cross_masters` at construction and
fails fast, so an inconsistent set of masters cannot be represented at all.

This is deliberately NOT a pipeline/orchestrator. It owns exactly one
responsibility — the layered foreign-key relationships between already-compiled
masters — and leaves master compilation and engine construction explicit in the
caller. Downstream consumers (the JTD compile, a future engine builder) bind
against this aggregate instead of threading the individual masters and trusting
the caller to have remembered the cross-checks.

The netting-set master is optional: a pure default/loss workflow (obligors +
factor universe + externally supplied EADs) needs no netting sets, and the only
cross-master invariant defined today — the netting-set -> obligor foreign key —
simply does not apply when it is absent.
"""

from __future__ import annotations

__all__ = [
    "CompiledCreditMasters",
]

from dataclasses import dataclass

from riskquant.credit.masters.cross_master.validate import validate_cross_masters
from riskquant.credit.masters.factor_universe.compiled import (
    CompiledCreditFactorUniverse,
)
from riskquant.credit.masters.netting_set.compiled import CompiledNettingSetMaster
from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledCreditMasters:
    """
    A mutually-consistent set of compiled credit masters.

    Construction validates every defined cross-master invariant and raises on the
    first violation, so holding an instance is proof the masters agree. The
    constituent masters stay directly accessible as fields for inspection and for
    passing on to the model-compilation edges.

    Parameters
    ----------
    obligor_master
        Compiled obligor master; the single source of truth for PD/LGD and the
        obligor universe every other master and model resolves against.
    factor_universe
        Compiled credit factor-universe master; the resolution target for factor
        names at JTD compile time.
    netting_set_master
        Optional compiled netting-set master. When supplied, every netting set's
        owning obligor must exist on the obligor master.

    Raises
    ------
    ValueError
        If any cross-master invariant fails (e.g. a netting set references an
        obligor uid absent from the obligor master).
    """

    obligor_master: CompiledObligorCredit
    factor_universe: CompiledCreditFactorUniverse
    netting_set_master: CompiledNettingSetMaster | None = None

    def __post_init__(self) -> None:
        # Cross-master invariants. Only the netting-set -> obligor foreign key is
        # defined today; it is vacuous (and skipped) when no netting-set master is
        # present. Future layered checks (rating/sector taxonomy joins, parent
        # closure) extend `validate_cross_masters` and are picked up here for free.
        if self.netting_set_master is not None:
            validate_cross_masters(
                obligor_master=self.obligor_master,
                netting_set_master=self.netting_set_master,
            )
