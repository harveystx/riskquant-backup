# src/riskquant/credit/exposure/evaluators/kernels/current_exposure_method.py
"""
Current Exposure Method (CEM) exposure model: kernel.

    NRC     = max(sum(MTM) - C, 0)                  net replacement cost
    GRC     = sum(max(MTM, 0))                      gross replacement cost (no C)
    NGR     = NRC / GRC                             netting-to-gross ratio
    Net_PFE = sum(gross_PFE) * (FLOOR + COEFF*NGR)  compressed add-on
    EAD     = NRC + Net_PFE

Collateral enters NRC only, never GRC, consistent with the Basel definition of
the netting-to-gross ratio. `gross_pfe` is scenario-independent, so its
per-set sum is taken once, outside the scenario loop; NRC / GRC / NGR are
per-scenario. Positional-or-keyword args (not keyword-only): Numba's `*,` parse
is unreliable, matching every other kernel here.

This kernel implements the Basel Current Exposure Method (CEM) for measuring
counterparty credit risk exposure on OTC derivative portfolios subject to
legally enforceable netting agreements.

## Regulatory background

CEM was introduced by the Basel Committee on Banking Supervision (BCBS) as the
standardized approach for calculating exposure at default (EAD) for derivative
contracts under the Basel II capital framework. Although CEM has largely been
superseded internationally by the Standardised Approach for Counterparty Credit
Risk (SA-CCR), it remains useful for historical analysis, legacy systems, and
comparison purposes.

The Basel netting adjustment applies only to qualifying legally enforceable
bilateral netting agreements; otherwise the gross add-on is used.

Primary regulatory references:

* Basel Committee on Banking Supervision (2006),
  *International Convergence of Capital Measurement and Capital Standards:
  A Revised Framework (Basel II)*, Annex 4, "Current Exposure Method".
* Basel Committee on Banking Supervision (2014),
  *The Standardised Approach for Measuring Counterparty Credit Risk Exposures
  (SA-CCR)*.

## Method

For each netting set and scenario:

```
NRC = max(sum(MTM) - C, 0)
```

where

* MTM is the mark-to-market value of each trade,
* C is recognized collateral posted against the netting set,
* NRC is the net replacement cost.

The gross replacement cost is

```
GRC = sum(max(MTM_i, 0))
```

where only positive trade MTMs contribute.

The Basel netting-to-gross ratio (NGR) is

```
NGR = NRC / GRC
```

with the convention that

```
GRC = 0  =>  NGR = 0
```

to avoid division by zero.

The gross potential future exposure (gross PFE) is supplied as an input and is
reduced for qualifying netting agreements using the Basel CEM compression
formula

```
Net PFE = Gross PFE × (0.4 + 0.6 × NGR)
```

Finally,

```
EAD = NRC + Net PFE
```

## Implementation notes

* Gross PFE is scenario-independent and is therefore summed once per netting
  set outside the scenario loop.
* NRC, GRC and NGR are evaluated independently for every scenario.
* Collateral affects NRC only and **not** GRC, consistent with the Basel CEM
  definition of the netting-to-gross ratio. This preserves the separation
  between collateral recognition and netting recognition.
* The implementation assumes that trade-level gross PFEs have already been
  computed according to the applicable supervisory add-on factors.
"""

from __future__ import annotations

import numba as nb
import numpy as np
import numpy.typing as npt

from riskquant.jit import parallel_njit

_F64_2D = nb.float64[:, ::1]
_F64_1D = nb.float64[::1]
_I64_1D = nb.int64[::1]


@parallel_njit(
    nb.void(
        _F64_2D,  # out
        _F64_2D,  # mtm
        _I64_1D,  # mtm_row_perm
        _I64_1D,  # ns_offsets
        _F64_1D,  # collateral
        _F64_1D,  # gross_pfe
        nb.float64,  # pfe_floor
        nb.float64,  # pfe_coefficient
    )
)
def _kernel_current_exposure_method(
    out: npt.NDArray[np.float64],  # (n_netting_sets, s_scenarios)
    mtm: npt.NDArray[np.float64],  # (M_total, s_scenarios)
    mtm_row_perm: npt.NDArray[np.int64],  # (M_view,)
    ns_offsets: npt.NDArray[np.int64],  # (n_netting_sets + 1,)
    collateral: npt.NDArray[np.float64],  # (n_netting_sets,)
    gross_pfe: npt.NDArray[np.float64],  # (M_view,) CSR order, scenario-free
    pfe_floor: float,  # NGR-compression floor (config; default 0.4)
    pfe_coefficient: float,  # NGR-compression slope (config; default 0.6)
) -> None:
    n_net_sets, s_scenarios = out.shape
    for n in nb.prange(n_net_sets):
        start = ns_offsets[n]
        end = ns_offsets[n + 1]
        coll = collateral[n]
        out_row = out[n]

        # 1. Gross PFE is scenario-independent, reduce once
        gross_pfe_set = 0.0
        for i in range(start, end):
            gross_pfe_set += gross_pfe[i]

        # 2. Allocate thread-local buffer for GRC.
        # For 10k scenarios, this is ~80KB and easily fits in L1/L2 cache.
        grc = np.zeros(s_scenarios, dtype=np.float64)

        # 3. Initialize out_row to zero to use it as our net_mtm accumulator
        for s in range(s_scenarios):
            out_row[s] = 0.0

        # 4. FAST PATH: Loop trades (outer) -> scenarios (inner)
        # This restores C-contiguous memory streaming along the scenario axis.
        for i in range(start, end):
            mtm_row = mtm[mtm_row_perm[i]]
            for s in range(s_scenarios):
                m = mtm_row[s]
                out_row[s] += m
                grc[s] += m * (m > 0.0)

        # 5. Final compression pass across the scenario vectors
        for s in range(s_scenarios):
            net_mtm = out_row[s]
            nrc = max(net_mtm - coll, 0.0)
            grc_s = grc[s]

            # NGR = NRC / GRC, zero-guard: GRC == 0 => NGR = 0
            ngr = nrc / grc_s if grc_s > 0.0 else 0.0

            net_pfe = gross_pfe_set * (pfe_floor + pfe_coefficient * ngr)
            out_row[s] = nrc + net_pfe
