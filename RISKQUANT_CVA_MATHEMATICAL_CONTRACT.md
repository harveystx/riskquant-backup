# RiskQuant — Ultra-Tier Mathematical CVA Contract

**Status:** Canonical mathematical specification for the first CVA implementation

**Scope:** Unilateral counterparty Credit Valuation Adjustment (CVA), with an explicitly defined first-generation specialisation and controlled extensions

**Authority:** This document is the mathematical north star for all RiskQuant CVA work. Code, APIs, abstractions, kernels, tests, documentation, and future extensions shall conform to the mathematical contracts herein. Where implementation convenience conflicts with a mathematical distinction stated here, the mathematical distinction wins.

---

# 0. Purpose and doctrine

RiskQuant shall first implement a mathematically exact and independently verifiable **unilateral CVA calculation**.

The implementation is **not initially an “XVA engine”**. CVA shall exist as a standalone valuation adjustment with explicit seams that can later support collateral, wrong-way risk (WWR), bilateral default/DVA, funding, margin, capital, and other XVA quantities.

The fundamental doctrine is:

> **Do not abstract away a mathematical distinction merely because the resulting code is cleaner.**

And, more formally:

$$
\boxed{
\text{Definition}
\neq
\text{Specialisation}
\neq
\text{Discretisation}
\neq
\text{Estimator}
}
$$

Every simplification must be explicitly derived from the more general quantity it approximates.

## Non-negotiable implementation doctrine

1. Do not call a specialised approximation the general CVA calculation.
2. Do not use historical P&L as future exposure without proving measure and valuation compatibility.
3. Do not net trades after applying the positive-part operator; netting precedes exposure truncation.
4. Do not infer legal netting scope from counterparty identity alone.
5. Do not treat a CDS spread as a hazard rate by identity.
6. Do not introduce WWR through an arbitrary correlation multiplier.
7. Do not make CVA/DVA/FVA/MVA/KVA universally additive by construction.
8. Do not hide measure, numeraire, closeout, recovery, collateral, sign, time-bucket, or interpolation conventions inside numerical kernels.
9. Do not double-count counterparty credit in future clean valuation and again in CVA.
10. Prefer a small mathematically exact implementation over a broad abstraction whose semantics are ambiguous.
11. Boundaries establish and validate semantics; kernels perform declared mathematics.
12. Invalid mathematical inputs fail fast rather than being silently repaired into plausible-looking numbers.
13. Distinguish unconditional exposure expectations from survival-conditioned exposure expectations; equality between them is an explicit consequence of the first-generation independence assumption.
14. Distinguish the market/risk-factor filtration used for conditional survival from a filtration that already reveals the default event.
15. Treat the default-time discretisation rule as part of the numerical contract rather than an incidental implementation detail.

---

# 1. Mathematical north star

The most general object required by the first unilateral CVA formulation is the expected discounted loss caused by counterparty default:

$$
\boxed{
CVA_0
=
\mathbb E^{\mathbb Q}
\left[
D(0,\tau_C)\,
\mathcal L_{\tau_C}\,
\mathbf 1_{\{\tau_C\le T\}}
\right]
}
$$

where:

* $\mathbb Q$ is the chosen pricing measure associated with the valuation numeraire;
* $\tau_C$ is counterparty default time;
* $D(0,t)$ is the stochastic discount factor to the valuation currency/numeraire;
* $\mathcal L_t$ is the **actual counterparty-default loss on the netting set at time $t$ under the selected closeout model**;
* $T$ is the latest future date at which relevant closeout exposure may be non-zero for the selected netting set and valuation model.

This is the **economic definition**.

Everything else in this document is either:

* a definition of one component;
* a modelling specialisation;
* a numerical discretisation;
* a Monte Carlo/statistical estimator;
* or a controlled future extension.

---

# 2. First-generation production scope

The first production implementation shall be:

$$
\boxed{
\text{Unilateral, uncollateralised, deterministic-recovery, deterministic-discounting, independent-default CVA}
}
$$

with the following assumptions:

1. Only counterparty default is modelled.
2. Own default is excluded; there is no DVA.
3. No FVA, MVA, KVA, or other XVA adjustment is included.
4. Recovery is deterministic:

$$
R_C\in[0,1].
$$

5. LGD is deterministic:

$$
LGD=1-R_C.
$$

6. Discount factors are deterministic:

$$
DF(t)>0.
$$

7. Counterparty default is independent of the future market-state / exposure process in the precise sense specified in Section 9.
8. The first exposure model is **uncollateralised clean-MTM closeout with instantaneous closeout and no margin-period dynamics**.
9. Future clean valuation is supplied by a future-state valuation/simulation layer.
10. Credit is supplied by a risk-neutral cumulative default/survival curve or equivalent deterministic hazard representation.
11. Default is integrated on an explicit time grid using the documented bucket convention in Section 19.
12. All CVA amounts are expressed in one explicitly selected valuation currency/numeraire.
13. The time coordinate used by the CVA integrator is an explicitly defined year-fraction coordinate.

The first-generation continuous-time specialisation is:

$$
\boxed{
CVA_0
=
(1-R_C)
\int_0^T
DF(t)\,EE(t)\,dPD_C(t)
}
$$

where:

$$
EE(t)
=
\mathbb E^{\mathbb Q}
\left[
(V_t^{\mathrm{clean}})^+
\right].
$$

This is a **specialisation plus a continuous-time representation**, not the general definition.

The canonical first bucketed numerical implementation is:

$$
\boxed{
CVA_0^{(N)}
=
(1-R_C)
\sum_{i=1}^{N}
DF_i\,EE_i\,
\left[
S_C(t_{i-1})-S_C(t_i)
\right]
}
$$

under the endpoint default/exposure convention specified later.

This is a **time-discretised numerical approximation**.

---

# 3. Terminology and mathematical status

The following table is normative.

| Object                                                                        | Mathematical status                                                    |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| $\mathbb E^\mathbb Q[D(0,\tau_C)\mathcal L_{\tau_C}\mathbf1_{{\tau_C\le T}}]$ | **General economic definition**                                        |
| $\mathcal L_t$                                                                | **Actual counterparty-default loss under the selected closeout model** |
| $U_t$                                                                         | **Unsecured closeout amount**                                          |
| $E_t=U_t^+$                                                                   | **Positive unsecured exposure specialisation**                         |
| $\mathcal L_t=LGD_tE_t$                                                       | **Recovery/closeout specialisation**                                   |
| $EE(t)=\mathbb E^\mathbb Q[E_t]$                                              | **Unconditional expected-exposure definition**                         |
| $EE^{surv}(t)=\mathbb E^\mathbb Q[E_t\mid\tau_C>t]$                           | **Survival-conditioned expected exposure**                             |
| $EE^{surv}(t)=EE(t)$                                                          | **First-generation independence consequence**                          |
| $CVA=\int DF,EE,dPD$                                                          | **Deterministic/independent factorisation**                            |
| $\sum_i DF_iEE_i\Delta PD_i$                                                  | **Time-discretised numerical approximation**                           |
| $\widehat{EE}*i=\sum_sw_sE*{s,i}$                                             | **Monte Carlo estimator**                                              |
| $EE\times PD\times LGD$                                                       | **Not a general identity**                                             |

No implementation may conflate these levels without an explicit derivation.

---

# 4. Economic interpretation

CVA is a **valuation adjustment caused by counterparty default risk**.

It is not inherently:

* VaR;
* Expected Shortfall;
* PFE;
* a stress loss;
* historical P&L;
* today’s positive MTM multiplied by a scalar PD;
* or an arbitrary $EE\times PD\times LGD$ product.

Under unilateral CVA and the conventions in this document:

$$
CVA_0\ge0,
$$

and the CVA-adjusted value is:

$$
\boxed{
V_0^{\mathrm{CVA}}
=
V_0^{\mathrm{clean}}
-
CVA_0.
}
$$

Hence:

$$
V_0^{\mathrm{CVA}}
\le
V_0^{\mathrm{clean}}.
$$

This is a model invariant under the first-generation assumptions.

---

# 5. Measure and numeraire contract

CVA is a pricing quantity.

The pricing measure is:

$$
\boxed{
\mathbb Q^N
=
\text{pricing measure associated with the selected valuation numeraire }N.
}
$$

For the first-generation implementation, the selected valuation currency/numeraire shall be fixed and externally inspectable.

All probabilistic inputs used in the CVA calculation must have an explicit measure/numeraire interpretation, including:

* future-state scenarios;
* expected exposure;
* survival probabilities;
* default probabilities;
* hazard rates;
* stochastic discount-factor inputs, where applicable.

The relationship between the pricing measure and discounting shall not be left implicit.

For the first-generation deterministic-discounting implementation:

$$
D(0,t)=DF(t),
$$

where $DF(t)$ is deterministic in the selected valuation currency/numeraire.

The following substitutions are prohibited unless an explicit model transformation is supplied:

$$
\mathbb P\text{-exposure}
\not\equiv
\mathbb Q^N\text{-exposure},
$$

$$
PD^{\mathbb P}
\not\equiv
PD^{\mathbb Q^N}.
$$

Historical scenario frequencies or historical default rates shall not silently become pricing probabilities.

For a different numeraire or valuation currency, the corresponding pricing-measure and discounting conventions must be explicitly re-established rather than assumed to be interchangeable.

---

# 6. Clean future valuation contract

For scenario $s$ and future date $t_i$:

$$
\boxed{
V_{s,i}^{\mathrm{clean}}
=
V^{\mathrm{clean}}(X_{s,i},t_i)
}
$$

where $X_{s,i}$ is the future market-factor state.

The clean valuation is from **our perspective**:

$$
V>0
\Rightarrow
\text{counterparty owes us value},
$$

$$
V<0
\Rightarrow
\text{we owe the counterparty value}.
$$

## Clean-value purity invariant

The future clean valuation shall contain **no counterparty credit adjustment**.

It shall not silently include:

* CVA;
* DVA;
* FVA;
* counterparty-credit spread adjustments;
* recovery/default-loss effects;
* collateral-credit adjustments.

This prevents double counting:

$$
\boxed{
\text{clean valuation contains no counterparty credit adjustment}.
}
$$

The valuation layer is responsible for product/market valuation only. Counterparty-default loss belongs downstream.

---

# 7. Future valuation is not historical P&L

Historical market-risk machinery commonly computes:

$$
\text{historical scenario}
\rightarrow
\Delta V
\quad\text{or}\quad
P\&L.
$$

CVA requires:

$$
\boxed{
(X_{s,i},t_i)
\rightarrow
V_{s,i}^{\mathrm{clean}}.
}
$$

These are different mathematical objects.

The following equivalence is prohibited:

$$
\boxed{
\text{historical P\&L scenario}
\not\equiv
\text{future risk-neutral exposure scenario}.
}
$$

Existing RiskQuant scenario infrastructure may be reused only if its semantics prove compatible with:

1. future time evolution;
2. the required pricing measure;
3. future clean valuation rather than historical P&L;
4. the exposure time grid;
5. scenario weighting.

Structural resemblance is insufficient.

---

# 8. Numeraire, currency, and units contract

All values aggregated within a netting set must be represented in a common valuation currency/numeraire **according to the contractual/modelled closeout conversion convention** before mathematical netting.

For a selected netting-set valuation currency/numeraire $N$:

$$
V_{\mathcal N,t}^{(N)}
=
\sum_{j\in\mathcal N}
V_{j,t}^{(N)}.
$$

Currency conversion is part of valuation/closeout semantics, not merely an implementation utility.

The first-generation implementation shall use one explicitly selected aggregation currency/numeraire.

For more general multi-currency structures, the relevant contractual closeout currency, FX conversion rule, and collateral currency treatment must be explicitly modelled before exposure is determined.

## Dimensional invariants

$$
[V]
=
[E]
=
[\mathcal L]
=
[CVA]
=
\text{currency},
$$

$$
[DF]
=
[PD]
=
[LGD]
=
1,
$$

and where applicable:

$$
[\lambda]
=
1/\text{time}.
$$

Therefore:

$$
[DF\times EE\times\Delta PD]
=
\text{currency}
$$

and:

$$
[DF\times E\times\lambda\times dt]
=
\text{currency}.
$$

A dimensionally inconsistent implementation is invalid regardless of numerical plausibility.

---

# 9. Formal independence contract

Let:

$$
\mathcal F_t^M
$$

denote the filtration generated by the future market/risk-factor information relevant to exposure.

For the first-generation model, counterparty survival shall satisfy:

$$
\boxed{
\mathbb Q^N
\left(
\tau_C>t
\mid
\mathcal F_t^M
\right)
=
S_C(t)
}
$$

where $S_C(t)$ is deterministic.

This states that the conditional survival probability does not depend on the future market state driving exposure.

Equivalently, under the generation-one assumptions, the default event carries no incremental dependence on the future exposure state.

Therefore:

$$
\boxed{
\mathbb E^{\mathbb Q^N}
[
E_t\mid\tau_C>t
]
=
\mathbb E^{\mathbb Q^N}[E_t]
}
$$

and hence:

$$
\boxed{
EE^{surv}(t)=EE(t).
}
$$

This equality is a **model consequence**, not a universal definition of expected exposure.

Do not merely pass a boolean named `independent=True` without the mathematical meaning represented above.

---

# 10. General default-loss contract

The general default loss is:

$$
\boxed{
\mathcal L_t
=
\text{counterparty-default loss on the netting set at }t.
}
$$

The general CVA definition is therefore:

$$
\boxed{
CVA_0
=
\mathbb E^{\mathbb Q^N}
\left[
D(0,\tau_C)
\mathcal L_{\tau_C}
\mathbf1_{\{\tau_C\le T\}}
\right].
}
$$

This deliberately separates the economic definition from any particular recovery or closeout representation.

---

# 11. Closeout and unsecured exposure contract

Define:

$$
\boxed{
U_t
=
\text{unsecured closeout amount at time }t.
}
$$

Positive unsecured exposure is:

$$
\boxed{
E_t
=
U_t^+.
}
$$

Collateral shall be denoted $M_t$ throughout this contract.

Under the first-generation closeout specialisation:

$$
\boxed{
U_t
=
V_t^{\mathrm{clean}}
-
M_t.
}
$$

The corresponding positive unsecured exposure is:

$$
\boxed{
E_t
=
(V_t^{\mathrm{clean}}-M_t)^+.
}
$$

For the first uncollateralised implementation:

$$
\boxed{
M_t=0,
}
$$

so:

$$
\boxed{
E_t=(V_t^{\mathrm{clean}})^+.
}
$$

The positive-part operation occurs **after netting and closeout adjustments**.

---

# 12. First-generation closeout specialisation

The first-generation engine shall use the following explicit closeout model:

1. The reference valuation is the future clean/risk-free replacement value.
2. Counterparty default causes immediate closeout at the default time approximation.
3. The unsecured amount is clean netting-set value less the modelled collateral balance.
4. Only positive unsecured value generates counterparty loss.
5. Recovery applies to that positive unsecured value through a deterministic LGD.
6. There is no margin-period-of-risk exposure evolution.
7. There is no liquidation-cost term.
8. There is no separate stochastic closeout adjustment.
9. The first production model is uncollateralised, so $M_t=0$.

Therefore, at default under the first-generation model:

$$
\boxed{
\mathcal L_t
=
(1-R_C)
(V_t^{\mathrm{clean}})^+.
}
$$

This equation is a **closeout/recovery specialisation** of the general loss $\mathcal L_t$.

It is not a universal identity for arbitrary legal closeout models.

Future closeout models may replace this mapping without changing the general CVA definition.

---

# 13. Recovery and LGD contract

The general model permits stochastic recovery:

$$
R_t\in[0,1],
$$

with:

$$
LGD_t=1-R_t.
$$

Under the simple recovery specialisation:

$$
\boxed{
\mathcal L_t
=
LGD_tE_t.
}
$$

Under first-generation deterministic recovery:

$$
R_t=R_C,
$$

$$
LGD=1-R_C,
$$

so:

$$
\boxed{
\mathcal L_t
=
(1-R_C)E_t.
}
$$

Recovery must remain conceptually distinct from exposure and default probability.

Do not embed recovery inside a “PD” object unless the object’s semantics explicitly state that it is no longer a pure default-probability quantity.

---

# 14. Netting-set contract

CVA exposure is measured at the legally enforceable netting-set level.

The contractual layer, not the CVA numerical kernel, establishes membership of $\mathcal N$.

The normative mathematical dependency is:

$$
\boxed{
\text{legal enforceability}
\rightarrow
\text{netting-set membership}
\rightarrow
\text{valuation aggregation}.
}
$$

For scenario $s$ and time $t_i$:

$$
\boxed{
V_{\mathcal N,s,i}^{\mathrm{clean}}
=
\sum_{j\in\mathcal N}
V_{j,s,i}^{\mathrm{clean},(N)}.
}
$$

Only after that classification and aggregation does exposure truncation occur:

$$
E_{\mathcal N,s,i}
=
\left(
V_{\mathcal N,s,i}^{\mathrm{clean}}
-
M_{\mathcal N,s,i}
\right)^+.
$$

In general:

$$
\boxed{
\left(\sum_jV_{j,t}\right)^+
\ne
\sum_j(V_{j,t})^+.
}
$$

This mathematical nonlinearity is exactly why netting scope must be established before exposure truncation.

Counterparty identity alone shall not define $\mathcal N$.

The contractual hierarchy shown conceptually is:

$$
\boxed{
\text{legal agreement data}
\rightarrow
\text{enforceable netting scope}
\rightarrow
\text{collateral/closeout terms}
\rightarrow
\text{trades}.
}
$$

This is a semantic dependency, not a claim that every legal structure has one universal object hierarchy.

---

# 15. Collateral contract

Collateral is contractual credit protection, not an arbitrary scalar adjustment.

The first simple collateral representation uses:

$$
U_t
=
V_t^{\mathrm{clean}}
-
M_t
$$

subject to an explicitly fixed sign convention for $M_t$.

The implementation must document whether positive $M_t$ means, for example:

* collateral held by us;

or:

* collateral posted by us.

The sign convention must be internally consistent across valuation, netting, exposure, and reporting.

Future collateral extensions may need to represent:

* variation margin;
* initial margin / independent amount;
* threshold;
* minimum transfer amount;
* margin frequency;
* margin call/settlement lag;
* collateral currency;
* haircut;
* collateral remuneration;
* margin period of risk;
* collateral return on default;
* closeout / settlement conventions.

These need not be implemented in generation one.

## Initial-margin warning

Initial margin shall not be treated as an automatically subtractable collateral balance merely because it is economically “collateral”.

Its legal availability, segregation, return, recovery treatment, and closeout role must be explicitly modelled before it enters the exposure formula.

---

# 16. Exposure definition

For the first uncollateralised model:

$$
\boxed{
E_t
=
(V_t^{\mathrm{clean}})^+.
}
$$

For the simple collateralised model:

$$
\boxed{
E_t
=
(V_t^{\mathrm{clean}}-M_t)^+.
}
$$

This exposure is the positive part of the unsecured closeout amount.

It is not “the tail” in the PFE/ES sense; it is the entire positive region of the unsecured distribution.

A precise interpretation is:

> CVA depends on the positive part of the future unsecured closeout distribution, weighted by the timing and probability of counterparty default.

---

# 17. Expected Exposure contract

The unconditional expected exposure is:

$$
\boxed{
EE(t)
=
\mathbb E^{\mathbb Q^N}[E_t].
}
$$

The survival-conditioned expected exposure is:

$$
\boxed{
EE^{surv}(t)
=
\mathbb E^{\mathbb Q^N}
[
E_t\mid\tau_C>t
].
}
$$

These are distinct mathematical quantities.

Under the first-generation independence assumption:

$$
\boxed{
EE^{surv}(t)=EE(t).
}
$$

Under genuine WWR, they are generally not equal.

Under equally weighted Monte Carlo:

$$
\boxed{
\widehat{EE}_i
=
\frac1S
\sum_{s=1}^{S}
E_{s,i}.
}
$$

For general probability weights:

$$
\boxed{
\widehat{EE}_i
=
\sum_{s=1}^{S}
w_sE_{s,i},
\qquad
w_s\ge0,
\qquad
\sum_sw_s=1.
}
$$

The expectation is the mathematical quantity; the hatted expression is the estimator.

The scenario law and scenario weights jointly define the probability measure represented by the estimator.

---

# 18. Expected Exposure versus PFE

Expected exposure:

$$
EE(t)
=
\mathbb E[V_t^+].
$$

Exposure quantile/PFE at confidence level $q$:

$$
PFE_q(t)
=
Q_q(V_t^+).
$$

These are different mathematical objects.

CVA uses expected loss weighted through the default-time distribution. PFE is a tail exposure statistic at a future date.

PFE shall not replace EE in the CVA contract unless an explicit approximation has been independently specified and validated.

Terminology such as “EPE” may be used only with an explicit definition because institutional/regulatory conventions differ.

Internally, RiskQuant should prefer unambiguous names such as:

* `expected_exposure`;
* `expected_exposure_profile`;
* `survival_conditioned_expected_exposure`.

---

# 19. Default-time contract and bucket convention

Counterparty default time is:

$$
\tau_C.
$$

The event that default occurs in bucket $(t_{i-1},t_i]$ is:

$$
\{t_{i-1}<\tau_C\le t_i\}.
$$

The continuous mathematical object refers to exposure at the actual default time:

$$
E_{\tau_C}.
$$

A bucketed implementation must explicitly state how the interval default mass is associated with exposure and discounting.

## First-generation default-bucket convention

RiskQuant shall use the following endpoint approximation unless a later model explicitly replaces it:

> Defaults in $(t_{i-1},t_i]$ are represented by the exposure and discount factor associated with the **pre-closeout endpoint convention at $t_i$**.

For a smooth exposure process this is represented as:

$$
EE_i=EE(t_i),
$$

$$
DF_i=DF(t_i).
$$

Then:

$$
CVA_i
=
LGD_iDF_iEE_i\Delta PD_i.
$$

This is a **numerical approximation**, not an assertion that all defaults in the bucket literally occur at $t_i$.

## Discontinuities

Where the clean valuation has a contractual discontinuity at $t_i$, such as a cashflow, exercise, fixing, maturity, or settlement event, the exposure value used by the endpoint convention shall be explicitly identified as the relevant **pre-closeout/pre-discontinuity** value required by the chosen closeout approximation.

The engine shall not silently choose between:

$$
V(t_i^-)
\quad\text{and}\quad
V(t_i^+).
$$

A numerical implementation must document which side of any discontinuity belongs to the closeout approximation.

---

# 20. Survival/default probability contract

Define unconditional risk-neutral survival probability:

$$
\boxed{
S_C(t)
=
\mathbb Q^N(\tau_C>t).
}
$$

Then cumulative default probability:

$$
\boxed{
PD_C(t)
=
1-S_C(t).
}
$$

For bucket $(t_{i-1},t_i]$:

$$
\boxed{
\Delta PD_i
=
PD_C(t_i)-PD_C(t_{i-1})
=
S_C(t_{i-1})-S_C(t_i).
}
$$

The first-generation curve invariants are:

$$
S_C(0)=1,
$$

$$
0\le S_C(t)\le1,
$$

and for increasing time:

$$
S_C(t_i)\le S_C(t_{i-1}).
$$

Therefore:

$$
\boxed{
\Delta PD_i\ge0.
}
$$

The total default mass through $T$ is:

$$
\sum_{i=1}^{N}\Delta PD_i
=
S_C(0)-S_C(T)
=
1-S_C(T).
$$

Invalid survival curves must fail fast or be rejected explicitly.

---

# 21. Hazard-rate contract

For a deterministic hazard/intensity curve $\lambda_C(t)$ that generates an absolutely continuous survival law:

$$
\boxed{
S_C(t)
=
\exp
\left(
-\int_0^t\lambda_C(u)\,du
\right)
}
$$

and:

$$
\boxed{
\lambda_C(t)
=
-\frac{d}{dt}\log S_C(t)
}
$$

where the derivative exists in the relevant sense.

Then:

$$
\boxed{
\frac{dPD_C(t)}{dt}
=
\lambda_C(t)S_C(t).
}
$$

A credit curve interface must distinguish:

* survival probability $S_C(t)$;
* cumulative default probability $PD_C(t)$;
* bucket default mass $\Delta PD_i$;
* deterministic hazard rate $\lambda_C(t)$.

These are related quantities, not aliases.

The hazard representation is applicable only where the default-time distribution admits the corresponding intensity/density representation.

If a model contains discrete default-time atoms or other non-absolutely-continuous components, the survival/default-mass representation remains valid while the ordinary $\lambda(t)S(t),dt$ representation is not by itself sufficient.

---

# 22. Credit-curve calibration contract

The conceptual market-data chain is:

$$
\boxed{
\text{CDS market data}
\rightarrow
\text{credit calibration}
\rightarrow
S_C(t),PD_C(t),\lambda_C(t)
\rightarrow
CVA.
}
$$

A CDS spread is **not** a hazard rate by identity.

Calibration can depend on:

* recovery;
* discounting;
* premium/accrual conventions;
* default settlement conventions;
* calibration instruments;
* curve parametrisation;
* interpolation;
* numerical integration.

The first RiskQuant CVA implementation may consume an already calibrated survival/default curve.

This is preferable to embedding an unvalidated CDS bootstrap merely to complete the feature.

---

# 23. Deterministic/independent CVA derivation

Under the first-generation assumptions:

* deterministic discounting;
* deterministic LGD;
* deterministic survival/default curve;
* market exposure/default independence;
* uncollateralised clean-MTM closeout;

we have:

$$
\mathcal L_t
=
(1-R_C)
(V_t^{\mathrm{clean}})^+.
$$

The general default-loss expectation specialises to:

$$
\boxed{
CVA_0
=
(1-R_C)
\int_0^T
DF(t)
EE(t)
\,dPD_C(t).
}
$$

The derivation depends on the first-generation independence condition:

$$
\mathbb Q^N(\tau_C>t\mid\mathcal F_t^M)
=
S_C(t),
$$

which gives:

$$
EE^{surv}(t)=EE(t).
$$

Equivalently, using a deterministic hazard representation when the corresponding intensity exists:

$$
\boxed{
CVA_0
=
(1-R_C)
\int_0^T
DF(t)
EE(t)
\lambda_C(t)S_C(t)\,dt.
}
$$

This is the canonical first-generation continuous-time CVA contract.

---

# 24. Discrete CVA integration contract

Given:

$$
0=t_0<t_1<\cdots<t_N=T,
$$

and endpoint convention:

$$
EE_i=EE(t_i),
$$

$$
DF_i=DF(t_i),
$$

then:

$$
\boxed{
CVA_0^{(N)}
=
(1-R_C)
\sum_{i=1}^{N}
DF_iEE_i
\Delta PD_i
}
$$

where:

$$
\Delta PD_i
=
S_C(t_{i-1})-S_C(t_i).
$$

Bucket contribution:

$$
\boxed{
CVA_i
=
(1-R_C)DF_iEE_i\Delta PD_i.
}
$$

Cumulative CVA:

$$
CVA_{\le i}
=
\sum_{j=1}^{i}
CVA_j.
$$

Aggregate:

$$
\boxed{
CVA
=
\sum_{i=1}^{N}
CVA_i.
}
$$

The implementation shall expose the profile used to obtain the scalar result.

---

# 25. Discretisation status and numerical error

The bucketed formula is not the continuous definition. It introduces a time-discretisation approximation:

$$
CVA_0^{(N)}
\approx
CVA_0.
$$

The total numerical error can arise from several distinct sources:

1. **valuation-model error** in $V_{s,i}^{\mathrm{clean}}$;
2. **time-grid/discretisation error**;
3. **Monte Carlo sampling error**;
4. **curve interpolation/calibration error**;
5. **numerical integration error**;
6. **proxy/regression error**, where applicable.

Validation shall distinguish these sources rather than attributing every discrepancy to “Monte Carlo noise”.

A numerical implementation shall distinguish mathematical identities from finite-precision acceptance tolerances.

For example:

$$
CVA=\sum_iCVA_i
$$

is a mathematical identity, while a production implementation may only verify:

$$
\left|
CVA-\sum_iCVA_i
\right|
\le
\varepsilon_{\text{reconcile}}.
$$

The tolerance must be explicit.

---

# 26. Discounting contract

For deterministic discounting:

$$
\boxed{
DF_i=DF(0,t_i)>0.
}
$$

The general definition uses the stochastic discount factor:

$$
D(0,\tau_C).
$$

If rates are stochastic, the correct object is joint:

$$
\mathbb E^{\mathbb Q^N}
\left[
D_t
\mathcal L_t
\mathbf1_{\{\tau_C\in dt\}}
\right].
$$

Under a suitable intensity representation this may be written schematically as:

$$
\mathbb E^{\mathbb Q^N}
\left[
D_t
\mathcal L_t
\lambda_C(t)
G_t
\,dt
\right],
$$

where $G_t$ is the conditional survival process defined in Section 28.

In general:

$$
\mathbb E[D_tE_t]
\ne
\mathbb E[D_t]\mathbb E[E_t].
$$

Therefore the first-generation factorisation using deterministic $DF_t$ shall not be baked into a general stochastic-discounting abstraction.

---

# 27. Time-coordinate contract

The continuous variable $t$ shall be represented by a defined year-fraction coordinate.

The mapping:

$$
\text{calendar date}
\rightarrow
t
$$

must use an explicitly selected day-count/time convention.

The same mathematical time coordinate must be used consistently when evaluating:

* hazard rates;
* integrated hazard;
* discount factors;
* interpolation coordinates;
* time-grid spacing;
* numerical integration.

A hazard rate measured in one time basis shall not be multiplied by a time increment expressed in another without an explicit conversion.

Therefore:

$$
[\lambda]=1/\text{year-fraction unit}
$$

must be consistent with:

$$
dt.
$$

---

# 28. Wrong-way risk contract

WWR exists when exposure and counterparty credit quality are dependent.

Let:

$$
\mathcal F_t^M
$$

denote the market/risk-factor filtration driving exposure.

Let:

$$
\mathcal F_t^{C}
$$

denote credit-model information beyond the market filtration where applicable.

The relevant conditional survival quantity shall be defined with respect to a filtration that does **not already reveal the default event**.

Therefore, for a stochastic intensity/Cox-style construction, define:

$$
\boxed{
G_t
=
\mathbb Q^N
\left(
\tau_C>t
\mid
\mathcal F_t^M
\vee
\mathcal F_t^{C,\mathrm{pre-default}}
\right).
}
$$

For a pure market-conditioned credit model this may reduce to:

$$
\boxed{
G_t
=
\mathbb Q^N
\left(
\tau_C>t
\mid
\mathcal F_t^M
\right).
}
$$

The key distinction is:

$$
\boxed{
G_t
\neq
\mathbb Q^N(\tau_C>t\mid\text{information that already reveals whether default occurred}).
}
$$

Otherwise the conditional survival probability collapses to the realised indicator:

$$
\mathbf1_{\{\tau_C>t\}}.
$$

Under an intensity/Cox construction with pre-default intensity $\lambda_C(t)$:

$$
\boxed{
G_t
=
\exp
\left(
-\int_0^t\lambda_C(u)\,du
\right)
}
$$

where $\lambda_C$ may depend on the evolving market/credit state.

The corresponding joint CVA representation is:

$$
\boxed{
CVA_0
=
\mathbb E^{\mathbb Q^N}
\left[
\int_0^T
D_t
\mathcal L_t
\lambda_C(t)
G_t
\,dt
\right].
}
$$

Under the simple recovery specialisation:

$$
\mathcal L_t
=
LGD_tE_t,
$$

so equivalently:

$$
\boxed{
CVA_0
=
\mathbb E^{\mathbb Q^N}
\left[
\int_0^T
D_t
LGD_t
E_t
\lambda_C(t)
G_t
\,dt
\right].
}
$$

There is **no additional multiplication by $E_t$** once $\mathcal L_t$ is defined as the loss amount.

The independent representation:

$$
EE_t\times\lambda_C(t)S_C(t)
$$

is not valid under genuine WWR in general.

## WWR architectural rule

WWR shall be represented through an explicit **joint exposure-credit law** or equivalent conditional model.

It shall not be implemented by multiplying independent CVA by an arbitrary correlation scalar.

A parameter called `correlation` is not, by itself, a mathematically specified WWR model.

---

# 29. Monte Carlo contract

Let:

$$
V_{s,i}^{\mathrm{clean}}
=
V^{\mathrm{clean}}(X_{s,i},t_i)
$$

for scenarios $s=1,\ldots,S$ and dates $i=1,\ldots,N$.

First-generation exposure:

$$
\boxed{
E_{s,i}
=
\max
\left(
V_{s,i}^{\mathrm{clean}},
0
\right).
}
$$

Equal-weight estimator:

$$
\boxed{
\widehat{EE}_i
=
\frac1S
\sum_{s=1}^{S}
E_{s,i}.
}
$$

Weighted estimator:

$$
\boxed{
\widehat{EE}_i
=
\sum_{s=1}^{S}
w_sE_{s,i}
}
$$

with:

$$
w_s\ge0,
\qquad
\sum_sw_s=1.
$$

The estimator must preserve explicit scenario/date semantics.

For a future-state array, the semantic shape shall be documented, for example:

$$
[S\times N]
$$

for scenario $\times$ time.

A transpose must never be capable of silently converting date axis into scenario axis.

---

# 30. Scenario measure and weighting contract

The scenario generator must state:

1. the probability measure represented by the generated scenarios;
2. whether that measure is the selected pricing measure $\mathbb Q^N$;
3. the interpretation of any non-uniform scenario weights;
4. whether weights are part of the sampling estimator or an importance-sampling correction.

For first-generation standard Monte Carlo:

$$
X_s\sim\mathbb Q^N,
$$

and equal weights are admissible:

$$
w_s=\frac1S.
$$

For weighted simulation:

$$
\widehat{EE}_i
=
\sum_sw_sE_{s,i},
$$

the weights and scenario-generation law together must define an unbiased or otherwise explicitly characterised estimator of the target pricing expectation.

Historical scenario probabilities shall not be silently reused merely because their numerical weights sum to one.

---

# 31. Scenario state versus portfolio valuation

The engine must preserve the distinction:

$$
X_{s,i}
\rightarrow
V_{s,i}^{\mathrm{clean}}
\rightarrow
E_{s,i}.
$$

Market-factor scenarios are not themselves exposures.

A risk-factor shock is not automatically a future portfolio value.

A scenario generator must not be allowed to masquerade as a valuation engine merely because it outputs arrays of the correct shape.

---

# 32. Netting before positive part: exact requirement

For netting set $\mathcal N$:

$$
V_{\mathcal N,s,i}^{\mathrm{clean}}
=
\sum_{j\in\mathcal N}
V_{j,s,i}^{\mathrm{clean}}.
$$

Then:

$$
E_{\mathcal N,s,i}
=
\left(
V_{\mathcal N,s,i}^{\mathrm{clean}}
-
M_{\mathcal N,s,i}
\right)^+.
$$

Never calculate:

$$
\sum_{j\in\mathcal N}
(V_{j,s,i}^{\mathrm{clean}})^+
$$

as the exposure of a legally netted set.

The correct operation order is:

$$
\boxed{
\text{future market state}
\rightarrow
\text{clean valuation}
\rightarrow
\text{numeraire/FX conversion}
\rightarrow
\text{legal netting}
\rightarrow
\text{collateral/closeout}
\rightarrow
\text{positive part}
\rightarrow
\text{scenario aggregation}
\rightarrow
\text{credit integration}.
}
$$

---

# 33. First-generation deterministic reference calculation

Given:

* dates $0=t_0<t_1<\cdots<t_N=T$;
* clean future valuations $V_{s,i}^{\mathrm{clean}}$;
* deterministic $DF_i$;
* deterministic recovery $R_C$;
* deterministic survival curve $S_i$;
* equally weighted scenarios;

calculate:

### Step 1 — exposure

$$
E_{s,i}
=
\max(V_{s,i}^{\mathrm{clean}},0).
$$

### Step 2 — expected exposure

$$
EE_i
=
\frac1S
\sum_sE_{s,i}.
$$

### Step 3 — default mass

$$
\Delta PD_i
=
S_{i-1}-S_i.
$$

### Step 4 — LGD

$$
LGD=1-R_C.
$$

### Step 5 — bucket CVA

$$
CVA_i
=
LGD\,DF_i\,EE_i\,\Delta PD_i.
$$

### Step 6 — aggregate

$$
\boxed{
CVA
=
\sum_iCVA_i.
}
$$

This is the canonical golden oracle for the first-generation CVA integrator.

---

# 34. Output profile contract

The engine shall expose at least:

$$
\boxed{
\left\{
t_i,
EE_i,
S_i,
\Delta PD_i,
DF_i,
LGD_i,
CVA_i
\right\}_{i=1}^{N}
}
$$

and the aggregate:

$$
CVA
=
\sum_iCVA_i.
$$

Recommended semantic fields:

| Field                             | Mathematical meaning |
| --------------------------------- | -------------------- |
| `date`                            | $t_i$                |
| `expected_exposure`               | $EE_i$               |
| `survival_probability`            | $S_C(t_i)$           |
| `incremental_default_probability` | $\Delta PD_i$        |
| `discount_factor`                 | $DF_i$               |
| `lgd`                             | $LGD_i$              |
| `cva_contribution`                | $CVA_i$              |
| `cva_cumulative`                  | $\sum_{j\le i}CVA_j$ |

Exact storage technology is an implementation concern. Mathematical semantics are not.

---

# 35. Mathematical invariants

The first-generation implementation shall enforce or test:

## Exposure

$$
EE_i\ge0.
$$

## Survival

$$
S_0=1,
$$

$$
0\le S_i\le1,
$$

$$
S_i\le S_{i-1}.
$$

## Default mass

$$
\Delta PD_i\ge0.
$$

## Total default mass

$$
0\le
\sum_i\Delta PD_i
\le1,
$$

and:

$$
\sum_i\Delta PD_i
=
1-S_N
$$

within the declared numerical tolerance.

## Recovery/LGD

$$
0\le R_C\le1,
$$

$$
0\le LGD\le1.
$$

## Discounting

$$
DF_i>0.
$$

## Bucket CVA

Under first-generation assumptions:

$$
CVA_i\ge0.
$$

## Aggregate CVA

$$
CVA\ge0.
$$

## Adjusted value

$$
V_0^{CVA}
=
V_0^{clean}
-
CVA
\le
V_0^{clean}.
$$

---

# 36. Monotonicity properties

Under the first-generation model, holding all other quantities fixed:

## Exposure monotonicity

If:

$$
EE_i^{(A)}
\le
EE_i^{(B)}
\quad\forall i,
$$

then:

$$
\boxed{
CVA^{(A)}
\le
CVA^{(B)}.
}
$$

## LGD monotonicity

If:

$$
LGD_A
\le
LGD_B,
$$

then:

$$
\boxed{
CVA_A
\le
CVA_B.
}
$$

## Default-mass monotonicity

If:

$$
\Delta PD_i^{(A)}
\le
\Delta PD_i^{(B)}
\quad\forall i,
$$

then:

$$
\boxed{
CVA^{(A)}
\le
CVA^{(B)}.
}
$$

These are suitable for property-based validation.

---

# 37. Additivity boundary

CVA is additive across separately modelled valuation units only when the units are genuinely separate under the contractual/model scope.

For disjoint netting sets $\mathcal N_k$ evaluated independently:

$$
\boxed{
CVA_{portfolio}
=
\sum_k
CVA_{\mathcal N_k}.
}
$$

This does **not** imply trade-level additivity within a netting set, because:

$$
\left(\sum_jV_j\right)^+
\ne
\sum_jV_j^+
$$

in general.

Do not expose a generic “trade CVA sum” abstraction unless its meaning is explicitly defined as a decomposition, not as the primitive economic calculation.

---

# 38. Golden / exact sanity cases

The implementation shall support exact or near-exact tests.

## Case A — zero exposure

If:

$$
V_t^{clean}\le0
\quad\forall t,
$$

then:

$$
\boxed{
CVA=0.
}
$$

## Case B — zero default probability

If:

$$
PD_C(T)=0,
$$

then:

$$
\boxed{
CVA=0.
}
$$

## Case C — zero LGD

If:

$$
R_C=1,
$$

then:

$$
\boxed{
CVA=0.
}
$$

## Case D — constant exposure, unit discounting

If:

$$
EE(t)=E_0,
\qquad
DF(t)=1,
$$

then:

$$
\boxed{
CVA
=
LGD\,E_0\,PD_C(T).
}
$$

## Case E — certain default by maturity

If:

$$
PD_C(T)=1,
$$

then:

$$
\sum_i\Delta PD_i=1.
$$

For constant exposure and unit discounting:

$$
\boxed{
CVA
=
LGD\,E_0.
}
$$

## Case F — netting

If:

$$
V_A=100,
\qquad
V_B=-90,
$$

then:

$$
V_{net}=10,
$$

and:

$$
\boxed{
E_{net}=10
}
$$

rather than:

$$
100^++(-90)^+=100.
$$

## Case G — perfect deterministic oracle

Choose deterministic $EE_i$, $DF_i$, $\Delta PD_i$, and $LGD_i$ and compare the production integrator exactly against:

$$
\sum_iLGD_iDF_iEE_i\Delta PD_i.
$$

The production result must match within a pre-declared numerical tolerance.

---

# 39. Scenario-weight sanity cases

For weights:

$$
\sum_sw_s=1,
$$

if every scenario has identical exposure:

$$
E_{s,i}=E_i^*
\quad\forall s,
$$

then:

$$
\widehat{EE}_i=E_i^*.
$$

This must hold independently of the number or ordering of scenarios.

Scenario permutation must not change the result provided weights are permuted consistently.

---

# 40. Time-grid contract

The exposure, discount, and credit calculations may use different native grids, but their mapping into the CVA integration grid must be explicit.

The integration grid:

$$
0=t_0<t_1<\cdots<t_N=T
$$

shall be strictly increasing and duplicate-free unless an explicit representation requires otherwise.

The implementation must define how it handles:

* valuation dates not present on the credit grid;
* credit dates not present on the valuation grid;
* discount dates not present on the exposure grid;
* interpolation between dates;
* extrapolation beyond available data;
* contractual discontinuities at integration dates.

No implicit date alignment or silent nearest-date selection is permitted where it changes the mathematical quantity.

Any interpolation scheme used for a production curve or exposure profile must be part of the model contract.

---

# 41. Maturity / horizon contract

$T$ means:

> the latest future date at which the selected netting set can have non-zero relevant closeout exposure under the implemented valuation model and contractual settlement rules.

It is not merely a scalar “portfolio maturity” field.

A netting set may contain trades with:

* different maturities;
* exercise dates;
* settlement dates;
* optionality;
* path-dependent termination;
* delivery periods.

The valuation layer owns the determination of the relevant future horizon.

The CVA engine consumes that horizon; it shall not independently invent a maturity rule from trade metadata alone.

---

# 42. Optionality and nonlinear valuation boundary

The CVA mathematics does not assume linear products.

For general derivatives:

$$
V_{s,i}^{clean}
=
V(X_{s,i},t_i)
$$

may involve:

* nonlinear payoff functions;
* options;
* exercise policies;
* path dependence;
* nested valuation;
* regression/proxy valuation;
* simulation-dependent exercise.

The CVA layer must consume future clean valuation through a stable interface and must not encode assumptions that future value is a linear function of spot or a P&L shock.

---

# 43. Bilateral/DVA boundary

The first implementation is unilateral:

$$
\boxed{
\text{counterparty default is modelled; our default is not.}
}
$$

Do not silently add DVA.

A later bilateral model requires explicit treatment of:

* counterparty default;
* own default;
* first-to-default ordering;
* bilateral closeout;
* own recovery;
* counterparty recovery;
* collateral return/closeout treatment.

The textbook decomposition:

$$
V_{bilateral}
\approx
V_{clean}-CVA+DVA
$$

may be useful under specified assumptions, but is not a universal mathematical identity for arbitrary bilateral closeout models.

Therefore DVA is a later model extension, not a simple boolean attached to an independent-CVA scalar.

---

# 44. XVA boundary

“XVA” shall mean a family of valuation adjustments rather than a universal additive identity.

Common labels include:

* CVA — counterparty credit;
* DVA — own credit;
* FVA — funding;
* MVA — initial-margin funding;
* KVA — capital cost.

The mathematical interactions among these adjustments depend on the valuation framework, funding policy, closeout, margin model, and dependencies.

Therefore the following shall **not** be treated as a universal contract:

$$
XVA
=
CVA+DVA+FVA+MVA+KVA+\cdots
$$

RiskQuant shall not create an additive XVA interface before unilateral CVA is independently correct.

---

# 45. Architectural mathematical seams

The preferred mathematical decomposition is:

## Seam A — future-state generation

$$
\boxed{
\text{market inputs}
\rightarrow
X_{s,i}
}
$$

This layer determines the future scenario state and its measure/weights.

## Seam B — clean future valuation

$$
\boxed{
(X_{s,i},t_i)
\rightarrow
V_{s,i}^{clean}
}
$$

## Seam C — contractual netting

$$
\boxed{
\{V_{j,s,i}^{clean}\}_{j\in\mathcal N}
\rightarrow
V_{\mathcal N,s,i}^{clean}
}
$$

## Seam D — closeout / collateral exposure

$$
\boxed{
(V_{\mathcal N,s,i}^{clean},M_{s,i},\text{agreement})
\rightarrow
U_{s,i}
\rightarrow
E_{s,i}=U_{s,i}^+
}
$$

## Seam E — exposure aggregation

$$
\boxed{
\{E_{s,i}\}_{s}
\rightarrow
EE_i
}
$$

## Seam F — credit curve

$$
\boxed{
\text{credit inputs}
\rightarrow
S_C,PD_C,\Delta PD_C,\lambda_C
}
$$

## Seam G — CVA integration

$$
\boxed{
(EE,DF,LGD,\Delta PD)
\rightarrow
CVA
}
$$

Each seam shall be independently testable against mathematical or golden-oracle expectations.

---

# 46. Kernel doctrine

Numerical kernels shall perform only declared mathematical transformations.

For the first-generation CVA integrator the kernel conceptually performs:

$$
CVA_i
=
LGD_iDF_iEE_i\Delta PD_i.
$$

It shall not perform:

* legal netting discovery;
* counterparty classification;
* collateral agreement lookup;
* credit-curve calibration;
* CDS convention selection;
* measure selection;
* numeraire selection;
* market-data retrieval;
* business-rule interpretation;
* object traversal;
* string-based dispatch;
* hidden interpolation policy;
* implicit sign-convention correction.

Those belong at validated semantic boundaries.

---

# 47. Fail-fast contract

Invalid mathematical inputs shall not be silently repaired.

Examples include:

* recovery outside $[0,1]$;
* LGD outside $[0,1]$;
* non-positive discount factors;
* survival outside $[0,1]$;
* increasing survival probabilities;
* negative default mass;
* inconsistent $PD$ and $S$;
* scenario weights negative or not summing to one;
* ambiguous scenario/date dimensions;
* mismatched grid semantics;
* duplicate dates where prohibited;
* future valuation data contaminated by CVA/DVA;
* physical-measure probability data supplied to a risk-neutral pricing contract without explicit transformation;
* inconsistent numeraire/currency;
* inconsistent day-count/time basis;
* NaN/None sentinel semantics entering kernels where finite numerical input is required.

The implementation shall fail at the semantic boundary rather than allow invalid data to enter numerical kernels.

---

# 48. No silent convention changes

The following are part of the mathematical contract and must be explicit:

* valuation perspective;
* pricing measure;
* valuation numeraire/currency;
* legal netting scope;
* collateral sign convention;
* closeout convention;
* recovery convention;
* probability representation;
* survival/default representation;
* hazard interpolation;
* exposure grid;
* credit grid;
* discount grid;
* time coordinate/day-count basis;
* date alignment/interpolation;
* endpoint/pre-discontinuity convention;
* default bucket convention;
* scenario weighting;
* scenario/time axis semantics.

A change in any one of these may change the mathematical quantity being priced.

---

# 49. No double-counting contract

Counterparty default shall enter the CVA calculation exactly once.

The canonical first-generation chain is:

$$
\boxed{
V^{clean}
\rightarrow
E
\rightarrow
EE
\rightarrow
\Delta PD
\rightarrow
CVA.
}
$$

The clean valuation contains no counterparty credit adjustment.

The exposure transformation contains no default probability.

The credit curve contains probability of default, not exposure.

The CVA integrator combines them once.

Therefore prohibited patterns include:

$$
V^{clean,CVA-adjusted}
\rightarrow
EE
\rightarrow
CVA
$$

and any equivalent double application of counterparty default loss.

At the general closeout level:

$$
\mathcal L_t
$$

already denotes the **loss amount**.

Therefore a stochastic-intensity representation shall contain:

$$
D_t\mathcal L_t\lambda_tG_tdt,
$$

not:

$$
D_t\mathcal L_tE_t\lambda_tG_tdt.
$$

---

# 50. Anti-patterns

## Anti-pattern 1 — scalar CVA shortcut

$$
CVA
=
LGD\times EPE\times PD
$$

with no explicit time profile.

## Anti-pattern 2 — today's exposure

$$
CVA
=
LGD\times V_0^+\times PD(T).
$$

This ignores the future exposure profile.

## Anti-pattern 3 — P&L substitution

Use historical VaR P&L scenarios as future CVA exposures without proving measure and future-valuation semantics.

## Anti-pattern 4 — trade-first positive part

$$
\sum_jV_j^+
$$

when the trades belong to a legally enforceable netting set.

## Anti-pattern 5 — counterparty equals netting set

Treating counterparty ID as sufficient legal netting scope.

## Anti-pattern 6 — CDS spread equals hazard

Treating spread as an intensity by identity.

## Anti-pattern 7 — arbitrary WWR multiplier

Applying a correlation factor to independent CVA without defining a joint exposure-credit model.

## Anti-pattern 8 — universal additive XVA

Treating CVA/DVA/FVA/MVA/KVA as independent scalars with guaranteed additivity.

## Anti-pattern 9 — hidden default bucket rule

Applying interval $\Delta PD$ to an exposure date without declaring the default-time approximation.

## Anti-pattern 10 — hidden stochastic discounting assumption

Using $DF\times EE$ when the model actually contains stochastic discount factors without proving the required factorisation.

## Anti-pattern 11 — credit-contaminated clean value

Supplying a valuation already adjusted for CVA or counterparty credit into the CVA exposure engine.

## Anti-pattern 12 — collateral as semantics-free subtraction

Subtracting a quantity called “collateral” without defining its sign, legal availability, currency, timing, or closeout meaning.

## Anti-pattern 13 — filtration that reveals default

Defining conditional survival by conditioning on an information set that already contains the realised default event.

## Anti-pattern 14 — unconditional EE used as universal WWR exposure

Treating:

$$
\mathbb E[E_t]
$$

as interchangeable with:

$$
\mathbb E[E_t\mid\tau_C>t]
$$

when exposure and default are dependent.

## Anti-pattern 15 — hazard representation without validity conditions

Treating every survival curve as if it necessarily possessed an ordinary intensity satisfying:

$$
dPD(t)=\lambda(t)S(t)\,dt.
$$

---

# 51. Validation hierarchy

Validation proceeds from mathematics outward.

## Level A — dimensions and scalar arithmetic

Verify units and the CVA integration kernel against hand calculations.

## Level B — credit curve invariants

Verify:

$$
S_0=1,
\quad
0\le S_i\le1,
\quad
\Delta PD_i\ge0.
$$

## Level C — exposure transformation

Verify positive-part and collateral semantics.

## Level D — netting

Verify aggregation before positive-part transformation.

## Level E — Monte Carlo expectation

Verify scenario weighting, measure semantics, permutation invariance, and convergence.

## Level F — CVA integration

Verify profile and aggregate against an independent golden oracle.

## Level G — temporal discretisation

Verify convergence as the integration grid is refined.

## Level H — discontinuity handling

Verify endpoint/pre-discontinuity conventions around contractual cashflows, exercise, settlement, and maturity boundaries.

## Level I — extensions

Only after Levels A–H are robust should the engine introduce stochastic discounting, complex collateral, WWR, bilateral default, or other XVA features.

---

# 52. Grid-refinement validation

Because the bucketed formula approximates a continuous integral, refinement should be tested.

For nested grids with maximum step size decreasing:

$$
\Delta t_{max}^{(1)}
>
\Delta t_{max}^{(2)}
>
\cdots,
$$

the resulting CVA values should exhibit convergence consistent with the chosen approximation and the underlying smoothness assumptions.

The test need not prescribe a universal convergence rate because that depends on the exposure and credit curves and the chosen numerical scheme.

It must, however, detect:

* gross non-convergence;
* discontinuity introduced by implementation errors;
* incorrect date alignment;
* inappropriate interpolation;
* inconsistent endpoint treatment.

---

# 53. Independent oracle requirement

The production implementation shall be validated against an independently constructed oracle wherever practical.

The oracle should not share the same implementation path, especially not:

* the same loop;
* the same interpolation helper;
* the same default-mass construction;
* the same scenario aggregation function.

For simple deterministic cases, the oracle may be a hand-derived scalar or independently evaluated expression.

The objective is to detect shared implementation errors, not merely reproduce them.

---

# 54. Reference example

Suppose:

| Date | Expected Exposure | Incremental PD | Discount Factor |
| ---- | ----------------: | -------------: | --------------: |
| 1Y   |       €10,000,000 |          0.50% |            0.97 |
| 2Y   |        €8,000,000 |          0.75% |            0.94 |
| 3Y   |        €6,000,000 |          1.00% |            0.91 |

and:

$$
R_C=40\%,
$$

so:

$$
LGD=60\%.
$$

Then:

$$
CVA_1
=
0.60
\times
0.97
\times
10{,}000{,}000
\times
0.005,
$$

$$
CVA_2
=
0.60
\times
0.94
\times
8{,}000{,}000
\times
0.0075,
$$

$$
CVA_3
=
0.60
\times
0.91
\times
6{,}000{,}000
\times
0.01.
$$

The total is:

$$
CVA=CVA_1+CVA_2+CVA_3.
$$

The numerical result shall be reproducible independently of the production implementation.

---

# 55. Controlled evolution path

The preferred development path is:

## Level 1 — Exposure-profile integration

Input:

$$
\{t_i,EE_i,DF_i,LGD_i,\Delta PD_i\}
$$

Calculate:

$$
CVA
=
\sum_iLGD_iDF_iEE_i\Delta PD_i.
$$

This isolates and proves the credit-weighted integration mathematics.

## Level 2 — Future-state simulation

Generate:

$$
X_{s,i}
$$

then value:

$$
V_{s,i}^{clean}.
$$

then exposure:

$$
E_{s,i}
=
(V_{s,i}^{clean})^+.
$$

## Level 3 — Contractual netting and simple collateral

Netted future value:

$$
V_{\mathcal N,s,i}^{clean}
$$

then:

$$
E_{s,i}
=
\left(
V_{\mathcal N,s,i}^{clean}
-
M_{s,i}
\right)^+.
$$

## Level 4 — Stochastic credit / WWR

Replace factorised default weighting with joint exposure-credit expectation:

$$
\mathbb E^{\mathbb Q^N}
\left[
\int
D_t
\mathcal L_t
\lambda_t
G_t
dt
\right].
$$

Under simple recovery:

$$
\mathcal L_t=LGD_tE_t.
$$

## Level 5 — Bilateral credit

Introduce own default, first-to-default effects, and explicit bilateral closeout.

## Level 6 — Broader XVA

Only then consider funding, initial-margin funding, capital, and coupled XVA effects.

Each level must preserve the semantics of the preceding levels and make any new assumptions explicit.

---

# 56. Code-archaeology contract

Before adding any new abstraction, inspect the existing RiskQuant codebase and determine whether the required mathematical primitive already exists.

At minimum, investigate the existing abstractions for:

1. future market-state generation;
2. future-dated valuation;
3. portfolio and trade valuation;
4. contractual/netting-set aggregation;
5. currency/numeraire conversion;
6. collateral representation;
7. discounting;
8. scenario weights;
9. survival/default curves;
10. hazard-rate representation;
11. exposure aggregation;
12. vectorised scenario × time storage;
13. deterministic integration;
14. independent golden-oracle infrastructure;
15. validation/invariant enforcement;
16. date/time-coordinate handling.

For every candidate abstraction, answer:

* What exact mathematical object does it represent?
* What are its units?
* What is its measure?
* What is its numeraire/currency?
* What is its sign convention?
* At what time does it live?
* Is it current-state, historical-scenario, or future-state information?
* Is it trade-level, netting-set-level, counterparty-level, or portfolio-level?
* What assumptions does it encode?
* What invariants does it guarantee?
* Does it represent clean value or credit-adjusted value?
* Can it satisfy the CVA contract without semantic distortion?

A type/name match is not evidence of mathematical compatibility.

---

# 57. Required archaeological challenge questions

Before proposing a CVA architecture, the reviewer shall be able to answer:

## Future valuation

Can the existing RiskQuant valuation infrastructure produce:

$$
V^{clean}(X_t,t)
$$

at arbitrary future dates under simulated future states?

If not, exactly which seam is missing?

## Probability measure

Where is $\mathbb Q^N$ established?

If it is not established, where can the engine safely establish or consume it?

## Numeraire

What is the valuation numeraire and currency?

Where is the conversion into that numeraire defined?

## Netting

Where is legal netting scope represented?

If it does not exist, what is the minimum abstraction needed without inventing a broader XVA framework?

## Collateral

Does an existing object represent contractual collateral semantics, or merely a number called “margin”?

## Credit

Does RiskQuant already represent survival/default probabilities or hazard curves?

What does the existing object actually mean?

## Time

Can the existing scenario engine represent:

$$
(s,t)
$$

rather than merely scenario snapshots around today?

What time coordinate and day-count basis are used?

## Kernel seam

Can the CVA integration be implemented as pure numerical arithmetic over validated arrays?

## Reuse

Can an existing abstraction be reused **without weakening or changing its mathematical semantics**?

If not, reuse is prohibited even if the code shape looks convenient.

---

# 58. First-generation mathematical API contract

The exact Python/API shape is an implementation choice, but semantically the first-generation computation must represent something equivalent to:

$$
\boxed{
(EE_i,DF_i,LGD_i,\Delta PD_i)_{i=1}^N
\rightarrow
(CVA_i)_{i=1}^N
\rightarrow
CVA.
}
$$

The exposure engine must separately represent:

$$
\boxed{
V_{s,i}^{clean}
\rightarrow
E_{s,i}
\rightarrow
EE_i.
}
$$

The credit layer must separately represent:

$$
\boxed{
\text{credit inputs}
\rightarrow
S_i,PD_i,\Delta PD_i,\lambda_i.
}
$$

This separation is a mathematical boundary, not merely an object-oriented preference.

---

# 59. Precision and numerical-policy requirements

The mathematical contract assumes sufficient numerical precision to preserve the sign, monotonicity, and mass properties of the inputs.

The implementation shall not use NaN, None, magic sentinel values, or silent clipping as substitutes for missing or invalid mathematical quantities inside numerical kernels.

Where clipping or projection is mathematically justified by an explicit model boundary, it must occur at that boundary and be visible in diagnostics.

Examples of prohibited silent repairs:

$$
S_i
\leftarrow
\min(1,\max(0,S_i))
$$

or:

$$
\Delta PD_i
\leftarrow
\max(0,\Delta PD_i)
$$

merely to make invalid input pass.

Such transformations change the supplied mathematical object.

Numerical tolerances may be used to assess floating-point representations of valid mathematical identities. Tolerance-based acceptance must not be confused with mathematical repair.

---

# 60. Required sign-convention statement

Every implementation and report must state the valuation perspective:

$$
V>0
\Rightarrow
\text{counterparty owes us}.
$$

For unilateral CVA:

$$
E=V^+
$$

in the first uncollateralised model.

Therefore a negative clean value cannot directly create unilateral CVA exposure at that instant.

The implementation shall not silently flip signs to match a reporting convention.

Any reporting transformation must occur after the mathematical calculation.

For collateralised extensions, the sign convention for $M_t$ must also be reported explicitly.

---

# 61. Required profile explainability

The result shall be explainable as:

$$
\boxed{
\text{future exposure profile}
\times
\text{default-time distribution}
\times
\text{LGD}
\times
\text{discounting}
}
$$

For the first-generation model, this is concretely:

$$
CVA_i
=
LGD_i
\times
DF_i
\times
EE_i
\times
\Delta PD_i.
$$

At minimum, RiskQuant must be able to answer:

* Why is expected exposure high at a given date?
* How much default probability lies in that interval?
* What discount factor was applied?
* What LGD was used?
* What is the bucket CVA contribution?
* How does the aggregate arise from the profile?
* What time and interpolation convention produced the profile?

A scalar-only CVA output is insufficient for validation and risk-management purposes.

---

# 62. Final canonical equations

## General unilateral CVA

$$
\boxed{
CVA_0
=
\mathbb E^{\mathbb Q^N}
\left[
D(0,\tau_C)
\mathcal L_{\tau_C}
\mathbf1_{\{\tau_C\le T\}}
\right].
}
$$

## General closeout decomposition

$$
\boxed{
U_t
=
\text{unsecured closeout amount},
\qquad
E_t=U_t^+.
}
$$

## Simple recovery/closeout specialisation

$$
\boxed{
\mathcal L_t
=
LGD_tE_t.
}
$$

## First-generation uncollateralised specialisation

$$
\boxed{
M_t=0,
\qquad
E_t=(V_t^{clean})^+.
}
$$

Therefore:

$$
\boxed{
\mathcal L_t
=
(1-R_C)
(V_t^{clean})^+.
}
$$

## Unconditional expected exposure

$$
\boxed{
EE(t)
=
\mathbb E^{\mathbb Q^N}[E_t].
}
$$

## Survival-conditioned expected exposure

$$
\boxed{
EE^{surv}(t)
=
\mathbb E^{\mathbb Q^N}[E_t\mid\tau_C>t].
}
$$

## First-generation independence consequence

$$
\boxed{
EE^{surv}(t)=EE(t).
}
$$

## First-generation deterministic/independent continuous-time CVA

$$
\boxed{
CVA_0
=
(1-R_C)
\int_0^T
DF(t)
EE(t)
\,dPD_C(t).
}
$$

## Hazard-rate representation

When the survival law admits the corresponding intensity representation:

$$
\boxed{
CVA_0
=
(1-R_C)
\int_0^T
DF(t)
EE(t)
\lambda_C(t)S_C(t)\,dt.
}
$$

## First-generation discrete endpoint approximation

$$
\boxed{
CVA_0^{(N)}
=
(1-R_C)
\sum_{i=1}^{N}
DF_i
EE_i
\left[
S_C(t_{i-1})-S_C(t_i)
\right].
}
$$

## First-generation Monte Carlo estimator

$$
\boxed{
\widehat{EE}_i
=
\sum_{s=1}^{S}
w_s
\max
(V_{s,i}^{clean},0),
\qquad
w_s\ge0,
\quad
\sum_sw_s=1.
}
$$

## WWR/intensity generalisation

$$
\boxed{
CVA_0
=
\mathbb E^{\mathbb Q^N}
\left[
\int_0^T
D_t
\mathcal L_t
\lambda_C(t)
G_t
\,dt
\right],
}
$$

with:

$$
\boxed{
G_t
=
\mathbb Q^N
\left(
\tau_C>t
\mid
\mathcal F_t^{\mathrm{pre-default}}
\right),
}
$$

where the conditioning filtration contains the market/credit state relevant to conditional survival but does not already reveal the realised default event.

Under simple recovery:

$$
\boxed{
\mathcal L_t
=
LGD_tE_t.
}
$$

Hence:

$$
\boxed{
CVA_0
=
\mathbb E^{\mathbb Q^N}
\left[
\int_0^T
D_t
LGD_t
E_t
\lambda_C(t)
G_t
\,dt
\right].
}
$$

---

# 63. Absolute implementation doctrine

The following statements are normative.

**The general CVA definition is expected discounted counterparty-default loss.**

**The first implementation is unilateral CVA, not general XVA.**

**Clean future valuation must contain no counterparty-credit adjustment.**

**Netting must occur before the positive-part exposure transformation.**

**Collateral semantics must be explicit; initial margin is not automatically subtractable.**

**Risk-neutral pricing measure and physical measure must never be silently interchanged.**

**The valuation measure is tied to an explicitly selected numeraire/currency.**

**The standard $DF\times EE\times\Delta PD$ formula is a factorised specialisation under explicit assumptions.**

**The bucketed sum is a numerical approximation to the continuous default integral.**

**A Monte Carlo sample mean is an estimator, not the mathematical definition of expected exposure.**

**Unconditional EE and survival-conditioned EE are distinct quantities; their equality is a first-generation independence consequence.**

**WWR is a joint exposure-credit model, not an arbitrary correlation correction.**

**Conditional survival must be conditioned on a pre-default information set, not on an information set that already reveals default.**

**CVA, DVA, FVA, MVA, and KVA are not assumed to be universally additive.**

**Do not double-count default or loss.**

**Do not infer semantics from names, shapes, or architectural convenience.**

**Kernels perform declared mathematics. Boundaries establish the assumptions and invariants.**

**Closeout assumptions must be explicit enough to determine the loss functional.**

**Default bucket timing, endpoint conventions, and discontinuity treatment are mathematical/numerical contracts.**

**Hazard-rate representations apply only where the survival law admits the corresponding intensity representation.**

**Time coordinates and day-count conventions are part of the mathematical contract.**

**Fail fast when the mathematical contract is violated.**

---

# 64. Completion criterion for the first CVA engine

The first RiskQuant CVA implementation is not complete merely because it produces a plausible scalar.

It is complete only when it can demonstrate all of the following:

1. The implementation can state exactly which mathematical quantity it calculates.
2. The first-generation assumptions are explicit and externally inspectable.
3. The chosen pricing measure and valuation numeraire are explicit.
4. Clean future valuation is distinct from historical P&L and from credit-adjusted valuation.
5. Legal netting scope is established before exposure truncation.
6. Exposure semantics and sign conventions are explicit.
7. The generation-one closeout/recovery mapping is explicit.
8. Credit curves are represented as mathematically defined survival/default/hazard quantities.
9. Default bucket timing and endpoint/discontinuity treatment are explicit.
10. Scenario weighting and scenario measure semantics are explicit.
11. The scalar CVA matches an independent oracle.
12. The bucket profile reconciles to the aggregate within declared numerical tolerance.
13. Mathematical invariants are enforced or tested.
14. Grid refinement can be demonstrated.
15. Monte Carlo convergence can be demonstrated where Monte Carlo is used.
16. The implementation can distinguish unconditional and survival-conditioned expected exposure.
17. The WWR extension has a mathematically correct pre-default conditional-survival seam.
18. No architectural abstraction is introduced solely because the term “XVA” appears; every abstraction must have a defensible mathematical owner.
19. The implementation can explain which assumptions would need to change for collateral, WWR, bilateral default, or broader XVA.
20. No default-loss quantity is counted twice through overlapping definitions of exposure and loss.

Only after these conditions are satisfied should RiskQuant move beyond unilateral first-generation CVA.
