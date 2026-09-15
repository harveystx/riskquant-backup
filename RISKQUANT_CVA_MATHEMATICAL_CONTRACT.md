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
8. Do not hide measure, numeraire, closeout, recovery, collateral, sign, time-bucket, event-order, or interpolation conventions inside numerical kernels.
9. Do not double-count counterparty credit in future clean valuation and again in CVA.
10. Prefer a small mathematically exact implementation over a broad abstraction whose semantics are ambiguous.
11. Boundaries establish and validate semantics; kernels perform declared mathematics.
12. Invalid mathematical inputs fail fast rather than being silently repaired into plausible-looking numbers.
13. Distinguish unconditional exposure expectations from survival-conditioned exposure expectations; equality between them is an explicit consequence of the first-generation independence assumption.
14. Distinguish the pre-default reference filtration used for conditional survival from a filtration that already reveals the default event.
15. Treat default-time discretisation and event ordering as part of the numerical/model contract.
16. Distinguish **numeraire**, **valuation currency**, **discount factor**, and **day-count/time coordinate**. They are related concepts but are not interchangeable.
17. Do not allow upstream curve interpolation, date alignment, or currency conversion to become hidden behaviour of a numerical CVA kernel.
18. Do not present a weighted Monte Carlo average as a valid pricing estimator unless the scenario law and weighting scheme jointly define the target pricing expectation.
19. Do not silently extrapolate beyond the domain of required market or credit inputs.
20. Preserve the mathematical distinction between an actual default loss, a signed closeout amount, and positive unsecured exposure.

---

# 1. Mathematical north star

The most general unilateral CVA object required by this contract is the expected discounted loss caused by counterparty default:

$$
\boxed{
CVA_0
=
\mathbb E^{\mathbb Q^N}
\left[
D^N(0,\tau_C)\,
\mathcal L_{\tau_C}\,
\mathbf 1_{\{\tau_C\le T\}}
\right]
}
$$

where:

* $\mathbb Q^N$ is the pricing measure associated with the explicitly selected numeraire process $N_t$;
* $\tau_C$ is counterparty default time;
* $D^N(0,t)$ is the corresponding discount-factor representation;
* $\mathcal L_t$ is the **actual counterparty-default loss amount on the netting set at time $t$ under the selected closeout model**;
* $T$ is the latest future date at which relevant closeout exposure may be non-zero under the selected contractual and valuation model.

This is the **general economic definition**.

Everything else in this document is either:

* a definition of one component;
* a modelling specialisation;
* a numerical discretisation;
* a Monte Carlo/statistical estimator;
* or a controlled future extension.

The general definition is a **pricing quantity**, not a generic risk statistic. It is the time-0 value of a contingent default-loss cashflow under a pricing measure. It is therefore analogous to the valuation of any other contingent claim, although the claim itself depends on the counterparty default event and the closeout model.

---

# 2. Numeraire and pricing-measure contract

A numeraire is a strictly positive value process:

$$
\boxed{
N_t>0.
}
$$

It is **not** a currency label.

The pricing measure associated with $N_t$ is denoted:

$$
\boxed{
\mathbb Q^N.
}
$$

For a payoff/value process represented in the same numeraire framework, the corresponding stochastic discount-factor representation is:

$$
\boxed{
D^N(0,t)
=
\frac{N_0}{N_t}.
}
$$

The particular construction of $N_t$ is a model choice and must be externally inspectable.

Examples may include:

* a money-market account;
* an OIS-related account;
* a collateral/discounting account under a specified collateral remuneration convention;
* another explicitly defined strictly positive tradable numeraire.

RiskQuant shall never equate:

$$
\text{currency}
\equiv
\text{numeraire}.
$$

The valuation currency is a unit in which the value is expressed.

The numeraire is a stochastic value process used to define the pricing measure and discounting representation.

For first-generation CVA, the implementation shall select one explicit valuation currency and one explicit deterministic discounting convention. The general mathematical contract nevertheless retains the distinction between currency and numeraire so that later extensions do not require redefining the foundations.

---

# 3. First-generation production scope

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

6. The valuation currency is explicitly fixed.
7. The valuation numeraire is explicitly fixed.
8. Discounting in that valuation setup is deterministic:

$$
DF(t)>0.
$$

9. The counterparty default process is independent of the future market/exposure process in the precise sense specified in Section 10.
10. The first exposure model is **uncollateralised clean-MTM closeout with instantaneous closeout and no margin-period dynamics**.
11. Future clean valuation is supplied by a future-state valuation/simulation layer.
12. Credit is supplied by a risk-neutral survival/default curve or equivalent deterministic hazard representation.
13. Default is integrated on an explicit time grid using the declared bucket and event-order convention.
14. The internal time coordinate used by the CVA numerical layer is explicitly defined.
15. Required credit and discount curves cover the entire integration horizon. Silent extrapolation is prohibited.

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
\mathbb E^{\mathbb Q^N}
\left[
(V_t^{\mathrm{clean}})^+
\right].
$$

This is a **specialisation** of the general economic definition.

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
\right].
}
$$

This is a **time-discretised approximation** to the continuous-time quantity.

---

# 4. Mathematical status hierarchy

The following table is normative.

| Object                                                                              | Mathematical status                                              |
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| $\mathbb E^{\mathbb Q^N}[D^N(0,\tau_C)\mathcal L_{\tau_C}\mathbf1_{{\tau_C\le T}}]$ | **General economic definition**                                  |
| $N_t$                                                                               | **Numeraire definition**                                         |
| $D^N(0,t)=N_0/N_t$                                                                  | **Discount-factor representation associated with the numeraire** |
| $U_t$                                                                               | **Signed unsecured closeout amount**                             |
| $E_t=U_t^+$                                                                         | **Positive unsecured exposure specialisation**                   |
| $\mathcal L_t=LGD_tE_t$                                                             | **Recovery/closeout specialisation**                             |
| $EE(t)=\mathbb E^{Q^N}[E_t]$                                                        | **Unconditional expected exposure**                              |
| $EE^{surv}(t)=\mathbb E^{Q^N}[E_t\mid\tau_C>t]$                                     | **Survival-conditioned expected exposure**                       |
| $EE^{surv}(t)=EE(t)$                                                                | **First-generation independence consequence**                    |
| $CVA=\int DF,EE,dPD$                                                                | **Deterministic/independent factorisation**                      |
| $\sum_iDF_iEE_i\Delta PD_i$                                                         | **Time-discretised numerical approximation**                     |
| $\widehat{EE}*i=\sum_sw_sE*{s,i}$                                                   | **Monte Carlo estimator**                                        |
| $EE\times PD\times LGD$                                                             | **Not a general identity**                                       |

No implementation may conflate these levels without an explicit derivation.

---

# 5. Economic interpretation and sign

CVA is a **valuation adjustment caused by counterparty default risk**.

It is not inherently:

* VaR;
* Expected Shortfall;
* PFE;
* a stress loss;
* historical P&L;
* today's positive MTM multiplied by a scalar PD;
* or a generic $EE\times PD\times LGD$ product.

The valuation perspective is:

$$
\boxed{
V>0
\Rightarrow
\text{counterparty owes us value}.
}
$$

Under unilateral CVA:

$$
E_t=V_t^+
$$

in the first uncollateralised specialisation.

Therefore:

$$
CVA_0\ge0,
$$

and:

$$
\boxed{
V_0^{CVA}
=
V_0^{clean}
-
CVA_0.
}
$$

Hence:

$$
V_0^{CVA}\le V_0^{clean}.
$$

### Sign sanity

If:

$$
V^{clean}=+100,
$$

then:

$$
E=100
$$

and a positive CVA contribution is possible.

If:

$$
V^{clean}=-100,
$$

then:

$$
E=0
$$

and that state contributes no unilateral counterparty CVA.

The implementation shall never silently reverse valuation perspective to match a downstream reporting sign convention.

---

# 6. Measure contract

CVA is priced under the selected pricing measure:

$$
\boxed{
\mathbb Q^N.
}
$$

The measure attached to every probabilistic input used by the pricing calculation must be externally inspectable, including:

* future market-state scenarios;
* scenario weights;
* expected exposure;
* survival probabilities;
* default probabilities;
* hazard rates;
* stochastic discounting inputs where applicable.

The following substitutions are prohibited unless an explicit measure transformation is supplied:

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

A valid sum of weights is insufficient to establish the correct measure:

$$
\sum_sw_s=1
\]

does not imply that the represented probability law is $\mathbb Q^N$.

---

# 7. Clean future valuation contract

For scenario $s$ and future date $t_i$:

$$

\boxed{
V_{s,i}^{\mathrm{clean}}
========================

V^{\mathrm{clean}}(X_{s,i},t_i).
}

$$

where $X_{s,i}$ is the future market-factor state.

The clean valuation shall be from our perspective:

$$

V>0
\Rightarrow
\text{counterparty owes us},

$$
$$

V<0
\Rightarrow
\text{we owe the counterparty}.

$$

## Clean-value purity invariant

Future clean valuation shall contain **no counterparty credit adjustment**.

It shall not silently include:

- CVA;
- DVA;
- FVA;
- counterparty-credit spread adjustments;
- recovery/default-loss effects;
- collateral-credit adjustments.

Therefore:

$$

\boxed{
\text{clean valuation contains no counterparty credit adjustment}.
}

$$

The valuation layer is responsible for product/market valuation only.

---

# 8. Future valuation is not historical P&L

Historical market-risk machinery commonly computes:

$$

\text{historical scenario}
\rightarrow
\Delta V
\quad\text{or}\quad
P&L.

$$

CVA requires:

$$

\boxed{
(X_{s,i},t_i)
\rightarrow
V_{s,i}^{clean}.
}

$$

These are different mathematical objects.

Therefore:

$$

\boxed{
\text{historical P&L scenario}
\not\equiv
\text{future risk-neutral exposure scenario}.
}

$$

Existing RiskQuant scenario infrastructure may be reused only if its semantics prove compatibility with:

1. future time evolution;
2. the required pricing measure;
3. future clean valuation rather than historical P&L;
4. the exposure time grid;
5. scenario weighting.

Structural resemblance is insufficient.

---

# 9. Time, currency, and dimensional contract

All values participating in a netting calculation must be represented in the same explicitly selected valuation/closeout currency after applying the contractual/modelled FX conversion convention.

The distinction is:

$$

\boxed{
\text{currency}
\neq
\text{numeraire}
\neq
\text{discount factor}.
}

$$

The generation-one implementation uses one explicit valuation currency/numeraire setup.

Dimensional invariants are:

$$

# [V]

# [E]

# [\mathcal L]

# [CVA]

\text{currency},

$$
$$

# [DF]

# [PD]

# [LGD]

1,

$$

and:

$$

# [\lambda]

1/\text{time}.

$$

Therefore:

$$

# [DF\times EE\times\Delta PD]

\text{currency},

$$

and:

$$

# [DF\times E\times\lambda\times dt]

\text{currency}.

$$

A dimensionally inconsistent implementation is invalid regardless of numerical plausibility.

---

# 10. Formal first-generation independence contract

Let:

$$

\mathcal F_t^M

$$

denote the **full market/risk-factor filtration relevant to future exposure**, including all historical and current market information required to determine the exposure process.

The first-generation model requires that counterparty default have deterministic conditional survival with respect to this filtration:

$$

\boxed{
\mathbb Q^N
\left(
\tau_C>t
\mid
\mathcal F_t^M
\right)
=======

S_C(t)
}

$$

for all deterministic $t\in[0,T]$.

In addition:

- $E_t$ is $\mathcal F_t^M$-measurable;
- the default process admits the deterministic survival/default law supplied to the credit layer;
- the exposure process is evaluated on the pre-default market process;
- the default compensator contains no additional dependence on market information.

Under these assumptions:

$$

\boxed{
\mathbb E^{\mathbb Q^N}
[
E_t\mid\tau_C>t
]
=

\mathbb E^{\mathbb Q^N}[E_t].
}

$$

Thus:

$$

\boxed{
EE^{surv}(t)=EE(t).
}

$$

This equality is a **consequence of the independence model**, not a definition.

The one-line conditional-survival condition shall not be interpreted as permission to ignore the information set. If $\mathcal F_t^M$ omits relevant historical market information through which default depends on exposure, the stated condition may appear numerically satisfied while the required independence is false.

---

# 11. Default-time versus deterministic-time exposure

The economic definition contains:

$$

E_{\tau_C},

$$

not merely:

$$

E_t.

$$

The generation-one factorisation requires the exposure process to be appropriately adapted/measurable to the market filtration and the default mechanism to be independent of that market process in the compensator sense required to replace default-time exposure by the deterministic-time exposure profile weighted by the default measure.

For a continuous default law, the conceptual factorisation is:

$$

\mathbb E^{Q^N}
[
D_t\mathcal L_t,dPD_C(t)
]
=

DF(t),
LGD_t,
EE(t),
dPD_C(t)

$$

under the first-generation deterministic assumptions.

The contract shall **not** claim that:

$$

# \mathbb E[E_{\tau_C}\mid\tau_C\in dt]

EE(t)

$$

follows solely from:

$$

Q(\tau_C>t\mid\mathcal F_t^M)=S_C(t).

$$

The additional exposure-adaptedness and default-compensator assumptions in this section are part of the factorisation contract.

---

# 12. General default-loss contract

The general default loss is:

$$

\boxed{
\mathcal L_t
============

\text{counterparty-default loss amount on the netting set at }t.
}

$$

The general CVA definition remains:

$$

\boxed{
CVA_0
=====

\mathbb E^{Q^N}
\left[
D^N(0,\tau_C)
\mathcal L_{\tau_C}
\mathbf1_{{\tau_C\le T}}
\right].
}

$$

This deliberately separates the economic loss from any particular recovery or closeout representation.

---

# 13. Closeout and unsecured exposure contract

Define:

$$

\boxed{
U_t
===

\text{signed unsecured closeout amount at }t.
}

$$

Positive unsecured exposure is:

$$

\boxed{
E_t=U_t^+.
}

$$

Under the first-generation simple closeout specialisation:

$$

\boxed{
U_t
===

V_t^{clean}-M_t.
}

$$

Therefore:

$$

\boxed{
E_t
===

(V_t^{clean}-M_t)^+.
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
E_t=(V_t^{clean})^+.
}

$$

The positive-part operation occurs **after netting, currency conversion, and closeout/collateral adjustments**.

---

# 14. First-generation closeout specialisation

The first-generation engine uses an explicit **clean-MTM replacement-value closeout specialisation** with:

1. clean future value as the reference closeout value;
2. immediate closeout at the default-time approximation;
3. unsecured amount equal to clean netting-set value less the signed collateral balance;
4. counterparty loss only on positive unsecured exposure;
5. deterministic recovery applied to that positive unsecured exposure;
6. no margin-period-of-risk dynamics;
7. no liquidation-cost term;
8. no additional stochastic closeout adjustment.

Thus:

$$

\boxed{
\mathcal L_t
============

LGD_tE_t.
}

$$

In the first uncollateralised model:

$$

\boxed{
\mathcal L_t
============

(1-R_C)(V_t^{clean})^+.
}

$$

This is a **model specialisation**, not a universal closeout theorem.

A future closeout model may instead define:

$$

# \mathcal L_t

\Phi(
V_t^{clean},
M_t,
\text{agreement},
\text{default state},
\text{settlement rules},
\ldots
),

$$

without changing the general CVA definition.

---

# 15. Collateral sign and collateral semantics

The canonical signed collateral convention is:

$$

\boxed{
M_t>0
\iff
\text{collateral held by us}.
}

$$

and:

$$

\boxed{
M_t<0
\iff
\text{collateral posted by us}.
}

$$

Under this convention:

$$

U_t=V_t^{clean}-M_t.

$$

Therefore:

- receiving €80 collateral gives $M_t=+80$ and reduces unsecured exposure;
- posting €80 collateral gives $M_t=-80$ and increases unsecured exposure.

Numerical sanity:

$$

V=100,\quad M=80
\Rightarrow
E=20,

$$

while:

$$

V=100,\quad M=-80
\Rightarrow
E=180.

$$

A sign reversal that leaves CVA unchanged is a semantic defect.

Collateral must be represented in the same valuation currency/numeraire used for closeout after applying the contractual FX convention.

Collateral is not merely a scalar subtraction. The collateral state must reflect the applicable agreement and timing rules.

---

# 16. Initial margin, variation margin, and independent amount

The contract distinguishes at least conceptually:

$$

M_t^{VM},
\qquad
M_t^{IM},
\qquad
M_t^{IA}.

$$

The first generation implements none of them:

$$

M_t=0.

$$

Variation margin, initial margin, and independent amount shall not be collapsed into one semantics-free `margin` quantity.

Initial margin is not automatically subtractable merely because it is economically collateral.

Before IM can enter a closeout exposure formula, the model must determine:

- whether it is legally available against the default loss;
- whether it is segregated;
- who holds it;
- whether it is returned on default;
- whether it is subject to recovery;
- whether it is available for set-off;
- the relevant closeout convention.

The same principle applies to independent amount.

---

# 17. Recovery and LGD contract

General recovery may be stochastic:

$$

R_t\in[0,1],

$$

with:

$$

LGD_t=1-R_t.

$$

Under the recovery specialisation:

$$

\boxed{
\mathcal L_t=LGD_tE_t.
}

$$

Under generation one:

$$

R_t=R_C,

$$
$$

LGD=1-R_C.

$$

Therefore:

$$

\boxed{
\mathcal L_t
============

(1-R_C)E_t.
}

$$

The generation-one model does **not** apply recovery to an arbitrary signed closeout amount before the positive-part operator.

A model in which recovery is instead applied to some other closeout quantity is a different recovery/closeout model and must be represented explicitly.

Recovery must remain distinct from default probability.

---

# 18. Netting-set contract

CVA exposure is measured at the legally enforceable netting-set level.

The mathematical dependency is:

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
V_{\mathcal N,s,i}^{clean}
==========================

\sum_{j\in\mathcal N}
V_{j,s,i}^{clean,(N)}.
}

$$

Only after netting does the exposure transformation occur:

$$

# E_{\mathcal N,s,i}

\left(
V_{\mathcal N,s,i}^{clean}
--------------------------

M_{\mathcal N,s,i}
\right)^+.

$$

In general:

$$

\boxed{
\left(\sum_jV_j\right)^+
\ne
\sum_jV_j^+.
}

$$

Example:

$$

V_A=100,\qquad V_B=-90.

$$

Correct:

$$

V_{net}=10,
\qquad
E_{net}=10.

$$

Incorrect trade-level positive-part aggregation:

$$

100^++(-90)^+=100.

$$

If:

$$

V_A=100,\qquad V_B=-100,

$$

then:

$$

V_{net}=0,
\qquad
E_{net}=0.

$$

Counterparty identity alone shall never define $\mathcal N$.

The legal layer must establish enforceable trade membership and any contractual restrictions on netting.

---

# 19. Multi-currency netting and closeout

For first-generation CVA, all values entering the mathematical netting operation shall be converted into one explicitly selected valuation/closeout currency.

The conversion must follow the selected contractual/model convention:

$$

# V_{j,t}^{(N)}

FX_{j\rightarrow N}(t),
V_{j,t}^{(j)}

$$

or the equivalent valuation-layer transformation appropriate to the chosen currency setup.

The canonical ordering is:

$$

\boxed{
\text{trade valuation}
\rightarrow
\text{contractual FX conversion}
\rightarrow
\text{legal netting}
\rightarrow
\text{collateral/closeout}.
}

$$

A collateral amount denominated in another currency must likewise be converted under the applicable collateral/closeout FX convention before entering:

$$

U_t=V_t-M_t.

$$

The first-generation implementation shall not attempt to infer multi-currency legal closeout from a generic reporting FX conversion.

---

# 20. Expected exposure contract

Define unconditional expected exposure:

$$

\boxed{
EE(t)
=====

\mathbb E^{Q^N}[E_t].
}

$$

Define survival-conditioned expected exposure:

$$

\boxed{
EE^{surv}(t)
============

\mathbb E^{Q^N}
[
E_t\mid\tau_C>t
].
}

$$

These are distinct quantities.

Under generation-one independence:

$$

\boxed{
EE^{surv}(t)=EE(t).
}

$$

Under WWR this equality generally fails.

The sample estimator is separate from the mathematical expectation:

$$

\boxed{
\widehat{EE}_i
==============

\sum_{s=1}^S
w_sE_{s,i}.
}

$$

---

# 21. Expected exposure versus PFE

Expected exposure:

$$

# EE(t)

\mathbb E[V_t^+].

$$

PFE at confidence level $q$:

$$

# PFE_q(t)

Q_q(V_t^+).

$$

These are different mathematical objects.

CVA uses the expectation of the relevant positive unsecured closeout exposure, not a future tail quantile.

PFE shall not replace EE unless a separate approximation is explicitly defined and independently validated.

Internal RiskQuant terminology should prefer:

- `expected_exposure`;
- `expected_exposure_profile`;
- `survival_conditioned_expected_exposure`.

---

# 22. Default-time law and survival probabilities

Define:

$$

\boxed{
S_C(t)
======

Q^N(\tau_C>t).
}

$$

Then cumulative default probability:

$$

\boxed{
PD_C(t)
=======

1-S_C(t).
}

$$

For bucket $(t_{i-1},t_i]$:

$$

\boxed{
\Delta PD_i
===========

# PD_C(t_i)-PD_C(t_{i-1})

S_C(t_{i-1})-S_C(t_i).
}

$$

The bucket includes defaults satisfying:

$$

t_{i-1}<\tau_C\le t_i.

$$

Therefore a default exactly at $t_i$ belongs to bucket $i$.

---

# 23. Default-event ordering and endpoint convention

The first-generation time grid is:

$$

0=t_0<t_1<\cdots<t_N=T.

$$

The bucket:

$$

(t_{i-1},t_i]

$$

is represented numerically by its **right endpoint** $t_i$.

This is an approximation:

> Defaults in $(t_{i-1},t_i]$ are represented by the exposure and discount factor at the right-endpoint closeout state associated with $t_i$.

For generation one, the canonical event-order rule is:

$$

\boxed{
\text{default closeout at }t_i
\text{ uses the pre-contractual-event clean state }V(t_i^-)
}

$$

when a scheduled cashflow, exercise, settlement, fixing, maturity, or other valuation discontinuity occurs at $t_i$.

Thus:

$$

EE_i

$$

must correspond to the declared pre-event exposure state.

The contract does not permit silent selection between:

$$

V(t_i^-)
\quad\text{and}\quad
V(t_i^+).

$$

The valuation layer must expose the state appropriate to this event-order convention.

A different closeout event ordering is a different model and must be explicitly declared.

---

# 24. Survival/default curve invariants

The first-generation credit representation requires:

$$

S_C(0)=1,

$$
$$

0\le S_C(t)\le1,

$$

and:

$$

S_C(t_i)\le S_C(t_{i-1}).

$$

Therefore:

$$

\boxed{
\Delta PD_i\ge0.
}

$$

Total default mass through $T$ is:

$$

\boxed{
\sum_{i=1}^{N}\Delta PD_i
=========================

1-S_C(T).
}

$$

This equality is algebraic given the bucket definition, subject to finite-precision tolerance.

A survival curve that violates these conditions is invalid.

No silent clipping or projection is permitted.

---

# 25. Credit-curve domain, interpolation, and alignment

The credit layer owns:

- credit-curve interpolation;
- conversion from native market-curve conventions to the internal time coordinate;
- evaluation of survival/default quantities at requested integration dates;
- any internally required integration of hazard rates.

The CVA numerical integrator does **not** perform credit interpolation.

The required interface is conceptually:

$$

\boxed{
\text{credit model}
\rightarrow
S_C(t_i),,
PD_C(t_i),,
\Delta PD_i,,
\lambda_C(t_i)
}

$$

on the declared CVA integration grid where those quantities are valid.

If a requested date lies outside the supported credit-curve domain:

$$

\boxed{
\text{FAIL FAST}.
}

$$

Silent extrapolation is prohibited.

Interpolation choice is mathematically material.

For example, if:

$$

S(1)=0.98,\qquad S(2)=0.90,

$$

then linear survival gives:

$$

S(1.5)=0.94,

$$

while log-linear survival gives:

$$

S(1.5)=\sqrt{0.98\cdot0.90}.

$$

These produce different default masses and therefore potentially different CVA values.

The interpolation convention must therefore be owned by the credit model and externally inspectable.

---

# 26. Hazard-rate contract

For an absolutely continuous deterministic survival law with positive survival:

$$

S_C(t)>0,

$$

and sufficient regularity:

$$

# S_C(t)

\exp
\left(
-\int_0^t
\lambda_C(u),du
\right),

$$

where:

$$

\boxed{
\lambda_C(t)
============

-\frac{d}{dt}\log S_C(t).
}

$$

The corresponding default-density relationship is:

$$

\boxed{
\frac{dPD_C(t)}{dt}
===================

\lambda_C(t)S_C(t).
}

$$

The relevant conditions are that the survival law be non-increasing, absolutely continuous on the relevant interval, strictly positive there, and admit the corresponding non-negative integrable intensity.

The following are distinct:

$$

S_C(t),
\qquad
PD_C(t),
\qquad
\Delta PD_i,
\qquad
\lambda_C(t).

$$

They are related quantities, not aliases.

For a valid deterministic intensity model:

$$

\boxed{
\lambda_C(t)\ge0.
}

$$

A negative hazard in the generation-one intensity representation is invalid and shall fail fast.

If:

$$

S_C(t)=0,

$$

then:

$$

-\log S_C(t)

$$

and therefore:

$$

# \lambda_C(t)

-\frac{d}{dt}\log S_C(t)

$$

are not defined in the ordinary finite sense.

The survival/default-mass representation remains valid.

---

# 27. Default atoms and general default measure

The survival/default-mass formulation is more general than an ordinary hazard integral.

If the default law contains an atom:

$$

# p_k

Q^N(\tau_C=t_k)>0,

$$

then the correct CVA contribution includes the atom explicitly.

A general default-measure representation is:

$$

\boxed{
CVA_0
=====

\int_{(0,T]}
D^N(0,t),
\mathbb E^{Q^N}
[
\mathcal L_t
\mid
\tau_C=t
]
,dPD_C(t).
}

$$

Under first-generation independence, deterministic LGD, and deterministic discounting, this reduces appropriately to an exposure expectation integrated against the default probability measure.

If the default law has both continuous and atomic components:

$$

# dPD_C

dPD_C^{cont}
+
\sum_kp_k\delta_{t_k},

$$

then:

$$

\boxed{
CVA
===

\int
DF(t),
LGD(t),
EE(t),
dPD_C^{cont}(t)
+
\sum_k
DF(t_k),
LGD(t_k),
EE_{t_k}^{atom},
p_k.
}

$$

The atom must use the explicitly selected event-order exposure at $t_k$.

An ordinary $\lambda(t)S(t)dt$ representation alone is insufficient when atomic default mass exists.

---

# 28. Credit calibration contract

The conceptual market-data chain is:

$$

\boxed{
\text{CDS market data}
\rightarrow
\text{credit calibration}
\rightarrow
S_C(t),PD_C(t),\Delta PD_i,\lambda_C(t)
\rightarrow
CVA.
}

$$

A CDS spread is **not** a hazard rate by identity.

The approximation:

$$

s\approx LGD\times\lambda

$$

may be useful as intuition but is not the calibration equation.

Actual calibration can depend on:

- recovery;
- discounting;
- premium/accrual conventions;
- default settlement conventions;
- calibration instruments;
- curve parametrisation;
- interpolation;
- numerical integration.

The recovery assumption used in CDS calibration and the recovery assumption used for CVA are conceptually distinct.

Therefore:

$$

R^{CDS}
\not\equiv
R^{CVA}

$$

unless the model explicitly imposes that relationship.

The first RiskQuant CVA implementation may consume an already calibrated survival/default curve rather than implementing a CDS bootstrap.

This is preferred over embedding an unvalidated calibration stack merely to complete the feature.

---

# 29. Discounting contract

For generation one:

$$

\boxed{
DF_i=DF(0,t_i)>0.
}

$$

$DF(t)$ shall mean:

> the deterministic pricing discount factor associated with the selected valuation currency, numeraire, and discounting/collateral convention.

The term "risk-free discount factor" shall not be used as an undefined synonym.

The discount layer owns:

- market discount-curve conventions;
- interpolation;
- conversion to the CVA integration dates;
- any native day-count/calendar handling.

If a required date is outside the supported discount-curve domain:

$$

\boxed{
\text{FAIL FAST}.
}

$$

No silent extrapolation.

If rates are stochastic, the general object is joint:

$$

\boxed{
CVA
===

\mathbb E^{Q^N}
\left[
\int
D_t
\mathcal L_t
,dPD_C(t)
\right]

$$

or, under an appropriate intensity construction:

$$

\boxed{
CVA
===

\mathbb E^{Q^N}
\left[
\int_0^T
D_t
\mathcal L_t
\lambda_t
G_t,dt
\right].
}

$$

In general:

$$

\mathbb E[D_tE_t]
\ne
\mathbb E[D_t]\mathbb E[E_t]

$$

because:

$$

# \mathbb E[D_tE_t]

\mathbb E[D_t]\mathbb E[E_t]
+
\operatorname{Cov}(D_t,E_t).

$$

The first-generation deterministic factorisation must not be embedded into a general stochastic-discounting abstraction.

---

# 30. Internal time-coordinate contract

RiskQuant shall use an explicit internal year-fraction coordinate:

$$

\boxed{
t
=

\operatorname{YearFraction}(t_0,\text{date})
}

$$

with the specific internal convention declared by the date/time subsystem.

For the first-generation contract, this shall be one canonical internal basis, for example ACT/365F if that is the repository's canonical convention.

External curves may use different market conventions.

The curve layer is responsible for converting native conventions into the internal coordinate before numerical integration.

The CVA kernel shall never infer day-count conventions from raw dates.

A rate expressed against one time basis shall not be multiplied by a time increment from another basis without an explicit conversion.

---

# 31. Grid ownership and alignment

The CVA engine has one explicit **integration grid**:

$$

0=t_0<t_1<\cdots<t_N=T.

$$

The integration grid must be:

- strictly increasing;
- duplicate-free unless a future model explicitly represents event multiplicity;
- within the domain of all required input curves;
- explicitly associated with the chosen time coordinate.

Upstream seams own their native-grid transformations.

Therefore:

$$

\boxed{
\text{exposure layer}
\rightarrow
EE(t_i),
}

$$
$$

\boxed{
\text{discount layer}
\rightarrow
DF(t_i),
}

$$
$$

\boxed{
\text{credit layer}
\rightarrow
S_C(t_i),\Delta PD_i.
}

$$

The CVA integration kernel receives aligned arrays.

It shall not:

- perform date matching;
- perform nearest-date selection;
- interpolate curves;
- extrapolate;
- infer event order.

---

# 32. Horizon contract

$T$ is:

> the latest future date at which the selected netting set can have non-zero relevant closeout exposure under the implemented valuation model and contractual settlement rules.

It is not merely a scalar "portfolio maturity."

The valuation layer owns the determination of the economically relevant horizon.

Examples requiring valuation awareness include:

- different trade maturities;
- exercise dates;
- settlement dates;
- path-dependent termination;
- delivery periods;
- cashflows after exercise;
- settlement lags.

The CVA engine consumes the validated horizon.

It shall not invent a maturity rule from trade metadata alone.

---

# 33. Optionality and nonlinear valuation boundary

The CVA mathematics does not assume linear products.

For general derivatives:

$$

# V_{s,i}^{clean}

V(X_{s,i},t_i)

$$

may involve:

- nonlinear payoff functions;
- options;
- exercise policies;
- path dependence;
- nested valuation;
- regression/proxy valuation;
- simulation-dependent exercise.

The CVA layer must consume future clean valuation through a stable interface.

It shall not encode assumptions that future value is a linear function of spot or a historical P&L shock.

---

# 34. Monte Carlo contract

For scenario $s$ and time $t_i$:

$$

# V_{s,i}^{clean}

V^{clean}(X_{s,i},t_i).

$$

First-generation exposure is:

$$

\boxed{
E_{s,i}
=======

\max(V_{s,i}^{clean},0)
}

$$

after the scenario-level netting and collateral transformation required for the selected netting set.

The semantic shape shall be explicit:

$$

\boxed{
E\in\mathbb R^{S\times N}
}

$$

where:

- axis 0 = scenario;
- axis 1 = time.

A transpose shall never silently change the mathematical meaning of the array.

---

# 35. Monte Carlo measure and weighting

The scenario generator must state:

1. the probability measure under which scenarios are generated;
2. the valuation numeraire associated with that measure;
3. the scenario sampling law;
4. the interpretation of any scenario weights.

For ordinary IID Monte Carlo under the pricing measure:

$$

X_s\sim Q^N,

$$

and:

$$

w_s=\frac1S.

$$

Then:

$$

# \widehat{EE}_i

\frac1S
\sum_sE_{s,i}

$$

is unbiased for:

$$

EE(t_i).

$$

For a weighted estimator:

$$

# \widehat{EE}_i

\sum_sw_sE_{s,i},

$$

the estimator is valid only if the scenario law and weighting scheme jointly target the desired expectation.

For importance sampling, weights may represent likelihood-ratio corrections such as:

$$

w_s
\propto
\frac{dQ^N}{dQ^{IS}}(X_s).

$$

Self-normalised weights do not generally preserve unbiasedness and must not be described as ordinary unbiased Monte Carlo weights.

The generation-one implementation may restrict itself to ordinary equal-weight Monte Carlo.

---

# 36. Monte Carlo independence, missing scenarios, and failure policy

Scenario observations need not be independent for the weighted expectation to be mathematically meaningful, but sampling error depends on their dependence structure.

For IID equal-weight Monte Carlo:

$$

# SE(\widehat{EE}_i)

\sqrt{
\frac{\widehat{\operatorname{Var}}(E_i)}{S}
}.

$$

For dependent scenarios, the estimator variance must account for cross-scenario covariance.

Generation-one numerical boundaries shall fail fast on:

- zero scenarios;
- missing scenario values;
- NaN valuations;
- NaN exposures;
- infinite valuations;
- malformed weights;
- negative weights;
- weights not satisfying the declared weighting contract.

No scenario dropping, `nanmean`, silent reweighting, or sentinel substitution is permitted.

---

# 37. Monte Carlo versus time-discretisation error

The production implementation shall distinguish at least:

1. valuation-model error;
2. Monte Carlo sampling error;
3. time-grid/discretisation error;
4. credit-curve/interpolation error;
5. discount-curve/interpolation error;
6. proxy/regression error, where applicable.

These shall not be collapsed into one unexplained tolerance.

For a Monte Carlo CVA calculation, the scenario-level CVA estimator should be capable of supporting an estimated sampling error where practical.

---

# 38. First-generation deterministic reference calculation

Given:

- dates $0=t_0<t_1<\cdots<t_N=T$;
- clean future valuations;
- deterministic $DF_i$;
- deterministic recovery;
- deterministic survival;
- equal-weight pricing-measure scenarios;

calculate:

### Step 1 — netting

$$

# V_{\mathcal N,s,i}^{clean}

\sum_{j\in\mathcal N}
V_{j,s,i}^{clean}.

$$

### Step 2 — exposure

For generation one:

$$

# E_{s,i}

\max(V_{\mathcal N,s,i}^{clean},0).

$$

### Step 3 — expected exposure

$$

# EE_i

\frac1S
\sum_sE_{s,i}.

$$

### Step 4 — default mass

$$

# \Delta PD_i

S_{i-1}-S_i.

$$

### Step 5 — LGD

$$

LGD=1-R_C.

$$

### Step 6 — bucket CVA

$$

# CVA_i

LGD,DF_i,EE_i,\Delta PD_i.

$$

### Step 7 — aggregate

$$

\boxed{
CVA
===

\sum_iCVA_i.
}

$$

This is the canonical golden oracle for the first-generation CVA integrator.

---

# 39. Discrete CVA integration contract

With:

$$

0=t_0<t_1<\cdots<t_N=T,

$$

the first-generation right-endpoint approximation is:

$$

\boxed{
CVA_0^{(N)}
===========

\sum_{i=1}^{N}
LGD_i
DF_i
EE_i
\Delta PD_i.
}

$$

For generation one:

$$

LGD_i=1-R_C.

$$

where:

$$

# \Delta PD_i

S_C(t_{i-1})-S_C(t_i).

$$

Bucket contribution:

$$

\boxed{
CVA_i
=====

LGD_iDF_iEE_i\Delta PD_i.
}

$$

Cumulative profile:

$$

# CVA_{\le i}

\sum_{j=1}^{i}CVA_j.

$$

Aggregate:

$$

\boxed{
CVA
===

\sum_{i=1}^{N}CVA_i.
}

$$

The aggregate is mathematically identical to the sum of bucket contributions, subject to finite-precision reconciliation tolerance in implementation.

---

# 40. Numerical discretisation and convergence

The bucketed expression is not the continuous definition:

$$

CVA_0^{(N)}
\approx
CVA_0.

$$

Under an absolutely continuous default law and sufficiently regular integrand:

$$

# f(t)

DF(t),LGD(t),EE(t)\lambda_C(t)S_C(t),

$$

the right-endpoint approximation converges under grid refinement when the relevant Riemann/Stieltjes integral is well defined.

Under sufficient smoothness/Lipschitz regularity, a right-endpoint scheme has first-order global discretisation error:

$$

O(\Delta t).

$$

This is not asserted universally in the presence of jumps, discontinuities, or default atoms.

Where exposure has jumps, the event-order convention must determine the representative value.

Where the default law has atoms, the atomic contribution must be represented separately or through the general default measure.

---

# 41. Bucket/profile output contract

The engine shall expose at least:

$$

\boxed{
\left{
t_i,
EE_i,
S_i,
\Delta PD_i,
DF_i,
LGD_i,
CVA_i
\right}_{i=1}^{N}
}

$$

and the aggregate:

$$

CVA=\sum_iCVA_i.

$$

Recommended semantic fields:

| Field | Mathematical meaning |
|---|---|
| `date` | $t_i$ |
| `expected_exposure` | $EE_i$ |
| `survival_probability` | $S_C(t_i)$ |
| `incremental_default_probability` | $\Delta PD_i$ |
| `discount_factor` | $DF_i$ |
| `lgd` | $LGD_i$ |
| `cva_contribution` | $CVA_i$ |
| `cva_cumulative` | $\sum_{j\le i}CVA_j$ |

Exact storage technology is an implementation concern.

Mathematical semantics are not.

---

# 42. Numerical tolerance and reconciliation policy

Mathematical identities shall not be confused with floating-point equality.

Examples:

$$

\sum_i\Delta PD_i=1-S_N,

$$

and:

$$

CVA=\sum_iCVA_i.

$$

Production reconciliation shall use a declared tolerance of the form:

$$

\boxed{
|x-y|
\le
\epsilon_{abs}
+
\epsilon_{rel}|y|.
}

$$

Tolerance values belong to numerical policy, not the economic definition.

They shall not be used to repair invalid inputs.

For example, the implementation shall not transform:

$$

\Delta PD_i<0

$$

into:

$$

\max(\Delta PD_i,0)

$$

merely because the value is "small."

Likewise it shall not silently transform:

$$

CVA_i<0

$$

into:

$$

\max(CVA_i,0).

$$

A negative bucket contribution outside an explicitly declared floating-point tolerance is a model/integration failure.

---

# 43. Mathematical invariants

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

0\le\sum_i\Delta PD_i\le1,

$$

and:

$$

\sum_i\Delta PD_i=1-S_N

$$

within declared numerical tolerance.

## Recovery/LGD

$$

0\le R_C\le1,

$$
$$

0\le LGD_i\le1.

$$

## Discounting

$$

DF_i>0.

$$

## Time grid

$$

t_i>t_{i-1}.

$$

## Bucket CVA

Under the first-generation assumptions:

$$

CVA_i\ge0

$$

within the declared floating-point tolerance.

## Aggregate CVA

$$

CVA\ge0

$$

within the declared numerical tolerance.

## Adjusted value

$$

# V_0^{CVA}

V_0^{clean}-CVA
\le
V_0^{clean}.

$$

---

# 44. Monotonicity properties

Holding all other quantities fixed:

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

LGD_A\le LGD_B,

$$

then:

$$

\boxed{
CVA_A\le CVA_B.
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

# 45. Golden and edge-case tests

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

## Case B — zero cumulative default probability

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

# CVA

LGD,E_0
\sum_i\Delta PD_i.

$$

Since:

$$

\sum_i\Delta PD_i=PD_C(T),

$$

we obtain:

$$

\boxed{
CVA
===

LGD,E_0,PD_C(T).
}

$$

Here $PD_C(T)$ is the **cumulative** default probability.

An incremental default probability cannot be substituted for $PD_C(T)$ in this expression without changing the calculation.

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
CVA=LGD,E_0.
}

$$

A default atom at $T$ belongs to the final bucket because:

$$

(t_{N-1},T]

$$

contains $T$.

## Case F — netting

If:

$$

V_A=100,\qquad V_B=-90,

$$

then:

$$

V_{net}=10,\qquad E_{net}=10.

$$

If:

$$

V_A=100,\qquad V_B=-100,

$$

then:

$$

V_{net}=0,\qquad E_{net}=0.

$$

## Case G — perfect deterministic oracle

For deterministic:

$$

EE_i,\quad DF_i,\quad LGD_i,\quad\Delta PD_i,

$$

compare production output against:

$$

\boxed{
\sum_iLGD_iDF_iEE_i\Delta PD_i.
}

$$

The production result must match within a pre-declared numerical tolerance.

## Case H — collateral sign

For:

$$

V=100,

$$

verify:

$$

M=80
\Rightarrow
E=20,

$$

while:

$$

M=-80
\Rightarrow
E=180.

$$

## Case I — zero scenarios

If:

$$

S=0,

$$

the expectation estimator is undefined.

Therefore:

$$

\boxed{
\text{FAIL FAST}.
}

$$

## Case J — empty netting set

For:

$$

\mathcal N=\varnothing,

$$

the mathematical aggregation is:

$$

V_{\mathcal N}=0,

$$

therefore:

$$

E_{\mathcal N}=0,

$$

and:

$$

\boxed{
CVA=0.
}

$$

The implementation may flag an empty legal netting set diagnostically, but shall not manufacture exposure.

## Case K — zero horizon

If:

$$

T=0

$$

and generation-one assumptions exclude an atom of default at time zero:

$$

\boxed{
CVA=0.
}

$$

---

# 46. Scenario-weight sanity cases

For weights:

$$

w_s\ge0,
\qquad
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

Scenario permutation must not change the result when the associated weights are permuted consistently.

---

# 47. Additivity boundary

CVA is additive across genuinely separate valuation units only when the total loss decomposes accordingly.

For disjoint netting sets $\mathcal N_k$ satisfying:

$$

# \mathcal L_{portfolio}

\sum_k\mathcal L_k,

$$

linearity of expectation gives:

$$

\boxed{
CVA_{portfolio}
===============

\sum_kCVA_{\mathcal N_k}.
}

$$

No independence assumption between netting sets is required for this linearity result.

However, trade-level additivity within one netting set does not generally hold because:

$$

\left(\sum_jV_j\right)^+
\ne
\sum_jV_j^+.

$$

Cross-default clauses do not by themselves determine additivity. The decisive question is whether the legal/closeout model defines a loss decomposition into genuinely separate valuation units.

Do not expose a generic "trade CVA sum" abstraction as though it were the primitive economic calculation.

A trade-level CVA allocation may exist only as an explicitly defined decomposition of the netting-set CVA.

---

# 48. Wrong-way risk contract

WWR exists when exposure and counterparty credit quality are dependent.

Define a **pre-default reference filtration**:

$$

\boxed{
\mathcal H_t
============

\text{market/credit information available before the default event is revealed}.
}

$$

The key requirement is that $\mathcal H_t$ must not already contain the realised event:

$$

\mathbf1_{{\tau_C\le t}}.

$$

Otherwise:

$$

# Q^N(\tau_C>t\mid\mathcal H_t)

\mathbf1_{{\tau_C>t}},

$$

which is a realised survival indicator rather than the conditional survival probability required for the intensity representation.

Define:

$$

\boxed{
G_t
===

Q^N(\tau_C>t\mid\mathcal H_t).
}

$$

For a pure market-conditioned credit model:

$$

\mathcal F_t^M\subseteq\mathcal H_t.

$$

Under a Cox/intensity construction, $\lambda_C(t)$ is the pre-default intensity, adapted/predictable with respect to $\mathcal H_t$.

Then:

$$

\boxed{
G_t
===

\exp
\left(
-\int_0^t
\lambda_C(u),du
\right).
}

$$

The joint CVA representation is:

$$

\boxed{
CVA_0
=====

E^{Q^N}
\left[
\int_0^T
D_t
\mathcal L_t
\lambda_C(t)
G_t
,dt
\right].
}

$$

Under simple recovery:

$$

\mathcal L_t=LGD_tE_t,

$$

so:

$$

\boxed{
CVA_0
=====

E^{Q^N}
\left[
\int_0^T
D_t
LGD_t
E_t
\lambda_C(t)
G_t
,dt
\right].
}

$$

There is no second multiplication by $E_t$.

Under genuine WWR:

$$

EE^{surv}(t)
\ne
EE(t)

$$

in general.

The independent representation:

$$

EE(t)\lambda_C(t)S_C(t)

$$

is therefore not generally valid.

---

# 49. WWR with default atoms

If the stochastic credit model has default atoms, the ordinary:

$$

\lambda_tG_tdt

$$

representation is insufficient on its own.

The general representation must include the corresponding atomic default probabilities.

Conceptually:

$$

\boxed{
CVA
===

E^{Q^N}
\left[
\int
D_t\mathcal L_t,d\Lambda_t^{cont}
\right]
+
\sum_k
E^{Q^N}
[
D_{t_k}\mathcal L_{t_k}
\mathbf1_{{\tau_C=t_k}}
].
}

$$

The exact compensator representation belongs to the stochastic-credit model.

The generation-one engine need not implement this.

---

# 50. WWR and collateral seam

Under future WWR/collateral extensions, the exposure state may depend jointly on:

- market state;
- credit state;
- collateral state;
- legal agreement;
- margin timing.

The minimum conceptual seam becomes:

$$

\boxed{
(\text{market state},
\text{credit state},
\text{collateral state},
\text{agreement})
\rightarrow
U_t
\rightarrow
E_t
\rightarrow
\mathcal L_t.
}

$$

The first-generation model uses only:

$$

M_t=0.

$$

---

# 51. Gaussian copula and other WWR models

A Gaussian copula may be used as part of a valid WWR model if it defines an explicit joint law.

That requires specification of:

- exposure-state marginal law;
- credit/default marginal law;
- copula structure;
- dependence parameters;
- mapping from latent variables to default time/intensity;
- pricing measure;
- simulation/estimation procedure.

A scalar `correlation` parameter alone is not a WWR model.

The contract therefore permits copula, factor, intensity, structural, or other joint-law constructions provided they explicitly define the joint exposure-credit law.

---

# 52. Bilateral/DVA boundary

The first implementation is unilateral:

$$

\boxed{
\text{counterparty default is modelled; our default is not}.
}

$$

Do not silently add DVA.

A later bilateral model requires explicit treatment of:

- counterparty default;
- own default;
- first-to-default ordering;
- bilateral closeout;
- own recovery;
- counterparty recovery;
- collateral return/closeout treatment.

The textbook decomposition:

$$

V_{bilateral}
\approx
V_{clean}-CVA+DVA

$$

is valid only under specified assumptions and closeout conventions.

It is not a universal identity for arbitrary bilateral closeout models.

DVA is therefore a later model extension, not a boolean attached to an independent-CVA scalar.

---

# 53. XVA boundary

“XVA” shall mean a family of valuation adjustments rather than a universal additive identity.

Common labels include:

- CVA — counterparty credit;
- DVA — own credit;
- FVA — funding;
- MVA — initial-margin funding;
- KVA — capital cost.

The mathematical interactions among these adjustments depend on:

- valuation framework;
- funding policy;
- collateral/margin model;
- closeout;
- default dependence;
- recursion/coupling.

Therefore:

$$

# XVA

CVA+DVA+FVA+MVA+KVA+\cdots

$$

shall not be treated as a universal mathematical contract.

RiskQuant shall not create a generic additive XVA interface before unilateral CVA is independently correct.

---

# 54. Architectural mathematical seams

The preferred decomposition is:

## Seam A — future-state generation

$$

\boxed{
\text{market inputs}
\rightarrow
X_{s,i}
}

$$

This layer determines the future scenario state and its probability-measure semantics.

## Seam B — clean future valuation

$$

\boxed{
(X_{s,i},t_i)
\rightarrow
V_{s,i}^{clean}
}

$$

This layer owns product/market valuation and event-aware future value.

## Seam C — contractual/netting aggregation

$$

\boxed{
{V_{j,s,i}^{clean}}*{j\in\mathcal N}
\rightarrow
V*{\mathcal N,s,i}^{clean}
}

$$

This layer owns legal membership and aggregation.

## Seam D — closeout/collateral exposure

$$

\boxed{
(V_{\mathcal N,s,i}^{clean},
M_{s,i},
\text{agreement},
\text{event state})
\rightarrow
U_{s,i}
\rightarrow
E_{s,i}=U_{s,i}^+
}

$$

The generation-one implementation is:

$$

M_{s,i}=0,

$$

and:

$$

U_{s,i}=V_{\mathcal N,s,i}^{clean}.

$$

## Seam E — exposure aggregation

$$

\boxed{
{E_{s,i}}_s
\rightarrow
EE_i
}

$$

This layer owns scenario weighting and estimator semantics.

## Seam F — credit curve/model

Generation one:

$$

\boxed{
\text{credit inputs}
\rightarrow
S_C(t_i),
PD_C(t_i),
\Delta PD_i,
\lambda_C(t_i)
}

$$

Future WWR:

$$

\boxed{
(\text{market/credit state})
\rightarrow
\lambda_t,G_t
}

$$

The stochastic-credit seam is therefore a generalisation, not merely a richer deterministic credit curve.

## Seam G — discount curve

$$

\boxed{
\text{discount inputs}
\rightarrow
DF_i
}

$$

This layer owns discount-curve interpolation and domain validation.

## Seam H — CVA integration

$$

\boxed{
(EE_i,DF_i,LGD_i,\Delta PD_i)
\rightarrow
CVA_i
\rightarrow
CVA
}

$$

The CVA kernel contains no legal, valuation, market-data, curve, or date semantics.

---

# 55. Kernel doctrine

Numerical kernels shall perform only declared mathematical transformations.

The first-generation CVA kernel conceptually performs:

$$

\boxed{
CVA_i
=====

LGD_iDF_iEE_i\Delta PD_i.
}

$$

Inputs are already:

- validated;
- finite;
- numerically represented;
- aligned to one integration grid;
- expressed under one declared mathematical convention.

The kernel shall not perform:

- legal netting discovery;
- counterparty classification;
- collateral agreement lookup;
- credit-curve calibration;
- CDS convention selection;
- measure selection;
- numeraire selection;
- currency conversion;
- date alignment;
- interpolation;
- extrapolation;
- event-order interpretation;
- market-data retrieval;
- business-rule interpretation;
- object traversal;
- string-based dispatch;
- hidden sign correction.

---

# 56. Fail-fast contract

Invalid mathematical inputs shall not be silently repaired.

Examples include:

- recovery outside $[0,1]$;
- LGD outside $[0,1]$;
- non-positive discount factors;
- survival outside $[0,1]$;
- increasing survival probabilities;
- negative default mass;
- inconsistent $PD$ and $S$;
- invalid hazard rates;
- scenario weights negative;
- invalid weighting normalisation;
- zero scenarios;
- ambiguous scenario/date dimensions;
- mismatched grids;
- duplicate dates where prohibited;
- non-monotone dates;
- dates outside supported curve domains;
- future valuation data contaminated by CVA/DVA;
- physical-measure probability data supplied to a pricing contract without explicit transformation;
- inconsistent valuation currency;
- inconsistent numeraire metadata;
- inconsistent time basis;
- NaN/None sentinels inside numerical kernels;
- infinite exposure or valuation;
- missing scenarios;
- unsupported default/event atoms passed into a generation-one intensity integrator.

No silent clipping is permitted.

In particular:

$$

S_i
\leftarrow
\min(1,\max(0,S_i))

$$

and:

$$

\Delta PD_i
\leftarrow
\max(0,\Delta PD_i)

$$

are prohibited repairs unless a distinct, explicitly documented mathematical projection model is being implemented.

---

# 57. No silent convention changes

The following are part of the mathematical contract and must be externally inspectable:

- valuation perspective;
- pricing measure;
- numeraire;
- valuation currency;
- closeout currency;
- legal netting scope;
- collateral sign convention;
- collateral currency conversion;
- closeout convention;
- recovery convention;
- probability representation;
- survival/default representation;
- hazard representation;
- curve interpolation;
- curve extrapolation policy;
- exposure grid;
- credit grid;
- discount grid;
- internal time coordinate;
- day-count conversion;
- event ordering;
- default bucket convention;
- endpoint convention;
- scenario weighting;
- scenario measure;
- scenario/time axis semantics;
- numerical reconciliation tolerances.

Changing any of these may change the mathematical quantity being priced.

---

# 58. No double-counting contract

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

At the general loss level:

$$

\mathcal L_t

$$

already denotes the loss amount.

Therefore the stochastic-credit formulation is:

$$

D_t\mathcal L_t\lambda_tG_t,

$$

not:

$$

D_t\mathcal L_tE_t\lambda_tG_t.

$$

Under simple recovery:

$$

\mathcal L_t=LGD_tE_t.

$$

Therefore:

$$

D_tLGD_tE_t\lambda_tG_t

$$

is the correct expanded expression.

---

# 59. Anti-patterns

## Anti-pattern 1 — scalar CVA shortcut

$$

CVA=LGD\times EPE\times PD

$$

with no explicit time profile.

## Anti-pattern 2 — today's exposure

$$

CVA=LGD\times V_0^+\times PD(T).

$$

## Anti-pattern 3 — historical P&L substitution

Using historical VaR P&L scenarios as future CVA exposures without proving future-state, measure, and valuation compatibility.

## Anti-pattern 4 — trade-first positive part

$$

\sum_jV_j^+

$$

for a legally netted set.

## Anti-pattern 5 — counterparty equals netting set

Using counterparty ID alone to determine legal netting.

## Anti-pattern 6 — CDS spread equals hazard

Treating CDS spread as $\lambda$ by identity.

## Anti-pattern 7 — arbitrary WWR multiplier

Applying an arbitrary correlation scalar to independent CVA.

## Anti-pattern 8 — universal additive XVA

Treating CVA/DVA/FVA/MVA/KVA as guaranteed independent additive scalars.

## Anti-pattern 9 — hidden default bucket rule

Applying $\Delta PD$ to an exposure date without defining the default-time approximation.

## Anti-pattern 10 — hidden stochastic discounting factorisation

Using $DF\times EE$ where discounting and exposure are stochastic without proving the required factorisation.

## Anti-pattern 11 — credit-contaminated clean value

Supplying CVA/DVA-adjusted future valuations into the exposure engine.

## Anti-pattern 12 — semantics-free collateral subtraction

Subtracting a scalar called "collateral" without agreement, sign, currency, timing, and legal meaning.

## Anti-pattern 13 — default-revealing filtration

Conditioning survival on an information set that already reveals whether default occurred.

## Anti-pattern 14 — unconditional EE under WWR

Using:

$$

E[E_t]

$$

as though it were always:

$$

E[E_t\mid\tau_C>t].

$$

## Anti-pattern 15 — invalid hazard representation

Forcing a survival curve containing atoms into:

$$

dPD=\lambda S,dt.

$$

## Anti-pattern 16 — silent curve extrapolation

Extending credit or discount curves beyond their validated domain.

## Anti-pattern 17 — hidden curve interpolation inside the CVA kernel

Performing date lookup or interpolation inside the numerical integration primitive.

## Anti-pattern 18 — ambiguous event ordering

Using an unspecified $V(t_i)$ around a cashflow, exercise, settlement, or maturity discontinuity.

## Anti-pattern 19 — historical weights as pricing weights

Using weights that represent $\mathbb P$ while treating them as $\mathbb Q^N$ pricing probabilities.

## Anti-pattern 20 — silent scenario dropping

Dropping NaN/missing scenarios and renormalising remaining weights.

## Anti-pattern 21 — tolerance as repair

Clipping mathematically invalid quantities merely because the violation is numerically small.

## Anti-pattern 22 — currency/numeraire conflation

Treating a reporting currency label as though it uniquely defines the pricing numeraire and discounting convention.

---

# 60. Validation hierarchy

Validation proceeds from mathematics outward.

## Level A — dimensions and scalar arithmetic

Verify units and the CVA integration kernel against hand calculations.

## Level B — measure and numeraire metadata

Verify that:

- scenario measure is declared;
- credit measure is declared;
- valuation numeraire is declared;
- valuation currency is declared;
- no physical-to-pricing substitution is implicit.

## Level C — credit curve invariants

Verify:

$$

S_0=1,

$$
$$

0\le S_i\le1,

$$
$$

\Delta PD_i\ge0.

$$

## Level D — exposure transformation

Verify positive-part and collateral semantics.

## Level E — netting

Verify aggregation before positive-part transformation.

## Level F — Monte Carlo expectation

Verify weighting, measure semantics, permutation invariance, sampling-error behaviour, and convergence.

## Level G — CVA integration

Verify profile and aggregate against an independent golden oracle.

## Level H — temporal discretisation

Verify convergence as the integration grid is refined.

## Level I — event discontinuities

Verify the selected pre-event/closeout convention around cashflows, exercise, settlement, and maturity.

## Level J — curve alignment/interpolation

Verify that credit and discount values are produced on the integration grid under the declared conventions and without hidden interpolation in the integrator.

## Level K — extensions

Only after Levels A–J are robust should the engine introduce stochastic discounting, complex collateral, WWR, bilateral default, or broader XVA features.

---

# 61. Grid-refinement validation

For nested grids satisfying:

$$

\Delta t_{\max}^{(1)}

>

\Delta t_{\max}^{(2)}

>

\cdots,

$$

the resulting CVA values should converge consistently with the selected numerical scheme and underlying regularity.

The test should detect:

- gross non-convergence;
- date-alignment errors;
- event-order errors;
- interpolation inconsistency;
- incorrect default-mass allocation;
- discontinuity artefacts.

No universal convergence rate is prescribed because the rate depends on:

- exposure regularity;
- discount-curve regularity;
- credit-curve regularity;
- event discontinuities;
- discretisation scheme.

---

# 62. Independent oracle requirement

The production implementation shall be validated against an independently constructed oracle wherever practical.

The oracle should not share the production implementation path, especially not:

- the same integration loop;
- the same interpolation helper;
- the same default-mass construction;
- the same scenario aggregation function.

For simple deterministic cases, the oracle may be a hand-derived scalar or independently evaluated expression.

The objective is to detect shared implementation errors, not merely reproduce them.

---

# 63. Monte Carlo convergence reporting

Where Monte Carlo is used, the implementation should be able to report an estimate of sampling uncertainty.

For IID equal-weight exposure at date $t_i$:

$$

# \widehat{SE}(\widehat{EE}_i)

\sqrt{
\frac{
\widehat{\operatorname{Var}}(E_{\cdot,i})
}{S}
}.

$$

Where practical, the CVA estimator should also support scenario-level sampling-error estimation.

Monte Carlo error must not be confused with time-discretisation error.

A result that changes under increasing $S$ is not evidence that the time-grid scheme is wrong.

A result that changes materially under grid refinement is not evidence that Monte Carlo sampling error is large.

These mechanisms must be diagnosed separately.

---

# 64. Reference example

Suppose:

| Date | Expected Exposure | Incremental PD | Discount Factor |
|---|---:|---:|---:|
| 1Y | €10,000,000 | 0.50% | 0.97 |
| 2Y | €8,000,000 | 0.75% | 0.94 |
| 3Y | €6,000,000 | 1.00% | 0.91 |

and:

$$

R_C=40%,

$$

so:

$$

LGD=60%.

$$

Then:

$$

# CVA_1

0.60
\times
0.97
\times
10{,}000{,}000
\times
0.005
=====

€29{,}100,

$$
$$

# CVA_2

0.60
\times
0.94
\times
8{,}000{,}000
\times
0.0075
======

€33{,}840,

$$
$$

# CVA_3

0.60
\times
0.91
\times
6{,}000{,}000
\times
0.01
====

€32{,}760.

$$

Therefore:

$$

\boxed{
CVA=€95{,}700.
}

$$

The result shall be reproducible independently of the production implementation.

---

# 65. Code-archaeology contract

Before adding any new abstraction, inspect the existing RiskQuant codebase and determine whether the required mathematical primitive already exists.

At minimum, investigate:

1. future market-state generation;
2. future-dated valuation;
3. portfolio/trade valuation;
4. contractual/netting-set aggregation;
5. currency/FX conversion;
6. collateral representation;
7. discounting;
8. scenario weights;
9. scenario measure metadata;
10. survival/default curves;
11. hazard-rate representation;
12. exposure aggregation;
13. vectorised scenario × time storage;
14. deterministic integration;
15. independent golden-oracle infrastructure;
16. invariant enforcement;
17. date/time-coordinate handling;
18. curve interpolation/extrapolation policy;
19. clean-versus-credit-adjusted valuation semantics.

For every candidate abstraction, answer:

- What exact mathematical object does it represent?
- What are its units?
- What is its pricing measure?
- What is its numeraire?
- What is its currency?
- What is its sign convention?
- At what time does it live?
- Is it current-state, historical-scenario, or future-state information?
- Is it trade-level, netting-set-level, counterparty-level, or portfolio-level?
- What assumptions does it encode?
- What invariants does it guarantee?
- Does it represent clean value or credit-adjusted value?
- What interpolation/extrapolation conventions does it encode?
- Can it satisfy the CVA contract without semantic distortion?

A type/name/shape match is not evidence of mathematical compatibility.

---

# 66. Required archaeological challenge questions

Before proposing a CVA architecture, the reviewer shall be able to answer:

## Future valuation

Can existing RiskQuant valuation infrastructure produce:

$$

V^{clean}(X_t,t)

$$

at arbitrary future dates under simulated future states?

If not, exactly which seam is missing?

## Probability measure

Where is $\mathbb Q^N$ established?

What object records its measure semantics?

## Numeraire

What is:

$$

N_t?

$$

What currency is the valuation expressed in?

How are the two distinguished?

## Discounting

What exactly produces:

$$

DF(t)?

$$

What curve convention and collateral/pricing convention does it represent?

## Netting

Where is legal netting scope represented?

What proves membership?

## Currency

Where is contractual FX conversion defined?

What happens before netting?

## Collateral

Does an existing abstraction represent contractual collateral semantics, or merely a number called "margin"?

## Credit

Does RiskQuant already represent:

$$

S,\quad PD,\quad\Delta PD,\quad\lambda?

$$

What does each object actually mean?

## Time

Can the existing scenario engine represent:

$$

(s,t)

$$

rather than merely scenario snapshots around today?

What is the canonical internal time coordinate?

## Event ordering

Can future valuation distinguish:

$$

V(t^-)
\quad\text{from}\quad
V(t^+)?

$$

If not, exactly which seam is missing?

## Kernel seam

Can the CVA integration be implemented as pure numerical arithmetic over aligned validated arrays?

## Reuse

Can an existing abstraction be reused **without weakening or changing its mathematical semantics**?

If not, reuse is prohibited even if the code shape looks convenient.

---

# 67. First-generation mathematical API contract

The exact Python/API shape is an implementation choice.

Semantically, the first-generation computation must represent:

$$

\boxed{
(EE_i,DF_i,LGD_i,\Delta PD_i)*{i=1}^N
\rightarrow
(CVA_i)*{i=1}^N
\rightarrow
CVA.
}

$$

The exposure engine separately represents:

$$

\boxed{
V_{s,i}^{clean}
\rightarrow
E_{s,i}
\rightarrow
EE_i.
}

$$

The credit layer separately represents:

$$

\boxed{
\text{credit inputs}
\rightarrow
S_i,PD_i,\Delta PD_i,\lambda_i.
}

$$

The discount layer separately represents:

$$

\boxed{
\text{discount inputs}
\rightarrow
DF_i.
}

$$

The scenario layer separately represents:

$$

\boxed{
\text{scenario law}
+
\text{weights}
\rightarrow
\widehat{EE}.
}

$$

These are mathematical boundaries, not merely object-oriented preferences.

---

# 68. Precision and numerical-policy requirements

The mathematical contract assumes sufficient numerical precision to preserve:

- sign;
- monotonicity;
- probability mass;
- profile reconciliation;
- integration accuracy.

No NaN, None, magic sentinel, or silent clipping may substitute for a missing or invalid mathematical quantity inside numerical kernels.

Where a finite-precision tolerance is mathematically appropriate, it shall be used only for validation/reconciliation.

It shall not alter the underlying supplied quantity.

---

# 69. Controlled evolution path

The preferred development path is:

## Level 1 — Exposure-profile integration

Input:

$$

{t_i,EE_i,DF_i,LGD_i,\Delta PD_i}

$$

Calculate:

$$

# CVA

\sum_iLGD_iDF_iEE_i\Delta PD_i.

$$

This isolates and proves the credit-weighted integration mathematics.

## Level 2 — Future-state simulation

Generate:

$$

X_{s,i}

$$

then:

$$

V_{s,i}^{clean}.

$$

then:

$$

E_{s,i}.

$$

then:

$$

EE_i.

$$

## Level 3 — Contractual netting and simple collateral

Determine:

$$

V_{\mathcal N,s,i}^{clean},

$$

then:

$$

U_{s,i},

$$

then:

$$

E_{s,i}.

$$

## Level 4 — Stochastic credit / WWR

Introduce:

$$

\lambda_t,
\quad
G_t,
\quad
\mathcal H_t,

$$

and replace factorised default weighting with the joint exposure-credit law:

$$

E^{Q^N}
\left[
\int
D_t
\mathcal L_t
\lambda_tG_tdt
\right].

$$

## Level 5 — Bilateral credit

Introduce own default, first-to-default effects, and explicit bilateral closeout.

## Level 6 — Broader XVA

Only then consider funding, initial-margin funding, capital, and coupled XVA effects.

Each level must preserve the semantics of the preceding levels and make every new assumption explicit.

---

# 70. Absolute implementation doctrine

The following statements are normative.

**The general CVA definition is expected discounted counterparty-default loss under an explicitly selected pricing measure.**

**CVA is a valuation quantity, not inherently a VaR/PFE/ES-style risk statistic.**

**A numeraire is a value process, not a currency label.**

**The pricing measure, numeraire, valuation currency, and discount factor must remain distinct but consistent.**

**The first implementation is unilateral CVA, not general XVA.**

**Clean future valuation must contain no counterparty-credit adjustment.**

**Netting must occur before the positive-part exposure transformation.**

**Collateral semantics must be explicit; initial margin is not automatically subtractable.**

**The canonical collateral sign is \(M>0\) for collateral held by us and \(M<0\) for collateral posted by us.**

**Risk-neutral pricing probabilities and physical probabilities must never be silently interchanged.**

**Unconditional EE and survival-conditioned EE are distinct mathematical quantities.**

**Their equality is a consequence of the first-generation independence model, not a definition.**

**The market filtration used for the independence condition must contain all exposure-relevant market information.**

**The pre-default survival filtration must not reveal the realised default event.**

**The standard \(DF\times EE\times\Delta PD\) formula is a factorised specialisation under explicit assumptions.**

**The bucketed sum is a numerical approximation to the continuous/default-measure formulation.**

**Default at \(t_i\) belongs to bucket \((t_{i-1},t_i]\).**

**The generation-one endpoint convention uses the explicitly declared pre-contractual-event closeout state at \(t_i\) when a discontinuity occurs.**

**A survival curve, cumulative PD, bucket default mass, and hazard rate are distinct mathematical objects.**

**Hazard-rate formulas require the corresponding intensity representation; default atoms must be represented through the default measure.**

**Credit and discount interpolation belong to their respective curve layers, not the CVA kernel.**

**Silent extrapolation beyond curve domains is prohibited.**

**A Monte Carlo sample mean is an estimator, not the mathematical definition of expected exposure.**

**Scenario measure and weighting jointly determine the estimator semantics.**

**WWR is an explicit joint exposure-credit model, not an arbitrary correlation multiplier.**

**CVA, DVA, FVA, MVA, and KVA are not assumed universally additive.**

**Default loss and exposure are distinct objects.**

**Do not double-count default or exposure.**

**Do not infer semantics from names, shapes, or architectural convenience.**

**Kernels perform declared mathematics. Boundaries establish assumptions and invariants.**

**Fail fast when the mathematical contract is violated.**

---

# 71. Completion criterion for the first CVA engine

The first RiskQuant CVA implementation is not complete merely because it produces a plausible scalar.

It is complete only when it can demonstrate all of the following:

1. The implementation states exactly which mathematical quantity it calculates.
2. The first-generation assumptions are explicit and externally inspectable.
3. The pricing measure is explicit.
4. The numeraire is explicit and distinct from valuation currency.
5. The discounting convention is explicit.
6. Clean future valuation is distinct from historical P&L and credit-adjusted valuation.
7. Legal netting scope is established before exposure truncation.
8. Currency/FX conversion before netting is explicit.
9. Exposure semantics and sign conventions are explicit.
10. The generation-one closeout/recovery mapping is explicit.
11. Variation margin, initial margin, and independent amount are not semantically conflated.
12. Credit curves are represented as mathematically defined survival/default/hazard quantities.
13. The credit model states its domain and interpolation convention.
14. Default bucket timing is explicit.
15. Event ordering at contractual discontinuities is explicit.
16. The internal time coordinate is explicit.
17. Scenario measure and weighting semantics are explicit.
18. The scalar CVA matches an independent oracle.
19. The bucket profile reconciles to the aggregate within declared numerical tolerance.
20. Mathematical invariants are enforced or tested.
21. Netting-before-positive-part behaviour is independently tested.
22. Collateral sign behaviour is independently tested.
23. Grid refinement can be demonstrated.
24. Monte Carlo convergence and sampling uncertainty can be demonstrated where Monte Carlo is used.
25. Monte Carlo, time-discretisation, and curve errors are distinguished.
26. The implementation can distinguish unconditional and survival-conditioned expected exposure.
27. The WWR seam uses a pre-default filtration and explicit joint exposure-credit law.
28. Default atoms are not incorrectly forced through an ordinary hazard-density representation.
29. Curve and date semantics are established before numerical kernels execute.
30. Invalid inputs fail fast rather than being silently repaired.
31. No abstraction exists solely because the term “XVA” appears.
32. Every architectural abstraction has a defensible mathematical owner.
33. No default-loss quantity is counted twice through overlapping exposure/loss definitions.
34. The implementation can state exactly which assumptions would change for collateral, WWR, bilateral default, or broader XVA.

Only after these conditions are satisfied should RiskQuant move beyond unilateral first-generation CVA.
$$
