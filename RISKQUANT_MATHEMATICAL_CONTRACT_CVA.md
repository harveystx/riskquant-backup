# RiskQuant — Ultra-Tier Mathematical CVA Contract

**Status:** **LOCKED — Canonical mathematical specification for the first CVA implementation**

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
16. Distinguish **numeraire**, **valuation currency**, **discount factor**, and **day-count/time coordinate**.
17. Do not allow upstream curve interpolation, date alignment, FX conversion, or event-order decisions to become hidden behaviour of a numerical CVA kernel.
18. Do not present a weighted Monte Carlo average as a valid pricing estimator unless the scenario law and weighting scheme jointly define the target pricing expectation.
19. Do not silently extrapolate beyond the domain of required market or credit inputs.
20. Preserve the mathematical distinction between an actual default loss, a signed closeout amount, and positive unsecured exposure.
21. Do not infer exposure/default independence from an incomplete market filtration.
22. Do not infer CVA additivity from bookkeeping separation; additivity requires a mathematically valid loss decomposition.
23. Treat default atoms explicitly rather than forcing them into an ordinary intensity density.
24. Distinguish mathematical identities from finite-precision implementation tolerances.
25. Do not use an intensity representation unless the intensity is the appropriate pre-default compensator intensity for the chosen reference filtration.
26. Do not claim smooth discretisation order across declared discontinuities.
27. Keep legal enforceability evidence outside numerical kernels while requiring the legal layer to establish the valuation-unit decomposition consumed by CVA.
28. Never allow an information set containing the realised default event to masquerade as conditional-survival information.
29. Distinguish a deterministic hazard function from a stochastic intensity process.
30. Distinguish the unconditional deterministic default-probability measure from the pathwise stochastic default compensator.
31. Do not rely on a smooth hazard representation when survival reaches zero; use the default-probability measure representation instead.
32. Do not assume that an expanded pre-default reference filtration preserves generation-one independence; generation one requires independence with respect to the reference filtration actually used by the credit model.
33. Do not leave the side of an endpoint discontinuity implicit for exposure or discounting.
34. Do not treat collateral FX conversion as a reporting-only transformation; it must occur according to the contractual/modelled closeout convention before collateral enters the closeout equation.

---

# 1. Mathematical north star

The general unilateral CVA quantity is the expected discounted loss caused by counterparty default:

$$
\boxed{
CVA_0
=
\mathbb E^{\mathbb Q^N}
\left[
D^N(0,\tau_C)\,
\mathcal L_{\tau_C}\,
\mathbf 1_{\{\tau_C\le T\}}
\right].
}
$$

where:

* $\mathbb Q^N$ is the pricing measure associated with the explicitly selected numeraire process $N_t$;
* $\tau_C$ is counterparty default time;
* $D^N(0,t)$ is the corresponding discount-factor representation;
* $\mathcal L_t$ is the actual counterparty-default loss amount on the selected valuation/netting unit at time $t$;
* $T$ is the latest future date at which relevant closeout exposure can be non-zero under the selected contractual and valuation model.

This is the **general economic definition**.

Everything else in this document is either:

* a definition of one component;
* a modelling specialisation;
* a numerical discretisation;
* a statistical estimator;
* or a controlled future extension.

The general CVA definition is a **pricing quantity**: it is the time-0 value of a contingent default-loss claim under a pricing measure. It is not inherently a VaR, PFE, ES, or other statistical risk functional.

---

# 2. Numeraire and pricing-measure contract

A numeraire is a strictly positive value process:

$$
\boxed{
N_t>0.
}
$$

It is not a currency label.

The pricing measure associated with $N_t$ is:

$$
\boxed{
\mathbb Q^N.
}
$$

The corresponding numeraire-relative discount representation is:

$$
\boxed{
D^N(0,t)
=
\frac{N_0}{N_t}.
}
$$

The selected numeraire must be externally inspectable.

Examples may include:

* a money-market account;
* an OIS-related account;
* a collateral remuneration account under an explicitly stated collateral convention;
* another explicitly selected strictly positive tradable numeraire.

RiskQuant shall never equate:

$$
\text{currency}
\equiv
\text{numeraire}.
$$

The valuation currency is the unit in which monetary values are expressed.

The numeraire is a value process used to define the pricing measure and discounting representation.

The first-generation implementation shall select one explicit:

* valuation currency;
* closeout/aggregation currency;
* valuation numeraire;
* deterministic discounting convention.

No implementation may infer one from another.

---

# 3. First-generation production scope

The first production implementation shall be:

$$
\boxed{
\text{Unilateral, uncollateralised, deterministic-recovery, deterministic-discounting, independent-default CVA.}
}
$$

Assumptions:

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

6. Valuation currency and numeraire are explicit.
7. Discounting is deterministic.
8. The counterparty default process is independent of the future market/exposure process with respect to the **same reference filtration used by the generation-one credit model**, as defined in Sections 10–12 and 49.
9. Closeout is the generation-one clean-MTM replacement-value specialisation with instantaneous closeout and no margin-period dynamics.
10. The first generation is uncollateralised:

$$
M_t=0.
$$

11. Future clean valuation is supplied by a future-state valuation/simulation layer.
12. Credit is supplied by a risk-neutral survival/default curve or equivalent deterministic hazard representation.
13. Default is integrated on an explicit time grid using the declared bucket and event-order convention.
14. The internal time coordinate is explicit.
15. Required credit and discount inputs cover the entire integration horizon.
16. Silent extrapolation is prohibited.

The first-generation continuous-time factorisation is:

$$
\boxed{
CVA_0
=
(1-R_C)
\int_{(0,T]}
DF(t)\,
EE(t)\,
dPD_C(t).
}
$$

where:

$$
EE(t)
=
\mathbb E^{Q^N}
[
(V_t^{clean})^+
].
$$

This is a **specialisation** of the general economic definition.

The first-generation bucket approximation is:

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

This is a **numerical discretisation**, not a general CVA identity.

---

# 4. Mathematical status hierarchy

| Object                                                                      | Mathematical status                          |
| --------------------------------------------------------------------------- | -------------------------------------------- |
| $\mathbb E^{Q^N}[D^N(0,\tau_C)\mathcal L_{\tau_C}\mathbf1_{{\tau_C\le T}}]$ | General economic definition                  |
| $N_t$                                                                       | Numeraire definition                         |
| $D^N(0,t)=N_0/N_t$                                                          | Numeraire-associated discount representation |
| $U_t$                                                                       | Signed unsecured closeout amount             |
| $E_t=U_t^+$                                                                 | Positive unsecured exposure specialisation   |
| $\mathcal L_t=LGD_tE_t$                                                     | Recovery/closeout specialisation             |
| $EE(t)=\mathbb E^{Q^N}[E_t]$                                                | Unconditional expected exposure              |
| $EE^{surv}(t)=\mathbb E^{Q^N}[E_t\mid\tau_C>t]$                             | Survival-conditioned expected exposure       |
| $EE^{surv}(t)=EE(t)$                                                        | First-generation independence consequence    |
| $CVA=\int DF,EE,dPD$                                                        | Deterministic/independent factorisation      |
| $\sum_iDF_iEE_i\Delta PD_i$                                                 | Time-discretised approximation               |
| $\widehat{EE}*i=\sum_s w_sE*{s,i}$                                          | Monte Carlo estimator                        |
| $dPD_C$                                                                     | Unconditional default-probability measure    |
| $d\Lambda_t$                                                                | Stochastic default compensator               |
| $\lambda_C^{det}(t)$                                                        | Deterministic hazard function                |
| $\lambda_t$                                                                 | Stochastic pre-default intensity process     |
| $EE\times PD\times LGD$                                                     | Not a general identity                       |

---

# 5. Economic interpretation and sign

The valuation perspective is:

$$
\boxed{
V>0
\Rightarrow
\text{counterparty owes us value}.
}
$$

Under unilateral generation-one CVA:

$$
E_t=(V_t^{clean})^+.
$$

Therefore:

$$
CVA_0\ge0,
$$

and:

$$
\boxed{
V_0^{CVA}
=
V_0^{clean}-CVA_0.
}
$$

If:

$$
V^{clean}=+100,
$$

then:

$$
E=100
$$

and counterparty default can produce loss.

If:

$$
V^{clean}=-100,
$$

then:

$$
E=0
$$

and that state contributes no unilateral counterparty CVA.

No reporting convention may silently reverse this sign.

---

# 6. Measure contract

All probabilistic pricing inputs must have externally inspectable measure semantics.

These include:

* future market-state scenarios;
* scenario weights;
* expected exposure;
* survival probabilities;
* default probabilities;
* hazard/intensity models.

The following substitutions are prohibited without an explicit measure transformation:

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

A set of weights summing to one does not establish that they represent $\mathbb Q^N$.

---

# 7. Clean future valuation contract

For scenario $s$ and future date $t_i$:

$$
\boxed{
V_{s,i}^{clean}
=
V^{clean}(X_{s,i},t_i).
}
$$

Future clean valuation is from our perspective and contains no:

* CVA;
* DVA;
* FVA;
* counterparty credit adjustment;
* recovery/default-loss adjustment;
* collateral-credit adjustment.

Therefore:

$$
\boxed{
\text{clean valuation contains no counterparty credit adjustment}.
}
$$

---

# 8. Future valuation is not historical P&L

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
V_{s,i}^{clean}.
}
$$

Therefore:

$$
\boxed{
\text{historical P\&L scenario}
\not\equiv
\text{future risk-neutral exposure scenario}.
}
$$

Existing scenario machinery may be reused only if its semantics establish compatibility with:

1. future evolution;
2. $\mathbb Q^N$;
3. future clean valuation;
4. exposure time;
5. scenario weighting.

---

# 9. Currency, time, and dimensions

All values entering a mathematical netting operation must be represented in the selected closeout/valuation currency after applying the declared contractual/model FX conversion.

The fundamental distinction is:

$$
\boxed{
\text{currency}
\neq
\text{numeraire}
\neq
\text{discount factor}.
}
$$

Dimensional invariants:

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

$$
[\lambda]
=
1/\text{time}.
$$

Hence:

$$
[DF\times EE\times\Delta PD]
=
\text{currency}.
$$

---

# 10. Formal first-generation independence contract

Let:

$$
\mathcal F_t^M
$$

denote the **full market/risk-factor filtration relevant to exposure**, including all past and current market information required to determine the exposure process.

For generation one, the reference filtration used by the credit model is identified with the market filtration:

$$
\boxed{
\mathcal F_t^{ref}
=
\mathcal F_t^M.
}
$$

The generation-one default condition is therefore:

$$
\boxed{
Q^N
(
\tau_C>t
\mid
\mathcal F_t^{ref}
)
=
S_C(t)
}
$$

for every deterministic $t\in[0,T]$.

Additionally:

1. $E_t$ is $\mathcal F_t^{ref}$-measurable;
2. the default process has deterministic compensator relative to $\mathcal F_t^{ref}$;
3. that compensator has no additional dependence on market information;
4. exposure is generated from the pre-default market process.

Therefore:

$$
\boxed{
EE^{surv}(t)
=
EE(t).
}
$$

Any strictly larger reference filtration:

$$
\mathcal F_t^{ref}\supsetneq\mathcal F_t^M
$$

represents a stochastic-credit/WWR extension unless it can itself be proven to preserve deterministic conditional survival:

$$
Q^N(\tau_C>t\mid\mathcal F_t^{ref})=S_C(t).
$$

This closes the filtration relationship between generation one and WWR.

---

# 11. Independence counterexample

Let:

$$
Y\in\{-1,+1\},
\qquad
Q^N(Y=1)=Q^N(Y=-1)=\frac12.
$$

Suppose:

$$
E_t=
\begin{cases}
100,&Y=1,\\
0,&Y=-1,
\end{cases}
$$

and:

$$
Q^N(\tau_C>t\mid Y=1)=0.9,
$$

$$
Q^N(\tau_C>t\mid Y=-1)=0.5.
$$

Then:

$$
Q^N(\tau_C>t)=0.7,
$$

while:

$$
E^{Q^N}[E_t]=50,
$$

and:

$$
E^{Q^N}[E_t\mid\tau_C>t]
=
\frac{0.5\times100\times0.9}{0.7}
\approx64.286.
$$

Hence:

$$
E[E_t\mid\tau_C>t]
\ne
E[E_t].
$$

Now let $X_t$ be independent of $Y$.

Then:

$$
\begin{aligned}
Q^N(\tau_C>t\mid X_t)
&=
E^{Q^N}
[
Q^N(\tau_C>t\mid Y)
\mid X_t
]
\\
&=
E^{Q^N}
[
Q^N(\tau_C>t\mid Y)
]
\\
&=
0.7.
\end{aligned}
$$

Thus an incomplete filtration containing $X_t$ but omitting $Y$ can falsely appear to satisfy the independence condition.

Therefore:

$$
\boxed{
\mathcal F_t^M
\text{ must contain all exposure-relevant market information}.
}
$$

---

# 12. Default-time versus deterministic-time exposure

The economic object contains:

$$
E_{\tau_C},
$$

not merely:

$$
E_t.
$$

The generation-one factorisation requires:

* exposure adaptedness/measurability;
* an appropriate default compensator;
* independence of the default mechanism from market exposure;
* the declared pricing measure.

For the deterministic default-probability measure:

$$
\boxed{
CVA
=
\int_{(0,T]}
DF(t)
LGD(t)
EE(t)
\,dPD_C(t).
}
$$

The contract shall not claim:

$$
E[E_{\tau_C}\mid\tau_C\in dt]
=
EE(t)
$$

from the survival condition alone.

---

# 13. General default-loss contract

The general loss amount is:

$$
\boxed{
\mathcal L_t
=
\text{actual counterparty-default loss amount}.
}
$$

The general definition remains:

$$
\boxed{
CVA_0
=
E^{Q^N}
[
D^N(0,\tau_C)
\mathcal L_{\tau_C}
1_{\{\tau_C\le T\}}
].
}
$$

---

# 14. Closeout and unsecured exposure

Define:

$$
\boxed{
U_t
=
\text{signed unsecured closeout amount}.
}
$$

Positive unsecured exposure is:

$$
\boxed{
E_t=U_t^+.
}
$$

Generation-one closeout specialises to:

$$
\boxed{
U_t
=
V_t^{clean}-M_t.
}
$$

Thus:

$$
\boxed{
E_t
=
(V_t^{clean}-M_t)^+.
}
$$

Generation one sets:

$$
M_t=0.
$$

---

# 15. First-generation closeout specialisation

Generation one uses clean-MTM replacement-value closeout with:

1. clean value as the reference closeout value;
2. instantaneous closeout;
3. no margin-period-of-risk dynamics;
4. no liquidation-cost term;
5. no additional closeout adjustment;
6. loss only on positive unsecured closeout exposure.

Therefore:

$$
\boxed{
\mathcal L_t=LGD_tE_t.
}
$$

and:

$$
\boxed{
\mathcal L_t
=
(1-R_C)(V_t^{clean})^+.
}
$$

This is a model specialisation, not a universal closeout theorem.

---

# 16. Collateral sign and FX sequencing

The canonical signed collateral convention is:

$$
\boxed{
M_t>0
\iff
\text{collateral held by us},
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

Collateral must be represented in the same closeout currency before entering:

$$
U_t=V_t^{clean}-M_t.
$$

For collateral originally denominated in currency $c$:

$$
M_t^{(N)}
=
\operatorname{FXConvert}_{c\rightarrow N}^{closeout}(M_t^{(c)}).
$$

The canonical sequence is therefore:

$$
\boxed{
\text{trade valuation}
\rightarrow
\text{trade FX conversion}
\rightarrow
\text{legal netting}
\rightarrow
\text{collateral FX conversion}
\rightarrow
\text{closeout}
\rightarrow
\text{positive part}.
}
$$

Reporting FX conversion is not automatically the closeout FX conversion.

---

# 17. Margin taxonomy

The conceptual categories are:

$$
M_t^{VM},
\qquad
M_t^{IM},
\qquad
M_t^{IA}.
$$

Generation one uses:

$$
M_t=0.
$$

These are not semantically interchangeable.

Future closeout shall define a mapping such as:

$$
\boxed{
U_t
=
\Phi
\left(
V_t^{clean},
M_t^{VM},
M_t^{IM},
M_t^{IA},
\text{agreement},
\text{event state}
\right).
}
$$

It shall not assume:

$$
U_t
=
V_t^{clean}
-
M_t^{VM}
-
M_t^{IM}
-
M_t^{IA}.
$$

unless that mapping has been explicitly established by the closeout model.

---

# 18. Recovery and LGD

General recovery:

$$
R_t\in[0,1],
$$

with:

$$
LGD_t=1-R_t.
$$

Under simple recovery:

$$
\boxed{
\mathcal L_t=LGD_tE_t.
}
$$

Generation one:

$$
R_t=R_C,
\qquad
LGD=1-R_C.
$$

A recovery convention acting on a signed closeout amount before positive-part truncation is a different model.

---

# 19. Netting-set contract

CVA exposure is measured at the legally enforceable netting-set level.

The dependency is:

$$
\boxed{
\text{legal enforceability}
\rightarrow
\text{netting membership}
\rightarrow
\text{valuation aggregation}.
}
$$

For scenario $s$ and time $t_i$:

$$
\boxed{
V_{\mathcal N,s,i}^{clean}
=
\sum_{j\in\mathcal N}
V_{j,s,i}^{clean}.
}
$$

Then:

$$
E_{\mathcal N,s,i}
=
\left(
V_{\mathcal N,s,i}^{clean}
-
M_{\mathcal N,s,i}
\right)^+.
$$

Trade-level positive parts are not the exposure of a legally netted set.

---

# 20. Multi-currency netting

For trade currency $c_j$:

$$
V_{j,t}^{(N)}
=
\operatorname{FXConvert}_{c_j\rightarrow N}^{closeout}
(
V_{j,t}^{(c_j)}
).
$$

The canonical ordering is:

$$
\boxed{
\text{valuation}
\rightarrow
\text{contractual FX conversion}
\rightarrow
\text{legal netting}.
}
$$

The closeout currency and FX convention must be contractual/model inputs.

Collateral FX conversion occurs after legal netting but before collateral enters the closeout equation, as specified in Section 16.

---

# 21. Expected exposure

Unconditional expected exposure:

$$
\boxed{
EE(t)
=
E^{Q^N}[E_t].
}
$$

Survival-conditioned expected exposure:

$$
\boxed{
EE^{surv}(t)
=
E^{Q^N}[E_t\mid\tau_C>t].
}
$$

Under generation one:

$$
\boxed{
EE^{surv}(t)=EE(t).
}
$$

This equality shall be **positively tested** on controlled independence cases and **negatively tested** under deliberate exposure/default dependence.

---

# 22. Expected exposure versus PFE

Expected exposure:

$$
EE(t)=E[V_t^+].
$$

PFE:

$$
PFE_q(t)=Q_q(V_t^+).
$$

These are distinct mathematical objects.

---

# 23. Survival and default probability

Define:

$$
\boxed{
S_C(t)=Q^N(\tau_C>t),
}
$$

and:

$$
\boxed{
PD_C(t)=1-S_C(t).
}
$$

For bucket $(t_{i-1},t_i]$:

$$
\boxed{
\Delta PD_i
=
Q^N(t_{i-1}<\tau_C\le t_i)
=
S_C(t_{i-1})-S_C(t_i).
}
$$

Generation-one invariants:

$$
S_C(0)=1,
$$

$$
0\le S_C(t)\le1,
$$

$$
S_C(t_i)\le S_C(t_{i-1}),
$$

$$
\Delta PD_i\ge0.
$$

---

# 24. Default timing and exposure endpoint convention

The integration grid is:

$$
0=t_0<t_1<\cdots<t_N=T.
$$

Bucket $i$ is:

$$
\boxed{
(t_{i-1},t_i].
}
$$

Therefore default exactly at $t_i$ belongs to bucket $i$.

At a contractual valuation discontinuity occurring at $t_i$, generation one uses the clean value immediately before the event:

$$
\boxed{
V_i^{closeout}=V(t_i^-).
}
$$

This applies to:

* cashflow;
* exercise;
* settlement;
* fixing;
* maturity;
* termination.

This is the canonical generation-one exposure endpoint convention.

---

# 25. Discount-factor endpoint convention

The generation-one discount factor used for bucket $i$ is:

$$
\boxed{
DF_i=DF(t_i^-)
}
$$

when the discount-factor representation has a declared discontinuity at $t_i$.

If the discount factor is continuous:

$$
DF(t_i^-)
=
DF(t_i)
=
DF(t_i^+).
$$

Therefore at a simultaneous event/discontinuity:

$$
\boxed{
EE_i
\text{ uses the pre-event exposure state},
\qquad
DF_i
\text{ uses the pre-event discount factor}.
}
$$

The discount layer owns detection and representation of declared discount discontinuities.

The CVA kernel receives only the resulting $DF_i$.

---

# 26. Survival jumps and bucket boundaries

The bucket default mass is:

$$
\boxed{
\Delta PD_i
=
Q^N(t_{i-1}<\tau_C\le t_i).
}
$$

If:

$$
p_i=Q^N(\tau_C=t_i)>0,
$$

then $p_i$ belongs to bucket $i$.

If survival is supplied on the grid:

$$
\Delta PD_i
=
S_C(t_{i-1})-S_C(t_i)
$$

includes the atom at $t_i$.

The associated atomic exposure uses:

$$
\boxed{
E_i^{atom}
=
\left(
V(t_i^-)-M(t_i^-)
\right)^+.
}
$$

---

# 27. Credit interpolation and domain

The credit layer owns:

* interpolation;
* native-to-internal time conversion;
* survival evaluation;
* default-mass construction;
* hazard integration where relevant;
* domain validation.

The CVA kernel receives aligned quantities.

Interpolation is mathematically material.

Outside the supported domain:

$$
\boxed{
\text{FAIL FAST}.
}
$$

No silent extrapolation.

---

# 28. Deterministic hazard-rate contract

The deterministic first-generation hazard is:

$$
\boxed{
\lambda_C^{det}(t).
}
$$

For an absolutely continuous survival law satisfying:

* $S_C(0)=1$;
* $S_C(t)>0$;
* $S_C$ absolutely continuous;
* $S_C$ non-increasing;
* $-\frac{S_C'(t)}{S_C(t)}$ non-negative and integrable;

define:

$$
\boxed{
\lambda_C^{det}(t)
=
-\frac{d}{dt}\log S_C(t).
}
$$

Then:

$$
\boxed{
S_C(t)
=
\exp
\left(
-\int_0^t
\lambda_C^{det}(u)\,du
\right),
}
$$

and:

$$
\boxed{
dPD_C(t)
=
\lambda_C^{det}(t)S_C(t)\,dt.
}
$$

A negative deterministic hazard is invalid.

If:

$$
S_C(t)=0,
$$

the finite ordinary hazard representation is not defined at and beyond that point.

The default-probability measure remains valid.

---

# 29. Default atoms and general default measure

The default-probability measure is more general than an ordinary hazard density.

Under generation-one independence:

$$
\boxed{
CVA
=
\int_{(0,T]}
DF(t)
LGD(t)
EE(t)\,
dPD_C(t).
}
$$

If:

$$
dPD_C
=
dPD_C^{cont}
+
\sum_kp_k\delta_{t_k},
$$

then:

$$
\boxed{
CVA
=
\int
DF(t)LGD(t)EE(t)\,
dPD_C^{cont}(t)
+
\sum_k
DF(t_k^-)
LGD(t_k^-)
EE_{t_k}^{atom}
p_k.
}
$$

The side convention is applied consistently to discounting, recovery, and exposure at the event.

For generation one, $LGD$ is deterministic and constant, so the side marker has no numerical effect but preserves the general event-order semantics.

---

# 30. Default compensator and deterministic default measure

Let:

$$
\Lambda_t
$$

denote the default compensator under the selected pre-default reference filtration.

The general stochastic-credit representation is:

$$
\boxed{
CVA
=
E^{Q^N}
\left[
\int_{(0,T]}
D_t\mathcal L_t\,d\Lambda_t
\right].
}
$$

The compensator satisfies:

$$
\boxed{
E^{Q^N}[\Lambda_t]
=
PD_C(t),
}
$$

and therefore, at the measure level:

$$
\boxed{
E^{Q^N}[d\Lambda_t]
=
dPD_C(t).
}
$$

Under generation one:

$$
\boxed{
d\Lambda_t
=
dPD_C(t)
=
\lambda_C^{det}(t)S_C(t)\,dt.
}
$$

This is the precise bridge between:

* the stochastic default compensator;
* the unconditional default-probability measure;
* the deterministic hazard representation.

They are related objects, not aliases.

---

# 31. Discounting contract

For generation one:

$$
\boxed{
DF_i=DF(t_i^-)>0
}
$$

under the endpoint convention of Section 25.

$DF(t)$ denotes the deterministic pricing discount factor associated with the selected:

* valuation currency;
* valuation numeraire;
* discounting/collateral convention.

The discount layer owns:

* curve interpolation;
* market conventions;
* time conversion;
* discontinuity metadata;
* domain validation.

Outside the supported domain:

$$
\boxed{
\text{FAIL FAST}.
}
$$

For stochastic discounting:

$$
\boxed{
CVA
=
E^{Q^N}
\left[
\int_{(0,T]}
D_t\mathcal L_t\,d\Lambda_t
\right].
}
$$

In general:

$$
E[D_tE_t]
\ne
E[D_t]E[E_t].
$$

---

# 32. Internal time coordinate

RiskQuant shall use one explicit internal year-fraction coordinate:

$$
\boxed{
t=\operatorname{YearFraction}(t_0,\text{date}).
}
$$

The exact convention shall be established by the repository's canonical date/time subsystem.

External curves may use different conventions.

Relevant layers convert native conventions into the internal coordinate.

The CVA kernel shall never infer day-count semantics from raw dates.

---

# 33. Grid ownership

The CVA engine has one explicit integration grid:

$$
0=t_0<t_1<\cdots<t_N=T.
$$

The exposure layer returns:

$$
EE(t_i),
$$

the credit layer returns:

$$
S_C(t_i),\Delta PD_i,
$$

and the discount layer returns:

$$
DF_i.
$$

The CVA integrator receives aligned arrays.

It shall not perform:

* date matching;
* nearest-date selection;
* interpolation;
* extrapolation;
* event-order interpretation;
* FX conversion.

---

# 34. Horizon

$T$ is:

> the latest future date at which relevant closeout exposure can be non-zero under the selected contractual and valuation model.

The valuation layer owns determination of this horizon.

The CVA engine consumes it.

---

# 35. Optionality and nonlinear valuation

For general derivatives:

$$
V_{s,i}^{clean}
=
V(X_{s,i},t_i).
$$

The valuation may involve:

* nonlinear payoffs;
* exercise;
* path dependence;
* nested valuation;
* regression;
* proxy valuation.

The CVA layer shall not assume linearity of future clean value.

---

# 36. Monte Carlo scenario contract

For scenario $s$ and time $t_i$:

$$
V_{s,i}^{clean}
=
V^{clean}(X_{s,i},t_i).
$$

After contractual netting, FX conversion, and closeout:

$$
\boxed{
E_{s,i}
=
\left(
V_{\mathcal N,s,i}^{clean}
-
M_{\mathcal N,s,i}
\right)^+.
}
$$

Generation one:

$$
M_{\mathcal N,s,i}=0.
$$

Semantic shape:

$$
\boxed{
E\in\mathbb R^{S\times N}
}
$$

with:

* axis 0 = scenario;
* axis 1 = time.

---

# 37. Monte Carlo measure and weighting

For ordinary generation-one IID Monte Carlo:

$$
X_s\sim Q^N,
$$

and:

$$
w_s=\frac1S.
$$

Therefore:

$$
\boxed{
\widehat{EE}_i
=
\frac1S
\sum_sE_{s,i}
}
$$

is unbiased for:

$$
EE(t_i).
$$

For general weighted estimators:

$$
\widehat{EE}_i
=
\sum_sw_sE_{s,i},
$$

validity requires the scenario law and weights jointly to target the desired pricing expectation.

For importance sampling:

$$
w_s
\propto
\frac{dQ^N}{dQ^{IS}}(X_s)
$$

may provide the required likelihood correction.

Self-normalised importance weights are a distinct estimator and are not generally unbiased.

Generation one may restrict itself to equal-weight IID pricing-measure Monte Carlo.

---

# 38. Dependent scenarios and sampling error

For:

$$
\widehat{EE}_i
=
\sum_sw_sE_{s,i},
$$

the variance is:

$$
\boxed{
\operatorname{Var}(\widehat{EE}_i)
=
\sum_{s=1}^{S}
\sum_{r=1}^{S}
w_sw_r
\operatorname{Cov}(E_{s,i},E_{r,i}).
}
$$

For IID equal-weight scenarios:

$$
\boxed{
\operatorname{Var}(\widehat{EE}_i)
=
\frac{\operatorname{Var}(E_i)}{S}.
}
$$

Therefore:

$$
\boxed{
SE(\widehat{EE}_i)
=
\sqrt{
\frac{
\widehat{\operatorname{Var}}(E_i)
}{S}
}.
}
$$

---

# 39. Error decomposition

The implementation shall distinguish:

1. valuation-model error;
2. Monte Carlo sampling error;
3. time-discretisation error;
4. credit interpolation/calibration error;
5. discount interpolation/model error;
6. proxy/regression error.

These shall not be hidden inside one unexplained tolerance.

---

# 40. First-generation deterministic reference calculation

Given:

* dates $0=t_0<\cdots<t_N=T$;
* clean future valuations;
* deterministic $DF_i$;
* deterministic recovery;
* deterministic survival;
* equal-weight pricing-measure scenarios;

calculate:

### Step 1 — valuation and netting

$$
V_{\mathcal N,s,i}^{clean}
=
\sum_{j\in\mathcal N}
V_{j,s,i}^{clean}.
$$

### Step 2 — exposure

$$
E_{s,i}
=
\max
\left(
V_{\mathcal N,s,i}^{clean},
0
\right).
$$

At a declared event discontinuity, use the pre-event state.

### Step 3 — expected exposure

$$
EE_i
=
\frac1S
\sum_sE_{s,i}.
$$

### Step 4 — default mass

$$
\Delta PD_i
=
S_{i-1}-S_i.
$$

### Step 5 — LGD

$$
LGD=1-R_C.
$$

### Step 6 — bucket CVA

$$
\boxed{
CVA_i
=
LGD\,DF_i\,EE_i\,\Delta PD_i.
}
$$

### Step 7 — aggregate

$$
\boxed{
CVA
=
\sum_iCVA_i.
}
$$

This is the canonical first-generation golden oracle.

---

# 41. CVA integration kernel

The generation-one kernel is:

$$
\boxed{
CVA_i
=
LGD_iDF_iEE_i\Delta PD_i.
}
$$

Inputs are already:

* finite;
* validated;
* aligned;
* expressed under declared conventions.

Outputs:

$$
(CVA_i)_{i=1}^{N},
$$

and optionally:

$$
CVA_{\le i}
=
\sum_{j\le i}CVA_j,
$$

and:

$$
CVA=\sum_iCVA_i.
$$

The kernel performs no semantic work.

---

# 42. Output profile

The engine shall expose:

$$
\boxed{
\{t_i,EE_i,S_i,\Delta PD_i,DF_i,LGD_i,CVA_i\}_{i=1}^{N}
}
$$

plus:

$$
CVA_{\le i}
$$

and aggregate:

$$
CVA.
$$

`discount_factor` shall carry the declared endpoint semantics from Section 25.

---

# 43. Numerical tolerance policy

Mathematical identities are distinct from finite-precision equality.

For:

$$
CVA=\sum_iCVA_i,
$$

the implementation may verify:

$$
\boxed{
|CVA-\sum_iCVA_i|
\le
\epsilon_{abs}
+
\epsilon_{rel}|CVA|.
}
$$

Likewise:

$$
\sum_i\Delta PD_i=1-S_N
$$

is verified within declared tolerance.

Tolerance shall never repair invalid quantities.

---

# 44. Mathematical invariants

Generation one shall enforce or test:

$$
EE_i\ge0,
$$

$$
S_0=1,
$$

$$
0\le S_i\le1,
$$

$$
S_i\le S_{i-1},
$$

$$
\Delta PD_i\ge0,
$$

$$
0\le R_C\le1,
$$

$$
0\le LGD_i\le1,
$$

$$
DF_i>0,
$$

$$
t_i>t_{i-1},
$$

and:

$$
CVA_i\ge0,
\qquad
CVA\ge0
$$

within declared finite-precision tolerance.

Behavioural sign tests shall verify:

$$
V>0\Rightarrow E>0,
$$

and:

$$
V<0\Rightarrow E=0
$$

in generation one.

---

# 45. Monotonicity

Holding all other quantities fixed:

If:

$$
EE_i^{(A)}\le EE_i^{(B)}
\quad\forall i,
$$

then:

$$
CVA^{(A)}\le CVA^{(B)}.
$$

If:

$$
LGD_A\le LGD_B,
$$

then:

$$
CVA_A\le CVA_B.
$$

If:

$$
\Delta PD_i^{(A)}
\le
\Delta PD_i^{(B)}
\quad\forall i,
$$

then:

$$
CVA^{(A)}
\le
CVA^{(B)}.
$$

---

# 46. Golden and edge cases

## Case A — zero exposure

$$
V_t^{clean}\le0\quad\forall t
\Rightarrow
CVA=0.
$$

## Case B — zero cumulative default probability

$$
PD_C(T)=0
\Rightarrow
CVA=0.
$$

## Case C — zero LGD

$$
R_C=1
\Rightarrow
CVA=0.
$$

## Case D — constant exposure and unit discounting

$$
EE(t)=E_0,\quad DF(t)=1.
$$

Then:

$$
CVA
=
LGD\,E_0
\sum_i\Delta PD_i
=
\boxed{
LGD\,E_0\,PD_C(T).
}
$$

$PD_C(T)$ is cumulative default probability.

## Case E — certain default by maturity

$$
PD_C(T)=1
$$

gives:

$$
\sum_i\Delta PD_i=1.
$$

For constant exposure and unit discounting:

$$
\boxed{
CVA=LGD\,E_0.
}
$$

If:

$$
S_C(T)=0,
$$

the final default mass is handled through the default-probability measure representation. A finite hazard at $T$ is not required.

## Case F — netting

$$
V_A=100,\quad V_B=-90
$$

gives:

$$
E=(100-90)^+=10.
$$

For:

$$
V_A=100,\quad V_B=-100,
$$

gives:

$$
E=0.
$$

## Case G — deterministic oracle

Compare production against:

$$
\boxed{
\sum_iLGD_iDF_iEE_i\Delta PD_i.
}
$$

## Case H — collateral sign

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

## Case I — zero scenarios

$$
S=0
\Rightarrow
\boxed{\text{FAIL FAST}}.
$$

## Case J — empty netting set

$$
\mathcal N=\varnothing
\Rightarrow
V_{\mathcal N}=0
\Rightarrow
E=0
\Rightarrow
\boxed{CVA=0}.
$$

## Case K — zero horizon

If:

$$
T=0
$$

and generation one excludes an atom of default at time zero:

$$
\boxed{
CVA=0.
}
$$

## Case L — independence equality

Construct scenarios and credit independently under generation-one assumptions and verify:

$$
\boxed{
EE^{surv}(t_i)=EE(t_i)
}
$$

within the declared statistical/numerical tolerance.

## Case M — deliberate WWR dependence

Construct a controlled dependence model and verify:

$$
\boxed{
EE^{surv}(t_i)\ne EE(t_i)
}
$$

for at least one date.

---

# 47. Additivity and cross-default

CVA is additive across valuation units only when the legal and closeout model provides an additive loss decomposition.

If:

$$
\mathcal L_{portfolio}
=
\sum_k\mathcal L_k,
$$

then:

$$
\boxed{
CVA_{portfolio}
=
\sum_kCVA_k.
}
$$

No statistical independence between the units is required.

Cross-default clauses do not by themselves determine additivity.

The decisive question is:

$$
\boxed{
\text{Does the legal/closeout model define one common loss scope or separate additive losses?}
}
$$

If cross-default provisions create one common legal closeout/netting scope, the relevant trades constitute one mathematical valuation unit.

If genuinely separate closeout units remain and:

$$
\mathcal L=\sum_k\mathcal L_k,
$$

additivity follows.

The legal-data layer owns the evidence for this classification, including where applicable:

* enforceable netting provisions;
* common closeout scope;
* common collateral pool;
* cross-default triggers;
* contractual set-off;
* applicable legal opinions.

The numerical CVA kernel does not determine legal enforceability.

---

# 48. Grid refinement and discretisation error

For smooth generation-one integrands:

$$
f(t)
=
DF(t)
LGD(t)
EE(t)
\lambda_C^{det}(t)
S_C(t),
$$

a right-endpoint scheme has first-order global behaviour under appropriate regularity:

$$
O(\Delta t).
$$

For differentiable discounting on bucket $i$:

$$
|DF(t_i^-)-DF(\tau)|
\le
K_i\Delta t_i
$$

for:

$$
\tau\in(t_{i-1},t_i].
$$

This is a smooth-bucket bound only.

If a discount factor or exposure has a genuine jump at:

$$
t^\star,
$$

the local error is driven by the jump magnitude and the default mass around the event.

The smooth $O(\Delta t)$ claim shall not be applied across an unresolved discontinuity.

---

# 49. WWR and stochastic credit contract

For WWR, define the **pre-default reference filtration**:

$$
\boxed{
\mathcal F_t^{ref}
=
\text{market/credit information available before default is revealed}.
}
$$

This notation is deliberately not identified with the standard default-indicator filtration:

$$
\sigma(\tau_C\wedge t),
$$

which reveals whether default has occurred.

The reference filtration must not contain:

$$
\mathbf1_{\{\tau_C\le t\}}.
$$

Define conditional survival:

$$
\boxed{
G_t
=
Q^N(
\tau_C>t
\mid
\mathcal F_t^{ref}
).
}
$$

Under generation one:

$$
\boxed{
\mathcal F_t^{ref}
=
\mathcal F_t^M
}
$$

and:

$$
\boxed{
G_t=S_C(t).
}
$$

A strictly larger reference filtration represents a stochastic-credit/WWR extension unless deterministic conditional survival remains explicitly proven.

---

# 50. Stochastic intensity and Cox construction

The stochastic pre-default intensity is:

$$
\boxed{
\lambda_t.
}
$$

It is an $\mathcal F_t^{ref}$-predictable non-negative process.

This is distinct from the deterministic generation-one hazard:

$$
\lambda_C^{det}(t).
$$

Under a Cox construction, let:

$$
\mathcal E\sim\operatorname{Exp}(1)
$$

and require:

$$
\boxed{
\mathcal E
\perp
\mathcal F_\infty^{ref},
}
$$

where:

$$
\mathcal F_\infty^{ref}
=
\bigvee_{t\ge0}
\mathcal F_t^{ref}.
$$

Define:

$$
\boxed{
\tau_C
=
\inf
\left\{
t\ge0:
\int_0^t\lambda_u\,du
\ge
\mathcal E
\right\}.
}
$$

Then:

$$
\begin{aligned}
G_t
&=
Q^N
\left(
\tau_C>t
\mid
\mathcal F_t^{ref}
\right)
\\
&=
Q^N
\left(
\mathcal E>
\int_0^t\lambda_u\,du
\mid
\mathcal F_t^{ref}
\right)
\\
&=
\boxed{
\exp
\left(
-\int_0^t\lambda_u\,du
\right).
}
\end{aligned}
$$

The corresponding continuous pre-default default compensator is:

$$
\boxed{
d\Lambda_t
=
\lambda_tG_t\,dt.
}
$$

---

# 51. WWR CVA representation

The general stochastic-credit CVA formulation is:

$$
\boxed{
CVA_0
=
E^{Q^N}
\left[
\int_{(0,T]}
D_t
\mathcal L_t
\,d\Lambda_t
\right].
}
$$

Under an absolutely continuous pre-default intensity:

$$
d\Lambda_t
=
\lambda_tG_tdt,
$$

giving:

$$
\boxed{
CVA_0
=
E^{Q^N}
\left[
\int_0^T
D_t
\mathcal L_t
\lambda_t
G_t\,dt
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
=
E^{Q^N}
\left[
\int_0^T
D_t
LGD_t
E_t
\lambda_t
G_t\,dt
\right].
}
$$

There is no second multiplication by $E_t$.

---

# 52. Generation-one bridge to the WWR formulation

Under generation one:

$$
\lambda_t
\rightarrow
\lambda_C^{det}(t),
$$

$$
G_t
\rightarrow
S_C(t),
$$

and:

$$
d\Lambda_t
\rightarrow
\lambda_C^{det}(t)S_C(t)\,dt.
$$

Since:

$$
G_t
=
Q^N(\tau_C>t\mid\mathcal F_t^{ref}),
$$

the unconditional survival probability is:

$$
S_C(t)=E^{Q^N}[G_t].
$$

Therefore, where differentiation and expectation may be interchanged under the required integrability conditions:

$$
\begin{aligned}
\frac{d}{dt}PD_C(t)
&=
-\frac{d}{dt}S_C(t)
\\
&=
-\frac{d}{dt}E^{Q^N}[G_t]
\\
&=
E^{Q^N}
[
\lambda_tG_t
].
\end{aligned}
$$

Hence:

$$
\boxed{
E^{Q^N}[d\Lambda_t]
=
dPD_C(t).
}
$$

Under generation one, the quantities are deterministic:

$$
d\Lambda_t
=
dPD_C(t)
=
\lambda_C^{det}(t)S_C(t)\,dt.
$$

Furthermore, because:

* $DF(t)$ is deterministic;
* $LGD(t)$ is deterministic;
* $E_t$ is independent of the default mechanism;

we obtain:

$$
\boxed{
E^{Q^N}
\left[
\int_{(0,T]}
D_tLGD_tE_t\,d\Lambda_t
\right]
=
\int_{(0,T]}
DF(t)LGD(t)EE(t)\,dPD_C(t).
}
$$

This is the precise mathematical bridge from the general stochastic-credit formulation to the generation-one deterministic CVA formula.

---

# 53. WWR with default atoms

If default has atoms, the ordinary:

$$
\lambda_tG_tdt
$$

representation is insufficient.

The compensator may contain continuous and atomic components:

$$
d\Lambda_t
=
d\Lambda_t^{cont}
+
\sum_k
\Delta\Lambda_{t_k}
\delta_{t_k}(dt).
$$

Then:

$$
\boxed{
CVA
=
E^{Q^N}
\left[
\int
D_t\mathcal L_t\,d\Lambda_t^{cont}
\right]
+
\sum_k
E^{Q^N}
\left[
D(t_k^-)
\mathcal L_{t_k}^{atom}
\Delta\Lambda_{t_k}
\right].
}
$$

The atomic loss uses the declared pre-event convention.

---

# 54. WWR and collateral seam

A future WWR/collateral model may require:

$$
\boxed{
(
\text{market state},
\text{credit state},
\text{collateral state},
\text{agreement},
\text{event state}
)
\rightarrow
U_t
\rightarrow
E_t
\rightarrow
\mathcal L_t.
}
$$

Collateral can itself depend on market state and therefore participate in exposure-credit dependence.

Generation one avoids this complexity through:

$$
M_t=0.
$$

---

# 55. WWR joint-law models and copulas

A valid WWR model must define an explicit joint law.

A Gaussian copula is admissible if it specifies, under $Q^N$:

* exposure-state marginal law;
* credit/default marginal law;
* copula structure;
* dependence parameter;
* mapping from latent variables to default time/credit state;
* simulation/estimation procedure.

For example:

$$
\begin{pmatrix}
Z_E\\
Z_C
\end{pmatrix}
\sim
N
\left(
0,
\begin{pmatrix}
1&\rho\\
\rho&1
\end{pmatrix}
\right),
$$

with:

$$
Z_E,Z_C
\sim N(0,1),
$$

and default mapping:

$$
\tau_C
=
F_C^{-1}
\left(
\Phi(Z_C)
\right),
$$

where $F_C$ is the specified marginal CDF of $\tau_C$ under $Q^N$.

The exact construction is model-specific.

The invariant is:

$$
\boxed{
\text{WWR requires an explicit joint law, not a scalar correlation multiplier}.
}
$$

---

# 56. Bilateral/DVA boundary

The first implementation is unilateral:

$$
\boxed{
\text{counterparty default is modelled; our default is not}.
}
$$

A bilateral extension requires:

* counterparty default;
* own default;
* first-to-default ordering;
* bilateral closeout;
* own recovery;
* counterparty recovery;
* collateral-return treatment.

The approximation:

$$
V_{bilateral}
\approx
V_{clean}-CVA+DVA
$$

is not universal.

DVA is not a Boolean attached to unilateral CVA.

---

# 57. XVA boundary

"XVA" denotes a family of valuation adjustments.

Examples include:

* CVA;
* DVA;
* FVA;
* MVA;
* KVA.

Their interactions depend on the valuation framework and modelling assumptions.

Therefore:

$$
XVA
=
CVA+DVA+FVA+MVA+KVA+\cdots
$$

is not a universal identity.

RiskQuant shall not build a generic additive XVA interface before unilateral CVA is independently correct.

---

# 58. Architectural mathematical seams

## Seam A — future-state generation

$$
\boxed{
\text{market inputs}
\rightarrow
X_{s,i}
}
$$

## Seam B — clean future valuation

$$
\boxed{
(X_{s,i},t_i)
\rightarrow
V_{s,i}^{clean}
}
$$

This layer owns event-aware clean valuation.

## Seam C — contractual/netting aggregation

$$
\boxed{
\{V_{j,s,i}^{clean}\}_{j\in\mathcal N}
\rightarrow
V_{\mathcal N,s,i}^{clean}
}
$$

The legal-data layer establishes membership and closeout scope.

## Seam D — closeout/collateral exposure

$$
\boxed{
(
V_{\mathcal N,s,i}^{clean},
M_{s,i},
\text{agreement},
\text{event state},
\text{FX convention}
)
\rightarrow
U_{s,i}
\rightarrow
E_{s,i}.
}
$$

## Seam E — exposure aggregation

$$
\boxed{
\{E_{s,i}\}_s
\rightarrow
EE_i
}
$$

This layer owns scenario weighting and estimator semantics.

## Seam F — credit model

Generation one:

$$
\boxed{
\text{credit inputs}
\rightarrow
S_C(t_i),\Delta PD_i.
}
$$

Optional deterministic hazard:

$$
\boxed{
S_C
\rightarrow
\lambda_C^{det}.
}
$$

Future stochastic credit/WWR:

$$
\boxed{
(\text{market/credit state})
\rightarrow
\lambda_t,G_t,d\Lambda_t.
}
$$

## Seam G — discounting

Generation one:

$$
\boxed{
\text{discount inputs}
\rightarrow
DF_i.
}
$$

Future stochastic discounting:

$$
\boxed{
\text{rates/collateral state}
\rightarrow
D_t.
}
$$

## Seam H — CVA integration

Generation one:

$$
\boxed{
(EE_i,DF_i,LGD_i,\Delta PD_i)
\rightarrow
CVA_i
\rightarrow
CVA.
}
$$

General stochastic-credit formulation:

$$
\boxed{
(\mathcal L_t,D_t,d\Lambda_t)
\rightarrow
CVA.
}
$$

---

# 59. Kernel doctrine

The generation-one kernel performs:

$$
\boxed{
CVA_i
=
LGD_iDF_iEE_i\Delta PD_i.
}
$$

It shall not perform:

* legal netting discovery;
* legal enforceability assessment;
* cross-default classification;
* collateral lookup;
* credit calibration;
* CDS convention selection;
* measure selection;
* numeraire selection;
* FX conversion;
* date alignment;
* interpolation;
* extrapolation;
* event-order interpretation;
* discontinuity detection;
* market-data retrieval;
* business-rule interpretation;
* object traversal;
* string dispatch;
* hidden sign correction.

---

# 60. Fail-fast contract

Invalid mathematical inputs shall fail before entering numerical kernels.

Examples:

* recovery outside $[0,1]$;
* LGD outside $[0,1]$;
* non-positive discount factors;
* survival outside $[0,1]$;
* increasing survival;
* negative default mass;
* negative deterministic hazard;
* malformed scenario weights;
* zero scenarios;
* NaN;
* infinity;
* ambiguous scenario/time dimensions;
* duplicate or non-monotone dates;
* unsupported internal time basis;
* dates outside credit/discount domains;
* physical-measure probabilities supplied to a pricing contract;
* inconsistent numeraire metadata;
* missing future valuation;
* credit-adjusted clean values;
* unsupported default atoms passed to a generation-one continuous-hazard-only path;
* inconsistent event-order metadata;
* undefined discount discontinuity side;
* undefined collateral FX convention.

No silent:

* clipping;
* projection;
* extrapolation;
* scenario dropping;
* reweighting;
* sign correction.

---

# 61. No silent convention changes

The following are externally inspectable:

* valuation perspective;
* pricing measure;
* numeraire;
* valuation currency;
* closeout currency;
* legal netting scope;
* cross-default/closeout scope;
* collateral sign;
* collateral currency;
* trade FX convention;
* collateral FX convention;
* closeout;
* recovery;
* survival representation;
* default-measure representation;
* deterministic hazard;
* stochastic intensity;
* pre-default reference filtration;
* interpolation;
* extrapolation policy;
* exposure grid;
* credit grid;
* discount grid;
* internal time coordinate;
* day-count conversion;
* event ordering;
* exposure endpoint side;
* discount-factor endpoint side;
* default bucket;
* scenario law;
* scenario weighting;
* scenario/time axis semantics;
* numerical tolerance.

Changing any of these may change the mathematical quantity being priced.

---

# 62. No double-counting contract

At generation one:

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

At the general level:

$$
\mathcal L_t
$$

already contains the actual default loss.

Therefore:

$$
D_t\mathcal L_t\,d\Lambda_t
$$

is correct.

The incorrect expression:

$$
D_t\mathcal L_tE_t\,d\Lambda_t
$$

double-counts exposure.

Under simple recovery:

$$
\mathcal L_t=LGD_tE_t.
$$

---

# 63. Anti-patterns

The following are prohibited:

1. Scalar CVA shortcut with no time profile.
2. Today's exposure multiplied by cumulative PD.
3. Historical P&L substituted for future pricing exposure.
4. Trade-level positive parts used for legally netted exposure.
5. Counterparty ID used as the legal netting definition.
6. CDS spread treated as hazard by identity.
7. Arbitrary correlation multiplier used as WWR.
8. Universal additive XVA.
9. Hidden default bucket convention.
10. Hidden stochastic-discounting factorisation.
11. Credit-contaminated clean valuation.
12. Semantics-free collateral subtraction.
13. Default-revealing filtration used for conditional survival.
14. Unconditional EE used under WWR.
15. Default atoms forced into ordinary hazard density.
16. Silent credit/discount extrapolation.
17. Hidden interpolation inside CVA kernel.
18. Ambiguous event ordering at discontinuities.
19. Physical-measure weights used as pricing weights.
20. Silent scenario dropping.
21. Numerical tolerance used as repair.
22. Currency treated as numeraire.
23. Cross-default bookkeeping boundaries mistaken for additivity.
24. Stochastic intensity used without validating its compensator/filtration meaning.
25. Smooth convergence claims made across unresolved discontinuities.
26. IM/IA treated as scalar exposure reduction without legal closeout semantics.
27. Deterministic hazard and stochastic intensity conflated.
28. Deterministic default measure and stochastic compensator treated as aliases.
29. Finite hazard assumed after survival reaches zero.
30. Discount-factor discontinuity side left implicit.
31. Cox trigger claimed independent without specifying independence from the full reference filtration.
32. Stochastic default intensity used without an explicit pre-default reference filtration.
33. A copula parameter used without a complete joint-law specification.
34. Generation-one independence claimed under $\mathcal F_t^M$ while silently using a larger $\mathcal F_t^{ref}$ without proving deterministic conditional survival.
35. Collateral converted using reporting FX rather than contractual/modelled closeout FX.
36. Recovery, exposure, and discounting assigned inconsistent event sides at an atomic/default endpoint.

---

# 64. Validation hierarchy

Validation proceeds from mathematics outward.

## Level A — dimensions and scalar arithmetic

## Level B — measure, numeraire, and currency metadata

## Level C — survival/default invariants

## Level D — exposure transformation

## Level E — legal netting and cross-default decomposition

## Level F — collateral sign and FX semantics

## Level G — event-order and discontinuity semantics

## Level H — Monte Carlo measure, weighting, and sampling error

## Level I — deterministic CVA integration

## Level J — independent golden oracle

## Level K — temporal grid refinement

## Level L — credit/discount interpolation and domain tests

## Level M — deterministic hazard/default-measure consistency

## Level N — stochastic-credit/WWR validation

## Level O — bilateral/XVA extensions

Levels N and O are controlled extensions and are gated on successful completion of Levels A–M.

---

# 65. Grid-refinement validation

For nested grids:

$$
\Delta t_{\max}^{(1)}
>
\Delta t_{\max}^{(2)}
>
\cdots,
$$

generation-one CVA should converge consistently with the selected numerical scheme under the required regularity conditions.

The test shall detect:

* gross non-convergence;
* date-alignment errors;
* interpolation inconsistency;
* event-order errors;
* default-mass allocation errors;
* discontinuity artefacts.

A universal convergence rate shall not be claimed where:

* exposure jumps;
* discounting jumps;
* survival has atoms;
* the numerical scheme does not resolve an event boundary.

---

# 66. Independent oracle requirement

The production implementation shall be tested against an independently constructed oracle wherever practical.

The oracle shall not share:

* the same integration loop;
* the same interpolation helper;
* the same default-mass construction;
* the same scenario aggregation;
* the same kernel.

---

# 67. Monte Carlo convergence and sampling uncertainty

For IID equal-weight generation-one Monte Carlo:

$$
SE(\widehat{EE}_i)
=
\sqrt{
\frac{
\widehat{\operatorname{Var}}(E_i)
}{S}
}.
$$

Where practical, the implementation should estimate CVA sampling uncertainty from the scenario-level CVA contributions.

Sampling error shall be distinguished from:

* time-discretisation;
* curve interpolation;
* valuation error.

---

# 68. Code-archaeology contract

Before adding any abstraction, inspect existing RiskQuant primitives for:

1. future market-state generation;
2. future-dated valuation;
3. trade/portfolio valuation;
4. contractual/netting aggregation;
5. FX/currency conversion;
6. collateral;
7. discounting;
8. scenario measures/weights;
9. survival/default curves;
10. hazard/intensity representation;
11. exposure aggregation;
12. scenario × time storage;
13. deterministic integration;
14. independent oracle infrastructure;
15. invariant validation;
16. time-coordinate handling;
17. interpolation/extrapolation;
18. clean-versus-credit-adjusted valuation;
19. event-order handling;
20. discontinuity metadata;
21. reference/pre-default filtration semantics.

For each candidate abstraction ask:

* What mathematical object does it represent?
* What are its units?
* What pricing measure does it use?
* What numeraire does it use?
* What currency does it use?
* What filtration/information does it assume?
* What sign convention does it use?
* What time coordinate does it use?
* What legal/model scope does it represent?
* What interpolation/extrapolation rules does it encode?
* What event-order conventions does it encode?
* What invariants does it guarantee?
* Can it satisfy this CVA contract without semantic distortion?

A matching class name, field name, or shape is not evidence of compatibility.

---

# 69. Required archaeological challenge questions

Before proposing CVA architecture, answer explicitly:

## Future valuation

Can RiskQuant produce:

$$
V^{clean}(X_t,t)?
$$

## Pricing measure

Where is:

$$
Q^N
$$

established?

## Numeraire

What exactly is:

$$
N_t?
$$

How is it distinguished from valuation currency?

## Discounting

What exactly produces:

$$
DF(t)?
$$

What are its interpolation and discontinuity-side conventions?

## Legal netting

Where is enforceable membership established?

## Cross-default

Does the legal/closeout model imply a common valuation unit or additive loss units?

## Currency

Where is contractual FX conversion performed?

## Collateral

Where is collateral represented, including sign and currency?

## Credit

What exactly do existing objects representing:

$$
S,\quad PD,\quad\Delta PD,\quad\lambda
$$

mean?

Are deterministic hazard and stochastic intensity explicitly distinguished?

## Default compensator

Where is:

$$
d\Lambda_t
$$

represented?

## WWR filtration

What exactly is:

$$
\mathcal F_t^{ref}?
$$

Why does it not reveal default?

## Time

What is the canonical internal year-fraction coordinate?

## Event ordering

Can future valuation distinguish:

$$
V(t^-)
\quad\text{from}\quad
V(t^+)?
$$

Can discounting distinguish:

$$
DF(t^-)
\quad\text{from}\quad
DF(t^+)?
$$

## Atoms

Can the credit model represent default atoms without forcing them into a finite hazard?

## Kernel

Can the integration kernel operate purely on aligned validated arrays?

## Reuse

Can an existing abstraction satisfy this mathematical contract without changing its semantics?

If not, reuse is prohibited.

---

# 70. First-generation mathematical API contract

Semantically:

$$
\boxed{
(EE_i,DF_i,LGD_i,\Delta PD_i)_{i=1}^{N}
\rightarrow
(CVA_i)_{i=1}^{N}
\rightarrow
CVA.
}
$$

Exposure:

$$
\boxed{
V_{s,i}^{clean}
\rightarrow
E_{s,i}
\rightarrow
EE_i.
}
$$

Credit:

$$
\boxed{
\text{credit inputs}
\rightarrow
S_i,PD_i,\Delta PD_i.
}
$$

Optional deterministic hazard:

$$
\boxed{
S_C
\rightarrow
\lambda_C^{det}.
}
$$

Discount:

$$
\boxed{
\text{discount inputs}
\rightarrow
DF_i.
}
$$

Scenario measure/weighting:

$$
\boxed{
\text{scenario law}
+
\text{weights}
\rightarrow
\widehat{EE}.
}
$$

Each is a mathematical boundary.

---

# 71. Precision and numerical policy

No NaN, None, magic sentinel, silent clipping, scenario removal, silent reweighting, or hidden extrapolation may substitute for invalid or missing mathematical quantities inside numerical kernels.

Finite-precision tolerances may validate identities.

They shall not repair invalid inputs.

---

# 72. Controlled evolution path

The **mathematical development path** and **validation hierarchy** are intentionally aligned:

## Level 1 — Exposure-profile integration

$$
\{t_i,EE_i,DF_i,LGD_i,\Delta PD_i\}
\rightarrow
CVA.
$$

## Level 2 — Future-state simulation

$$
X_{s,i}
\rightarrow
V_{s,i}^{clean}
\rightarrow
E_{s,i}
\rightarrow
EE_i.
$$

## Level 3 — Contractual netting and collateral

$$
V_{\mathcal N,s,i}^{clean}
\rightarrow
U_{s,i}
\rightarrow
E_{s,i}.
$$

## Level 4 — Stochastic credit / WWR

Introduce:

$$
\mathcal F_t^{ref},
\quad
\lambda_t,
\quad
G_t,
\quad
d\Lambda_t.
$$

Use:

$$
\boxed{
CVA
=
E^{Q^N}
\left[
\int
D_t\mathcal L_t\,d\Lambda_t
\right].
}
$$

For an absolutely continuous Cox/intensity model:

$$
d\Lambda_t
=
\lambda_tG_tdt.
$$

## Level 5 — Bilateral credit

Introduce own default, first-to-default ordering, and bilateral closeout.

## Level 6 — Broader XVA

Only then consider FVA, MVA, KVA, and coupled XVA effects.

The validation hierarchy in Section 64 is the corresponding implementation/test gate.

---

# 73. Reference example

Suppose:

| Date | Expected Exposure | Incremental PD | Discount Factor |
| ---- | ----------------: | -------------: | --------------: |
| 1Y   |       €10,000,000 |          0.50% |            0.97 |
| 2Y   |        €8,000,000 |          0.75% |            0.94 |
| 3Y   |        €6,000,000 |          1.00% |            0.91 |

and:

$$
R_C=40\%,
\qquad
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
0.005
=
€29{,}100,
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
0.0075
=
€33{,}840,
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
0.01
=
€32{,}760.
$$

Therefore:

$$
\boxed{
CVA=€95{,}700.
}
$$

---

# 74. Absolute implementation doctrine

**The general CVA definition is expected discounted counterparty-default loss under an explicitly selected pricing measure.**

**CVA is a pricing quantity, not inherently a VaR/PFE/ES-style risk statistic.**

**A numeraire is a value process, not a currency label.**

**The pricing measure, numeraire, valuation currency, and discount factor are distinct but consistent concepts.**

**The first implementation is unilateral CVA, not general XVA.**

**Clean future valuation contains no counterparty-credit adjustment.**

**Netting occurs before positive-part truncation.**

**Collateral semantics are explicit; initial margin is not automatically subtractable.**

**\(M>0\) means collateral held by us; \(M<0\) means collateral posted by us.**

**Physical and pricing probabilities are never silently interchanged.**

**Unconditional EE and survival-conditioned EE are distinct.**

**Their equality is a first-generation independence consequence.**

**The generation-one reference filtration is the exposure-relevant market filtration.**

**A larger reference filtration belongs to a stochastic-credit/WWR extension unless deterministic conditional survival remains proven.**

**The pre-default reference filtration must not reveal realised default.**

**The generation-one hazard is a deterministic function; a WWR intensity is a stochastic predictable process.**

**The deterministic default-probability measure and stochastic default compensator are distinct objects.**

**The standard \(DF\times EE\times\Delta PD\) formula is a factorised specialisation under explicit assumptions.**

**The bucketed sum is a numerical approximation.**

**Default exactly at \(t_i\) belongs to \((t_{i-1},t_i]\).**

**Generation one uses pre-event exposure and pre-event discounting at declared discontinuities.**

**Collateral FX conversion uses the contractual/modelled closeout convention before collateral enters the closeout equation.**

**Survival jumps and default atoms are represented through the default-probability measure or compensator, not forced into an ordinary finite hazard.**

**Credit and discount interpolation belong to their respective layers.**

**Silent extrapolation is prohibited.**

**Monte Carlo measure and weighting jointly determine estimator semantics.**

**Scenario dependence affects sampling error through cross-scenario covariance.**

**WWR is an explicit joint exposure-credit model, not a correlation multiplier.**

**CVA/DVA/FVA/MVA/KVA are not assumed universally additive.**

**Additivity is determined by the legal/closeout loss decomposition, including cross-default scope.**

**Default loss and exposure are distinct objects.**

**Do not double-count default or exposure.**

**Do not infer semantics from names, shapes, or architectural convenience.**

**Kernels perform declared mathematics.**

**Boundaries establish semantics and invariants.**

**Fail fast when the mathematical contract is violated.**

---

# 75. Completion criterion for the first CVA engine

The first RiskQuant CVA implementation is complete only when it can demonstrate:

1. exact mathematical identity of the calculated quantity;
2. explicit first-generation assumptions;
3. explicit pricing measure;
4. explicit numeraire;
5. explicit valuation/closeout currency;
6. explicit discounting convention;
7. explicit discount discontinuity/event-order convention;
8. clean future valuation separated from historical P&L and credit-adjusted valuation;
9. legal netting scope established before exposure truncation;
10. explicit trade FX conversion;
11. explicit collateral sign and collateral FX conversion;
12. explicit closeout and recovery model;
13. VM/IM/IA distinction;
14. mathematically defined survival/default quantities;
15. explicit credit interpolation and domain rules;
16. explicit deterministic hazard conditions;
17. explicit default-measure treatment of atoms;
18. explicit default bucket convention;
19. explicit event ordering at discontinuities;
20. explicit internal time coordinate;
21. explicit scenario measure and weighting;
22. independent golden-oracle agreement;
23. profile-to-scalar reconciliation within declared tolerance;
24. invariant enforcement;
25. behavioural netting and collateral-sign tests;
26. positive independence test showing $EE^{surv}=EE$;
27. negative dependence test showing $EE^{surv}\ne EE$ under deliberate WWR;
28. grid-refinement convergence;
29. Monte Carlo convergence and sampling-error reporting;
30. separation of Monte Carlo, discretisation, curve, and valuation errors;
31. mathematically valid pre-default reference filtration for WWR extensions;
32. explicit distinction between stochastic intensity and deterministic hazard;
33. explicit distinction between default compensator and unconditional default measure;
34. Cox construction validated with trigger independence from the full reference filtration;
35. explicit treatment of WWR default atoms;
36. decisive cross-default/additivity semantics;
37. no silent repair, extrapolation, scenario removal, or measure substitution;
38. no XVA abstraction introduced solely because the term "XVA" appears;
39. a defensible mathematical owner for every architectural abstraction;
40. no double counting of default loss;
41. explicit statement of what changes for collateral, WWR, bilateral default, or broader XVA.

Only after these conditions are satisfied should RiskQuant move beyond unilateral first-generation CVA.

---
