---
marp: true
theme: uncover
class: invert
style: |
  section {
    font-size: 20px;
    text-align: left;
  }
  h2 {
    font-size: 38px;
  }
---

<style scoped>
h2 {
    font-size: 68px;
  }
</style>

## RiskQuant  Credit VaR

---

## The Commander-in-Chief's Question

*Can you give me a high level summary of how long it would take to build a CvaR model?  I don’t want something massive – as basic as possible.  Worst-case we could just look at something where we just apply a stress test and assume our top 2-3 sub-investment grade exposures default but I want to compare what that would look like vs. having a model where we run simulations and can articulate a loss at 95th or 97.5th percentile.*

---

## Solution

**Default-mode (jump-to-default) credit loss models. One engine. One input set.**

---

## Business Question (1/2)

**What these models help us answer:**
- **How much capital** should we hold against credit default losses? <!-- (economic capital = VaR/ES − EL) -->
- **What loss should we expect** over the horizon? (provisioning / planning input)
- **How bad is the tail** at 97.5 / 99%? (risk appetite, stress)
- **Where is the risk concentrated**; which names and sectors drive the tail? (attribution, limits)

---

## Business Question (2/2)

**What they do *not* solve or model:**
- **Not market risk**: default-mode only; no credit-spread / mark-to-market P&L.
- **Not rating migration**: single default horizon, not multi-period staging.
- **Not PD / LGD / EAD estimation**: they **consume** these as inputs, they don't produce them.
- **No default timing, no multi-year paths, no contagion** (no "A defaults ⇒ B defaults" cascade).
- **They don't predict *which* scenario occurs**: they characterise a **distribution** of possible outcomes.
- **They don't consider wrong-way risk**: PDs do not scale with exposure.

**Bottom line:** the models size and attribute **credit default loss over one time horizon**; this is not a market-risk, migration, or PD/LGD-estimation tool.

---

## How to Read this Deck

- **One framework, many models.** Every model answers the same question - *how big can credit losses get?* - and differs in **one thing only: how defaults move together**.
<!-- - **Expected Loss is the same for all of them.** Correlation does not change the average loss; it changes the **tail** (the 97.5th / 99th percentile). That is the entire reason we have more than one model. -->
- Each model slide is laid out the same way: **type → inputs → maths → correlation → calibration → risks → relationships → bottom line.**

**Bottom line:** the model choice is a choice about *the tail*, not the average.

---

## The Core Idea

- Each obligor *i* has three numbers: **EAD** (exposure), **PD** (*n*-year default probability), **LGD** (loss given default).
- Default is binary: **Dᵢ ∈ {0, 1}**. Loss if it defaults: **Lossᵢ = Dᵢ × EADᵢ × LGDᵢ**.
- Portfolio loss in a scenario: **L = Σᵢ Lossᵢ**. We simulate many scenarios → a **loss distribution**.
- From that distribution we read: **EL** (expected loss), loss std. deviation, **VaR** and **ES** at 95 / 97.5 / 99%, **UL** (VaR − EL).

<!-- > **Expected Loss ≈ Σ PDᵢ × LGDᵢ × EADᵢ - independent of the model.**
> Models differ only in the **dependence structure** between the Dᵢ, which sets the tail. -->

**Bottom line:** all our numbers come from one simulated loss distribution; the models mostly reshapes its tail.

---

## Model Assumptions

<style scoped>
section {
  font-size: 17px;
}
</style>

**Shared by every model (state these on every output):**
- **Default mode (jump-to-default)**: loss only on outright default in the horizon; no rating-migration / mark-to-market loss.
- **Single period**: one horizon, set by the PD; no timing of default within it, no multi-year path.
- **Deterministic LGD**: recovery is a fixed fraction; no LGD volatility and no PD–LGD (downturn-LGD) link.
- **Inputs are exogenous and exact**: every figure is **conditional** on PD / LGD / EAD (garbage in, garbage out).
- **Dependence via shared factors only**: correlation comes from common factors / regimes / pools; **no direct contagion**.
- **Static book**: the portfolio is fixed over the horizon (no new business, amortisation, or hedging).
- **Monte-Carlo estimates**: tail figures carry simulation error; the deep tail needs enough scenarios to be stable.

**Model-specific assumptions to flag:**
- pool **homogeneity** (Beta-Binomial ignores per-name PD);
- **binary** regime (Stress-Regime);
- **Gaussian tail-independence** (Gaussian copulas);
- correlation is a **regulatory prescription, not fitted** (Basel variants).

**Bottom line:** point-in-time, default-only, static-book, conditional-on-inputs — read every number through these.

---

## Required Input Data

<style scoped>
section {
  font-size: 17.8px;
}
</style>

**Per obligor — mandatory (the three numbers that drive everything):**

| Field | Meaning | Type | Source |
|---|---|---|---|
| `uid` | obligor identifier | id | exposure / reference systems |
| `EAD` | exposure at default | currency | exposure / limits systems |
| `PD` | 1-yr default probability, [0, 1) | rate | rating / scoring models |
| `LGD` | loss given default, [0, 1] | rate | recovery / collateral models |

**Per obligor — optional, model-dependent:** `group` label (grouped models) · **factor loadings** `(uid, factor, loading)` (multi-factor) · per-name `rho` override (SF copulas) · overlay flags `force_default`, `in_stress_test`.

**Model parameters:** ρ (SF Gaussian) · ν (Student-t) · α, β (Beta-Binomial) · stress prob & PD multiplier (Stress-Regime) · loadings (MF). **Basel variants derive ρ from PD — no ρ input.**

**Run controls:** scenario count · confidence levels · random seed (for reproducibility).

**Data-quality dependencies:** results are **most sensitive to PD and LGD accuracy**; then EAD completeness, consistent obligor IDs, and (for MF / grouped) the quality of sector / group mapping.

**Bottom line:** three numbers per name drive the result; the correlation inputs are few but set the tail — **PD/LGD accuracy is the binding constraint.**

---

## The Model Ladder (Taxonomy)

<style scoped>
section {
  font-size: 12px;
}
</style>

| Model | Type | Correlation Mechanism | Tail | Calibrate |
|---|---|---|---|---|
| Forced Default | Deterministic stress | n/a (certainty) | — | none |
| Independent Bernoulli | JTD, independent | **none** | thinnest | PDs |
| Beta-Binomial | JTD, common rate | shared random default rate | moderate | α, β |
| Beta-Binomial (Grouped) | JTD, segmented | shared rate **within pool** | moderate | α, β per pool |
| Stress-Regime Bernoulli | JTD, regime mixture | shared on/off regime | moderate–fat | prob, multiplier |
| Stress-Regime (Grouped) | JTD, segmented | shared regime **within group** | moderate–fat | prob, mult per group |
| SF Gaussian Copula | JTD, 1-factor copula | one systemic factor, ρ | fat | ρ |
| SF Gaussian (Basel ρ) | JTD, 1-factor copula | ρ = R(PD) regulatory | fat | none (from PD) |
| SF Student-t Copula | JTD, 1-factor + heavy tail | factor ρ **+ tail dependence** | fatter | ρ, ν |
| SF Student-t (Basel ρ) | JTD, 1-factor + heavy tail | R(PD) + tail dependence | fatter | ν |
| MF Gaussian Copula | JTD, multi-factor copula | **multiple** sector factors | fat, sector-aware | loadings |
| MF Student-t Copula | JTD, multi-factor + heavy tail | sectors + tail dependence | fattest | loadings, ν |

**Bottom line:** read top-to-bottom as increasing realism of *how defaults cluster* — and increasing calibration burden.

---

## Forced Default

**Type:** deterministic stress (degenerate JTD) · **Correlation:** n/a · **Calibrate:** none · **Tail:** point mass

- **Models:** the pure consequence of a chosen set of names all defaulting. PD is ignored.
- **Inputs:** EAD, LGD (PD unused).
- **Maths:** Dᵢ = 1 for all *i* → **L = Σ EADᵢ × LGDᵢ** (one outcome; VaR = ES = EL).
- **Correlation:** none needed — everything defaults together by assumption.
- **Calibration:** none.
- **Model risks:** not a probability statement (says nothing about *likelihood*); absurdly conservative if applied to the whole book; the stressed set is chosen judgmentally.
- **Relationships:** the deterministic limit of every other model; same engine, run with certainty.
- **Bottom line:** the assumption-light "if these names go, we lose $X" — a clean floor, not a forecast.

---

## Independent Bernoulli

**Type:** JTD, independent defaults · **Correlation:** none · **Calibrate:** PDs only · **Tail:** thinnest

- **Models:** each name defaults on its own coin-flip at its own PD, with **no linkage**.
- **Inputs:** EAD, PD, LGD.
- **Maths:** **Dᵢ ~ Bernoulli(PDᵢ)**, mutually independent. EL = Σ PDᵢ·LGDᵢ·EADᵢ.
- **Correlation:** explicitly **zero**.
- **Calibration:** none beyond the PD inputs.
- **Model risks:** **understates the tail whenever defaults are correlated** — i.e. always, in a downturn. Overstates diversification; the 99th-percentile comes out too low.
- **Relationships:** the **lower bound** for every correlated model — real correlation only makes the tail worse.
- **Bottom line:** great for Expected Loss and a tail *floor*; never quote its 99% as the firm's tail risk.

---

## Beta-Binomial

**Type:** JTD, common stochastic default rate (mixture) · **Correlation:** shared random rate · **Calibrate:** α, β · **Tail:** moderate

- **Models:** "good years / bad years" — a single **random** default rate is shared by the pool, so defaults cluster.
- **Inputs:** EAD, LGD, and pool α, β. **Note: per-obligor PD is *not* used** — the pool default rate is α / (α + β).
- **Maths:** draw **p ~ Beta(α, β)** per scenario; each obligor defaults at that common p. EL = (α / (α+β)) × Σ LGDᵢ·EADᵢ.
- **Correlation:** induced by the **shared** p — high-p scenarios default many names at once.
- **Calibration:** α, β set the **mean** default rate (α/(α+β)) and its **volatility** (dispersion), from historical pool default-rate data.
- **Model risks:** **homogeneity** — ignores PD differences between names; single driver; suits a uniform pool, not a book of distinct large credits.
- **Relationships:** the simplest step up from Independent Bernoulli (adds one common factor); homogeneous-pool cousin of the copula models.
- **Bottom line:** a quick correlated-pool tail for *similar* credits (retail / SME), not for a lumpy corporate book.

---

## Beta-Binomial — Grouped (Segmented Pools)

**Type:** JTD, segmented common-rate · **Correlation:** shared rate **within a pool**, independent across · **Calibrate:** α, β per pool · **Tail:** moderate

- **Models:** the Beta-Binomial, but the book is split into **pools** (e.g. by rating band or sector); each pool has its **own** random default rate.
- **Inputs:** EAD, LGD, a **pool label** per obligor, and (α, β) **per pool**.
- **Maths:** one **pⱼ ~ Beta(αⱼ, βⱼ)** per pool *j*; names in pool *j* default at pⱼ. **Block-diagonal** correlation (within-pool correlated, across-pool independent).
- **Correlation:** within a pool only — there is **no** cross-pool linkage.
- **Calibration:** (α, β) per pool, each to that segment's mean rate and volatility.
- **Model risks:** as Beta-Binomial, per pool (homogeneity *within* a pool); assumes pools are independent (no shared macro factor across pools).
- **Relationships:** Beta-Binomial generalised to *segments*; the "grouped" (not multi-factor) family — each name sits in **one** pool.
- **Bottom line:** lets us differentiate IG vs HY (or sector pools) while keeping the model simple.

---

## Stress-Regime Bernoulli

<style scoped>
section {
  font-size: 20px;
}
</style>

**Type:** JTD, two-state regime mixture · **Correlation:** shared on/off regime · **Calibrate:** stress prob, PD multiplier · **Tail:** moderate–fat

- **Models:** an explicit **downturn switch** — with some probability the economy is "stressed" and everyone's PD jumps.
- **Inputs:** EAD, PD, LGD, stress probability q, PD multiplier m. **Per-obligor PD *is* used** (unlike Beta-Binomial).
- **Maths:** regime **R ~ Bernoulli(q)**; effective PDᵢ = PDᵢ in normal, **min(PDᵢ × m, 1)** in stress; then **Dᵢ ~ Bernoulli(effective PDᵢ)**.
- **Correlation:** the **shared regime** — in a stress scenario all PDs rise together, clustering defaults.
- **Calibration:** **q** from historical downturn frequency (~1-in-7 to 1-in-10 yrs); **m** from PD uplift in downturns (~2–4×). Both are committee-friendly, transparent levers.
- **Model risks:** the binary regime is coarse; conditional independence *within* a regime understates within-regime clustering; q and m are judgmental.
- **Relationships:** a *probabilistic* version of the Forced-Default stress; bridges stress testing and simulation.
- **Bottom line:** the model when you want a tail driven by a **describable** event (a recession), not an abstract correlation number.

---

## Stress-Regime Bernoulli — Grouped (Segmented Regimes)

<style scoped>
section {
  font-size: 20px;
}
</style>

**Type:** JTD, segmented regime mixture · **Correlation:** shared regime **within a group** · **Calibrate:** prob, multiplier per group · **Tail:** moderate–fat

- **Models:** the stress regime, but each **group** has its own regime (e.g. an energy-sector shock independent of a rates shock).
- **Inputs:** EAD, PD, LGD, a **group label** per obligor, and (stress prob, multiplier) **per group**.
- **Maths:** one regime **Rⱼ ~ Bernoulli(qⱼ)** per group *j*; in-group PDs scale by mⱼ when Rⱼ fires. **Block-diagonal** (regimes independent across groups).
- **Correlation:** within a group via its shared regime; independent across groups.
- **Calibration:** per group — each segment's downturn frequency and severity.
- **Model risks:** as Stress-Regime, per group; assumes group regimes are independent (no common macro switch).
- **Relationships:** Stress-Regime generalised to segments; "grouped" family (each name in **one** regime).
- **Bottom line:** model sector-specific downturns hitting different sub-portfolios at different times.

---

## Single-Factor Gaussian Copula

<style scoped>
section {
  font-size: 20px;
}
</style>

**Type:** JTD, one-factor copula (Vasicek / Basel ASRF / CreditMetrics) · **Correlation:** one systemic factor, ρ · **Calibrate:** ρ · **Tail:** fat

- **Models:** systematic clustering through a single "state of the economy" factor, while keeping each name's own PD.
- **Inputs:** EAD, PD, LGD, asset correlation **ρ** (global, with optional per-obligor override).
- **Maths:** latent **Aᵢ = √ρ · M + √(1−ρ) · εᵢ**, with M ~ N(0,1) systemic, εᵢ ~ N(0,1) idiosyncratic. **Default if Aᵢ ≤ Φ⁻¹(PDᵢ).**
- **Correlation:** the **shared M**; ρ is the asset (latent) correlation — one intuitive dial.
- **Calibration:** ρ from the Basel prescription (~0.12–0.24) or estimated from default-rate / equity correlations. **The tail is highly sensitive to ρ.**
- **Model risks:** ρ misspecification (small errors move the tail a lot); **Gaussian copula has weak tail dependence** — can under-state many-names-default-together crises (the 2008 critique); single factor (no sectors).
- **Relationships:** the **industry / regulatory standard** (basis of Basel IRB); single-factor special case of the multi-factor copula.
- **Bottom line:** the default model for a credible 97.5 / 99% number — what we put next to the stress test.

---

## SF Gaussian — Basel ρ Variant

<style scoped>
section {
  font-size: 20px;
}
</style>

**Type:** as SF Gaussian, regulatory correlation · **Correlation:** ρ = R(PD) per obligor · **Calibrate:** none (from PD) · **Tail:** fat

- **Delta vs base:** instead of one global ρ, each obligor's correlation is the **Basel II/III IRB corporate weighting**.
- **Maths:** **R(PD) = 0.12·w + 0.24·(1−w)**, with **w = (1 − e^(−50·PD)) / (1 − e^(−50))** → **0.24 at PD→0, 0.12 at PD→1** (better credits are *more* systematic). Used as ρᵢ in the SF Gaussian latent equation.
- **Correlation:** per-name, PD-driven; a bespoke per-obligor ρ still overrides where supplied.
- **Calibration:** **none** — ρ is a deterministic function of PD (regulatory).
- **Model risks:** the regulatory curve is a *prescription*, not estimated from our book (may not fit our portfolio); corporate curve only (SME-size / retail variants not applied).
- **Relationships:** SF Gaussian with the rho-sourcing swapped to the Basel formula; the most defensible-to-regulators correlation choice.
- **Bottom line:** the regulator-aligned SF Gaussian — no ρ to argue about, it comes from PD.

---

## Single-Factor Student-t Copula

<style scoped>
section {
  font-size: 20px;
}
</style>

**Type:** JTD, one-factor copula with heavy tails · **Correlation:** factor ρ + **tail dependence** · **Calibrate:** ρ, ν · **Tail:** fatter

- **Models:** the SF Gaussian **plus** a built-in tendency for many names to default *together* in extremes.
- **Inputs:** EAD, PD, LGD, ρ, degrees of freedom **ν**.
- **Maths:** as SF Gaussian but the latent variables share a common **chi-square / Student-t mixing at ν**; low ν ⇒ fat joint tails, ν → ∞ ⇒ Gaussian.
- **Correlation:** systematic factor ρ **and tail dependence** (joint extremes), which the Gaussian misses.
- **Calibration:** ρ as before; **ν** from tail-dependence estimates or expert judgment (≈ 5–10 typical). Tail is sensitive to ν.
- **Model risks:** **ν is genuinely hard to calibrate**; risk of double-counting conservatism (low ν *and* high ρ); still single factor.
- **Relationships:** SF Gaussian + tail dependence; the post-2008 "stress-aware" upgrade.
- **Bottom line:** a more conservative tail than Gaussian at the same ρ — use when crisis co-movement is the worry.

---

## SF Student-t — Basel ρ Variant

<style scoped>
section {
  font-size: 20px;
}
</style>

**Type:** as SF Student-t, regulatory correlation · **Correlation:** R(PD) + tail dependence · **Calibrate:** ν · **Tail:** fatter

- **Delta:** SF Student-t with ρ supplied by the **Basel R(PD)** curve (only ν remains a free parameter).
- **Correlation:** PD-driven per-name ρ **and** tail dependence at ν; per-name ρ still overrides.
- **Calibration:** ν only; ρ from PD.
- **Model risks:** combines the t-copula's ν-calibration difficulty with the prescription-not-fit risk of the regulatory curve.
- **Relationships:** the regulator-aligned, crisis-aware single-factor model.
- **Bottom line:** Basel correlation with a fat tail — the conservative regulator-aligned option.

---

## Multi-Factor Gaussian Copula

<style scoped>
section {
  font-size: 20px;
}
</style>

**Type:** JTD, multi-factor copula · **Correlation:** several sector/region factors, per-obligor loadings · **Calibrate:** loadings · **Tail:** fat, sector-aware

- **Models:** default dependence with **sector / region structure** — names in the same sector correlate more, so **concentration shows up**.
- **Inputs:** EAD, PD, LGD, and per-obligor **factor loadings** (which systemic factors a name loads on, and how much).
- **Maths:** **Aᵢ = Σₖ βᵢₖ · Fₖ + √(1 − Σₖ βᵢₖ²) · εᵢ**, with Fₖ ~ N(0,1) systemic factors and loadings βᵢₖ (Σβ² ≤ 1). Default if Aᵢ ≤ Φ⁻¹(PDᵢ).
- **Correlation:** **multiple** shared factors; a name can load on **several at once** — this is the genuine *multi-factor* case (vs the "grouped" models where a name sits in one pool).
- **Calibration:** factor definitions + per-obligor loadings, from sector/region membership and an intra-/inter-sector correlation matrix.
- **Model risks:** loading / factor misspecification; **data-hungry**; still Gaussian tail-independence (no extreme co-movement).
- **Relationships:** generalises SF Gaussian to many factors; the right tool for **concentration** questions.
- **Bottom line:** the model when "are we over-exposed to one sector?" is the question.

---

## Multi-Factor Student-t Copula

<style scoped>
section {
  font-size: 20px;
}
</style>

**Type:** JTD, multi-factor copula with heavy tails · **Correlation:** sector factors + tail dependence · **Calibrate:** loadings, ν · **Tail:** fattest

- **Models:** the most complete — sector/region clustering **and** crisis co-movement together.
- **Inputs:** EAD, PD, LGD, factor loadings, ν.
- **Maths:** the MF Gaussian latent structure **plus** the shared chi-square / Student-t mixing at ν.
- **Correlation:** multi-factor (sectors) **and** tail dependence.
- **Calibration:** loadings (as MF Gaussian) **plus** ν — the heaviest calibration burden.
- **Model risks:** compounded calibration risk (loadings *and* ν); easy to over-engineer relative to the question / data.
- **Relationships:** MF Gaussian + tail dependence; the top of the ladder.
- **Bottom line:** worst-case, fully-loaded tail — reach for it only when concentration *and* crisis co-movement both matter.

---

## Tail Severity at a Glance

<style scoped>
section {
  font-size: 20px;
}
</style>

- **Thinnest → fattest tail (same book, same PDs):**
  Independent Bernoulli → Beta-Binomial / Stress-Regime → **SF Gaussian** → SF Student-t / MF Gaussian → **MF Student-t**.
- **Forced Default** sits outside the ladder: a *deterministic* worst case for the chosen names.
- **Expected Loss is the same across all of them** — only the tail moves.
- The Basel and grouped variants don't change the family's tail character; they change **where the correlation comes from** (PD-driven, or segmented).

**Bottom line:** moving down the ladder buys realism in the tail at the cost of more calibration.

---

## Choosing a Model

<style scoped>
section {
  font-size: 18px;
}
</style>

| The Question | Use |
|---|---|
| "What if our worst 2–3 names default?" | **Forced Default** (hard stress) |
| "What's our expected loss?" | **Independent Bernoulli** (EL is model-invariant) |
| "What's a credible 97.5 / 99% loss?" | **SF Gaussian** (or **Basel** ρ for regulator alignment) |
| "What if a downturn hits?" (describable) | **Stress-Regime** (grouped for sector-specific shocks) |
| "What about crisis co-movement / fat tails?" | **SF Student-t** |
| "Are we over-concentrated in a sector?" | **MF Gaussian** |
| "Concentration *and* crisis co-movement?" | **MF Student-t** |
| Homogeneous retail / SME pool(s) | **Beta-Binomial** (grouped for multiple segments) |


**Recommended headline trio:** Forced Default (stress) · Independent Bernoulli (EL & floor) · SF Gaussian (the percentile number). Escalate only when tail dependence or concentration is the specific concern.

---

## Governance & Validation

<style scoped>
section {
  font-size: 20px;
}
</style>

- **Reproducible:** every run records model, parameters, scenario count, seed and an input hash — same inputs reproduce the numbers exactly.
- **Benchmarks we check:** EL reconciles to Σ PD·LGD·EAD; the tail is **monotone in correlation** (more ρ ⇒ fatter tail); per-obligor tail contributions **foot** to portfolio ES / UL.
- **Monte-Carlo error:** deep-tail figures (e.g. 99.9th) need enough scenarios to be stable — quote tail numbers with that in mind.
- **Standing caveats (print on outputs):** default-mode only; deterministic LGD; single period; conditional on PD/LGD/EAD and the correlation assumption.
- **Validation status:** _[to be completed with Model Validation]._

**Bottom line:** standard, literature-grounded models; reproducible and self-checking; limitations stated up front.

---

## Some Next Steps

<style scoped>
section {
  font-size: 20px;
}
</style>

- **Model validation sign-off**: independent review of methodology, implementation and parameters (the outstanding validation-status line).
- **Calibrate correlation to our book**: estimate ρ / ν / loadings from **internal** default and sector data rather than the current Basel / expert defaults; **add SME-size and retail Basel curves** (only the corporate R(PD) is implemented today).
- **Backtest & benchmark**: reconcile EL to provisions; compare the tail to realised and stress outcomes; document VaR sensitivity to ρ and ν.
- **Reporting integration**: wire the loss distribution and per-obligor / per-sector tail attribution into the presentation and Excel reporting layer.
- **Governance**: agree the standard model(s) for **capital** vs **stress**, the run cadence, and parameter-change control.
- **Documentation**: finalise the Technical Model Document and complete its validation section.

**Bottom line:** the engine is built, reproducible and self-checking — the path to production is **validation + calibration-to-our-book + governance sign-off.**

---

## Appendix: Glossary

<style scoped>
section {
  font-size: 16px;
}
</style>

- **Jump-to-Default (JTD):** loss only from outright default in the horizon (default mode).
- **Expected Loss (EL):** expected loss (mean).
- **Unexpected Loss (UL):** unexpected loss - a capital buffer defined as a high quantile (VaR or ES-based) minus expected loss (EL).
- **VaR / ES:** loss not exceeded at confidence α / average loss in the worst (1−α) — ES is the conditional mean of losses beyond the VaR threshold (tail conditional expectation), and is coherent in the sense of satisfying subadditivity.
- **Economic capital (EC):** internally chosen extreme loss measure (usually VaR or ES) minus expected loss (EL), across all risk types, at a firm-defined confidence level. Note: UL is a credit-risk decomposition concept; EC is a firm-wide capital allocation construct.
- **Asset correlation (ρ):** correlation of latent systematic risk factors in structural / copula-based default models (e.g., Gaussian copula / ASRF framework).
- **Copula:** the dependence structure linking individual PDs into a joint distribution; Gaussian vs Student-t differ in **tail dependence**.
- **Tail dependence:** tendency for extreme outcomes (many defaults) to occur together — present in the t-copula, weak in the Gaussian.
- **Factor loading (β):** Sensitivity of obligor’s latent asset value to the systematic factor in a one-factor or multi-factor credit model.
- **R(PD):** Regulatory (Basel) asset correlation mapping in IRB, where correlation decreases with PD (for corporate exposures), reflecting diversification of higher-risk names.

<!-- --- -->

<!-- ## Appendix — slide-conversion notes

- Each `## ` heading = one slide; `---` separates slides.
- **Marp / reveal-md:** renders as-is.
- **Pandoc → PPTX:** `pandoc CREDIT_VAR_MODEL_REVIEW.md -o deck.pptx --slide-level=2`.
- Tables and the unicode maths render in all of the above; no LaTeX required. -->
