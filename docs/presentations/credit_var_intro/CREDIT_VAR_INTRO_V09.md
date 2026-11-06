---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    background-color: #181A1F;
    color: #D1D5DB;
    font-size: 24px;
    padding: 70px 90px;
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
  }
  h1 { color: #ffffff; font-size: 42px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 22px; }
  h2 { color: #FFFFFF; font-size: 28px; font-weight: 600; margin-bottom: 20px; }
  h3 { color: #E30076; font-size: 22px; font-weight: 500; }
  p, li { line-height: 1.55; color: #D1D5DB; }
  strong { color: #ffffff; font-weight: 600; }
  em { color: #FF66B2; font-style: italic; }

  table { width: 100%; font-size: 18px; border-collapse: collapse; margin-top: 18px; }
  table tr { background-color: transparent !important; }
  table tr:nth-child(2n) { background-color: rgba(255,255,255,0.03) !important; }
  th, td { border: none !important; padding: 11px 16px; text-align: left; }
  th { border-bottom: 2px solid #E30076 !important; color: #FFFFFF; font-weight: 600; }
  td { color: #E8E8E8 !important; border-bottom: 1px solid rgba(255,255,255,0.06) !important; }

  blockquote {
    border-left: 4px solid #E30076; margin: 24px 0; padding: 14px 22px;
    background: rgba(255,255,255,0.04); border-radius: 0 8px 8px 0;
    color: #FFFFFF; font-size: 22px; font-weight: bold; font-style: italic;
  }
  pre { background: #0F1114; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; font-size: 16px; }
  code { background: #0F1114; color: #FFFFFF; padding: 3px 7px; border-radius: 4px; font-style: normal; }
  section::after { color: rgba(255,255,255,0.3); }
  section::before { content: ''; position: absolute; bottom: 0; left: 0; width: 100%; height: 4px; background: #E30076; }
  section.lead h1 {
    font-size: 72px;
    line-height: 1.0;
    margin-bottom: 8px;
    font-weight: 700;
  }
  section.lead h2 {
      font-size: 32px;
      font-weight: 400;
      color: #CCCCCC;
      margin-top: 28px;
  }
  section.lead blockquote {
      margin-top: 48px;
      font-size: 24px;
  }
  img {
    display: block;
    margin: 0 auto;
  }
---


<!-- _class: lead -->

# RiskQuant · Credit VaR

## *Credit losses when counterparties default together*

{{DATE}}

<style scoped>
  p {
    font-size: 18px;
    font-style: italic;
    color: #888888;
    margin-top: 40px;
  }
</style>

---

<!-- _class: lead -->

<style scoped>
  section {
    background-color: #E30076;
    text-align: center;
  }
  h1, em {
    color: #0F1114 !important;
    text-align: center;
  }
</style>

# EXECUTIVE SUMMARY

---

<!-- _class: lead -->


# Part 0 · Where We Stand Today

## *What Credit VaR needs, what exists today, and what can run now*

> The default models and the Excel tool\* exist today. Exposure at default is the one input that needs more work.

<style scoped>
  .footnote {
    position: absolute;
    bottom: 30px;
    left: 90px;
    font-size: 12px !important;
    color: #888888 !important;
    font-style: italic !important;
  }
</style>

<div class="footnote">
  * The Excel tool is effectively deprecated. The web app replaces it, with delivery planned for November 2026.
</div>

---

# What Credit VaR Needs

Credit VaR needs four inputs. Exposure at default (EAD) and loss given default (LGD) set the size of each loss. Probability of default (PD) and default dependence set how often counterparties default, alone or together.

| # | Input | In plain terms |
|---|---|---|
| 1 | **Exposure at default (EAD)** | The amount we are owed when a counterparty defaults |
| 2 | **Loss given default (LGD)** | The share of that amount we lose after recovery |
| 3 | **Probability of default (PD)** | The chance that each counterparty defaults |
| 4 | **Default dependence** | The chance that counterparties default together |

> EAD, LGD and PD set the expected loss in most models. If a model keeps each counterparty's PD, default dependence leaves the expected loss unchanged and shapes the tail. Some models replace or scale the PD, so they also change the expected loss.

---

# What We Already Have

LGD, PD and default dependence exist today. Exposure at default exists in part.

| # | Input | Status |
|---|---|---|
| 1 | Exposure at default (EAD) | **Partial** · The current EAD input is stressed Replacement Cost (RC): the net MTM of each netting set after any collateral, floored at zero. SA-CCR EAD is a separate calculation: 1.4 × (RC + PFE). PFE is a supervisory add-on for possible future exposure. SA-CCR support covers unmargined linear trades only. A forward-looking exposure model is a separate investment. |
| 2 | Loss given default (LGD) | **Have** · input per counterparty |
| 3 | Probability of default (PD) | **Have** · input per counterparty |
| 4 | Default dependence | **Have** · the `riskquant` model suite is built |

> Credit VaR runs today with the (stressed) Replacement Cost as EAD. A forward-looking exposure model, which gives exposure over time, is planned.

---

# What We Can Do Today

Each model runs on the current or stressed Replacement Cost (RC) as EAD. In each scenario, the loss is EAD × LGD, summed over the counterparties that default.

<style scoped>
  /* Shrinks the table specifically for this slide */
  table { font-size: 15px !important; }
  th, td { padding: 8px 12px !important; }
  code { font-size: 14px; }

  /* Bypasses the style-stripper by using a class */
  .footnote {
    position: absolute;
    bottom: 30px;
    left: 90px;
    font-size: 12px !important;
    color: #888888 !important;
    font-style: italic !important;
    z-index: 10;
  }
</style>


| Model | What it assumes |
|---|---|
| **Independent Bernoulli** | Each counterparty defaults on its own. No default clustering. |
| **Stress-Regime Bernoulli** | In some scenarios a shared stress state multiplies every PD. |
| **Beta-Binomial** | One shared default probability is drawn for each scenario. |
| **Single-Factor Gaussian Copula** | One shared factor drives all defaults. |
| **Single-Factor Student-t Copula** | The same shared factor, with more default clustering in extreme scenarios. |


> All five models use the same portfolio. Each model allows a different amount of default clustering, so the tail loss can differ. Forced Default (a deterministic upper bound) and other models are also available.

<!-- <div class="footnote">
  * Actual OTC positions and market values used. Credit assumptions (PD/LGD) are illustrative inputs. Stress cases revalue the portfolio under EUR price shocks and recompute exposure. No collateral benefit assumed.
</div> -->

---

# STX B.V. - GOO - Base Case

![width:1020px](images/credit_var_nl_goo_base.png)

---

# Credit VaR is Operational Today

The Excel tool runs Credit VaR on an uploaded portfolio.

- **Configure:** Enter counterparties with EAD, PD and LGD. Select a model.
- **Run:** One click returns the loss distribution, VaR, ES and unexpected credit loss (UL).
- **Attribute:** ES contribution and incremental VaR and ES show which counterparties drive the tail.
- **Stress:** Force chosen counterparties to default inside the simulation, or list them for a separate what-if loss. Change exposures outside the tool and run again.
- **Exposure (now):** EAD comes from the current or stressed Replacement Cost (RC).
- **Exposure (future):** A forward-looking exposure model is the next step.

> Each run records the model, the inputs, the seed and the software version.

> Note that the Excel tool is deprecated in favour of a web app based tool (planned Nov-2026).
---

# RiskQuant Credit VaR Excel Tool

![width:900px](images/rq_credit_var_excel_tool_control.png)

---

# RiskQuant Credit VaR Excel Tool

![width:900px](images/rq_credit_var_excel_tool_obligors.png)

---

<!-- _class: lead -->

<style scoped>
  section {
    background-color: #E30076;
    text-align: center;
  }
  h1, em {
    color: #0F1114 !important;
    text-align: center;
  }
</style>

# FORBIDDEN MATERIALS

---

<!-- _class: lead -->

# Part 1 · Credit VaR

## *Default dependence assumptions shape the tail loss*

---

# What Credit VaR Measures

Credit VaR answers one question: *if counterparties default, what losses could our trading portfolio experience?*

> **Expected Loss is the average credit loss. Credit VaR is the loss at a chosen confidence level, for example 99%.**

| Component | Question | Determines |
|---|---|---|
| **Exposure and LGD** | How much do we lose if a counterparty defaults? | The size of each loss |
| **Default probability** | How likely is each counterparty to default? | The average loss |
| **Default dependence** | How likely are counterparties to **default together**? | **The shape of the tail (extreme losses)** |

---

# Expected Loss and Credit VaR

**Expected Loss** is the average counterparty credit loss.

**Credit VaR** is the credit loss that scenarios exceed with a small, chosen probability (e.g., 1%, which is a 99% confidence level).

| Expected Loss | Credit VaR |
|---|---|
| "What is our average credit loss?" | "What loss could we face in a severe event?" |
| Driven by exposure, LGD and PD | Driven by exposure, LGD, PD and **default dependence** |
| Typically used to estimate the expected cost of counterparty default | Typically used to inform capital and risk appetite |

> Counterparties can default together. Credit VaR measures the tail of the loss, including the effect of default clustering (imposed by a model).

---

# The Business Questions Behind the Tail

- **How much** could we lose in a severe counterparty default event?
- **Which counterparties** contribute most to the tail loss?
- **How much** would the tail fall if we exit one counterparty?
- **What is the loss** if a chosen group of counterparties defaults together?
- **How much** does the tail change when the dependence assumption changes?

> Counterparties can default together. Each model makes a different assumption about how often that happens.

---

# The Core Data Behind Credit VaR

Credit VaR uses **four** inputs. EAD and LGD set the **size of a loss**. PD and default dependence set **how often losses occur and how much they cluster**.

| Input | Question answered | Source |
|---|---|---|
| **Exposure at default (EAD)** | How much value is exposed if a counterparty defaults? | Exposure engine (current/stressed RC) or an external exposure model |
| **Loss given default (LGD)** | How much of that exposure is lost after recovery? | Counterparty table |
| **Probability of default (PD)** | How likely is each counterparty to default? | Counterparty table |
| **Default dependence** | How likely are defaults to occur together? | Model choice and parameters (for example ρ, ν, factor loadings) |

> For each counterparty, the loss is EAD × LGD when the counterparty defaults. Default dependence decides which counterparties default in the same scenario.

---

# Where it Sits in the Risk Stack

- The **exposure engine** calculates EAD for each counterparty. Today the EAD supplied to Credit VaR is the current or stressed Replacement Cost (RC). A forward-looking model (exposure over time) is not available yet.
- The **Credit VaR engine** simulates defaults and sums the losses.
- A credit loss arises only when a counterparty's trades have positive net replacement value at default.

```text
   Trades and MTM
     │
     ▼
   Exposure Engine            netting · collateral · replacement cost
     │
     ▼
   Counterparty EAD       ──►  amount owed to STX if the counterparty defaults
     │
     ▼
   Credit VaR Engine           default simulation · dependence · aggregation
     │
     ▼
   Loss Distribution      ──►  risk metrics (EL · VaR · ES · UL)
```

---

<!-- _class: lead -->

# Part 2 · Jump-to-Default (JTD)

## *From counterparty defaults to portfolio loss distributions*

> A default creates a loss.
> Default dependence decides how many losses occur in the same scenario.

---

<!-- # Jump-to-Default (In a Nutshell)

**The event:** a counterparty defaults. Our positive net exposure to it at the time of default is the **exposure at default (EAD)**. The credit loss is **EAD × LGD**. **LGD** is the share of **EAD** that we do not recover.

We do not lend principal. Our credit exposure is the net mark-to-market that a counterparty owes us on unrealised trades. Trades offset each other only inside the same legally enforceable netting set. A counterparty can have several netting sets, and their exposures add up. Collateral reduces the amount further. Netting sets are not yet clearly defined.

**Each scenario:**

1. Take the exposure at default (EAD) of each counterparty.
2. Decide which counterparties default.
3. Apply the LGD.
4. Sum the losses over the portfolio.

Repeat for many scenarios (250,000 by default). The result is the portfolio loss distribution.

> Step 2 is the modelling choice. When one counterparty defaults, how likely is another to default in the same scenario?

--- -->

# Jump-to-Default (In a Nutshell)

**The event:** a counterparty defaults. Our positive net exposure to it is the **exposure at default (EAD)**. The credit loss is **EAD × LGD**, where **LGD** is the share of **EAD** that we do not recover.

Exposure is the net mark-to-market that a counterparty owes us, after any collateral. Trades offset only inside the same legally enforceable netting set. Exposures of different netting sets add up. Netting sets are not yet clearly defined.

**Each scenario:** take the EAD of each counterparty, decide which counterparties default, apply the LGD, and sum the losses. Repeat for 250,000 scenarios (default). The result is the portfolio loss distribution.

> Deciding which counterparties default is the modelling choice. When one counterparty defaults, how likely is another to default in the same scenario?

---

# What JTD Credit VaR Does Not Answer

JTD Credit VaR measures the loss **when default happens**.

It does not answer:

| Question | Where it sits |
|---|---|
| How large could exposure become before default? | CCR modelling |
| What is the loss from a sudden move in market prices? | Market risk |
| Does exposure increase as the counterparty's credit quality falls (wrong-way risk)? | Not modelled. The model does not capture wrong-way risk. In the simulation, EAD is a fixed input. |
| How long does close-out take after default? | CCR modelling |
| How does credit quality fall before default? | Not modelled. This needs a credit migration model. |

> JTD Credit VaR measures default loss only. It is one part of total counterparty risk.

---

# What Drives Expected Loss and Tail Loss

| Driven by exposure, LGD and PD | Shaped by default dependence |
|---|---|
| **Expected Loss** | **The tail: VaR and ES** |
| Same for every model that keeps each counterparty's PD | Can differ strongly between models |

Example: €100m exposure, 40% recovery → **€60m loss if it defaults.**

Models that keep the counterparty's PD give it the same average cost. They differ on how often it defaults in the same scenario as twenty others.

> Exposure, LGD and PD set the average loss. Default dependence shapes the tail loss.

---

# Six Models on the Same Portfolio

The five default models answer the same question: **When one counterparty defaults, how much does it change the likelihood that others default too?** Forced Default is a deterministic upper bound. It has no probability or dependence assumption.

| # | Model | Assumption added |
|---|---|---|
| 1 | Forced Default (bound) | Every counterparty defaults |
| 2 | Independent Bernoulli | Defaults are unrelated |
| 3 | Beta-Binomial | Default rates can move higher or lower together |
| 4 | Stress-Regime Bernoulli | The market can enter a crisis state |
| 5 | Gaussian Copula | Counterparties share common risk drivers |
| 6 | Student-t Copula | Extreme tail dependence |

> Independent Bernoulli and the copulas keep each counterparty's PD. Their expected loss is the same, apart from simulation error. Beta-Binomial replaces the PD and Stress-Regime scales it, so their expected loss differs. The tail loss depends on how much default clustering each model allows.

---

# 1 · Forced Default: *"assume everything fails all at once"*

**Intuition:** every counterparty defaults in every scenario. PD plays no part. This is a deterministic bound. It has no dependence assumption.

**Answers:** *what is the largest default loss for the given EAD and LGD?*

| Strengths | Weaknesses |
|---|---|
| Needs only EAD and LGD | Not a realistic scenario |
| Gives an upper bound for the given EAD and LGD | Not suitable as a capital estimate |

**Use it for:** an upper-bound check, new portfolios with no PD history, and hard stress cases.

> For the same EAD and LGD, no other model gives a loss above this value. A change in EAD or LGD changes the bound.

---

# 2 · Independent Bernoulli: *"unrelated defaults"*

**Intuition:** each counterparty defaults with its own PD. A default does not change the chance of any other default.

**Answers:** *what does the loss look like if defaults are unrelated?*

| Strengths | Weaknesses |
|---|---|
| Simple, fast, gives a full distribution | Does not allow default clustering |
| Captures single-counterparty concentration | **Understates the tail and overstates diversification** |

**Use it for:** granular, homogeneous (sub)portfolios, and as the baseline for the other models.

> Counterparties share common drivers, so this model overstates diversification.

---

# 3 · Beta-Binomial: *"one shared default rate"*

**Intuition:** the model draws one default probability for the whole portfolio in each scenario. Every counterparty uses that probability. In a high-rate scenario, many counterparties default together.

**Answers:** *what happens when the portfolio default rate changes from scenario to scenario?*

| Strengths | Weaknesses |
|---|---|
| Creates default clustering without specifying risk factors | One pool: no sector, geography or counterparty drivers |
| Introduces volatility in the portfolio default rate | Ignores each counterparty's own PD |

**Use it for:** portfolios where default uncertainty matters and a factor model is not required.

> The Beta parameters set the average PD for every counterparty.

---

# 4 · Stress-Regime Bernoulli: *"dependence through scenarios"*

**Intuition:** in some scenarios a shared stress state is active. In those scenarios the model multiplies every PD by a fixed factor, up to 100%.

**Answers:** *what happens when a defined stress state raises all PDs together?*

| Strengths | Weaknesses |
|---|---|
| Makes default dependence explicit and interpretable | Reality is not always a small number of discrete states |
| Easy to link losses to defined stress scenarios | Requires judgement on stress severity and likelihood |

**Use it for:** stress testing, limit discussions and adverse-scenario analysis.

> One shared stress state links all counterparties. The state also raises expected loss.

---

# 5 · Gaussian Copula: *the standard desk model*

**Intuition:** each counterparty has its own component and a shared component. When the shared factor is low, many counterparties weaken together and defaults cluster.

**Answers:** *how much do shared drivers reduce our diversification?*

| Strengths | Weaknesses |
|---|---|
| Standard market model (Vasicek, Basel ASRF) | One shared factor only |
| One correlation parameter (ρ) controls clustering | Gaussian tails understate joint extreme defaults |

**Use it for:** portfolio tail risk measurement and counterparty risk capital.

> The Student-t copula adds tail dependence to this model.

---

# 6 · Student-t Copula: *"extremes cluster"*

**Intuition:** the same shared factor as the Gaussian copula, plus a shared random scale. In extreme scenarios, counterparties are more likely to default together than under the Gaussian copula.

**Answers:** *how large can the loss become when defaults cluster in the tail?*

| Strengths | Weaknesses |
|---|---|
| Captures stronger tail dependence | One extra parameter (degrees of freedom, ν) needs a value |
| More realistic extreme-event behaviour | Tail loss can be larger than under the Gaussian copula at the same ρ. The effect depends on the portfolio and the confidence level. |

**Use it for:** extreme portfolio losses and stress scenarios where diversification falls.

> In extreme scenarios, diversification between counterparties falls.

---

# Comparing the Models: No Single Model Wins Everywhere

| Model | Realism | Transparency | Tail honesty | Data need | Business fit |
|---|---|---|---|---|---|
| Forced Default | n/a | High | n/a | Minimal | Stress bound |
| Independent Bernoulli | Low | High | Low | Low | Baseline view |
| Beta-Binomial | Med | High | Med | Low | Portfolio stress |
| Stress-Regime | Med | High | Med-High | Med | Scenario analysis |
| Gaussian Copula | High | Med | Med | Med-High | Core risk view |
| Student-t Copula | **High** | Med | **High** | Med-High | Extreme tail view |

> Simple models are easier to explain. Richer models capture more dependence. No single model is best in every column, so we run several.

---

# Which Model Answers Which Question?

| The question we need to answer | Model |
|---|---|
| What is the largest default loss for the given EAD and LGD? | Forced Default |
| What if defaults are independent? | Independent Bernoulli |
| What if the portfolio default rate is higher or lower than expected? | Beta-Binomial |
| What happens in a defined stress scenario? | Stress-Regime Bernoulli |
| What if counterparties are exposed to the same economic drivers? | Gaussian Copula |
| What if extreme events create unusually strong clustering? | Student-t Copula |

> Compare the models to see how much the dependence assumption changes the tail.

---

<!-- _class: lead -->

# Part 3 · Reading the Output

## *From loss distribution to risk decisions*

> Credit VaR also shows which counterparties drive the tail.

---

# The Metrics Every Model Returns

Every model returns the same metrics. The values differ most in the tail.

| Metric | Reads as | Feeds |
|---|---|---|
| **Expected Loss (EL)** | average counterparty default loss | expected credit cost |
| **Loss volatility (σ)** | how much the portfolio loss varies | unexpected loss |
| **VaR** | the loss that scenarios exceed with probability 1 − α, where α is the confidence level (α = 99% gives 1%) | risk-appetite limits |
| **Expected Shortfall (ES)** | the *average* loss in the scenarios beyond VaR | internal capital |
| **Unexpected Credit Loss (UL)** | two alternative measures: VaR − EL (VaR-based UL) and ES − EL (ES-based UL) | internal capital assessment |

> VaR is one point of the loss distribution. ES is the average of all losses beyond that point.

Models that keep each counterparty's PD give the same EL, apart from simulation error. VaR and ES can differ between models.

---

# Incremental VaR & ES: *"what does this counterparty add?"*

Hold the portfolio constant, remove one counterparty and re-measure the tail. The difference is its incremental risk.

> `incremental_risk(i) = risk(portfolio) − risk(portfolio excl. i)`

**Answers:** *how much does this counterparty add to the tail, and how much would the tail fall if we exit it?*

| Property | Consequence |
|---|---|
| Leave-one-out: exit one counterparty | Shows the tail reduction from an exit |
| Computed for both VaR and ES | Both come from the same simulated losses, so no new simulation runs |
| **Not additive** | Incremental values do not sum to portfolio risk |

**Use it for:** limit discussions, novation and unwind decisions for existing counterparties.

A new counterparty or trade needs a new run with the changed EAD.

---

# Contribution ES

Contribution ES splits today's ES across the counterparties. The contributions add up exactly to the portfolio ES.

> `contribution_ES(i)` = average loss of counterparty `i` in the portfolio's worst `1−α` share of scenarios.

> `portfolio_ES = sum_i contribution_ES(i)`

---

<!-- _class: lead -->

# RiskQuant · Credit VaR

> Credit VaR measures the tail of the credit loss distribution. Default dependence shapes that tail and can amplify it.



---

<!-- _class: lead -->

# Appendix: STX Credit VaR


---


# STX B.V. - GOO - Base Case

![width:1020px](images/credit_var_nl_goo_base.png)

---


# STX B.V. - GOO - Stressed (+20%)

![width:1020px](images/credit_var_nl_goo_stress_20pct_up.png)

---

# STX B.V. - GOO - Stressed (-20%)

![width:1020px](images/credit_var_nl_goo_stress_20pct_down.png)

---

# STX B.V. - GOO - Random Up/Down 10%-20% (PC Level)

![width:1020px](images/credit_var_nl_goo_stress_random.png)
