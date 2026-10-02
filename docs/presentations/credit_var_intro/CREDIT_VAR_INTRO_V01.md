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
---

<!-- _class: lead -->

# RiskQuant · Credit VaR

---

<!-- _class: lead -->

# Part 1 · Credit VaR

## *Measuring tail losses from counterparty default*

> Credit VaR measures potential losses in extreme scenarios and shows how much the result depends on assumptions about default dependence.

---
# What Credit VaR Measures

Credit VaR estimates the **distribution of potential losses** arising from counterparty defaults over a chosen risk horizon.

> **Expected Loss tells us what we expect. Credit VaR measures what we must survive.**

| Component | Question | Determines |
|---|---|---|
| **Exposure** | How much are we owed if a counterparty defaults? | The size of each loss |
| **Default probability** | How likely is each counterparty to default? | The average loss |
| **Default dependence** | How likely are counterparties to default together? | **The extreme losses (tail)** |

> **Exposure determines how much we lose. Default probability determines how often we lose. Default dependence determines how bad things become when defaults cluster.**

---

# Why Credit VaR Exists

**Expected Loss** measures the ***average counterparty credit loss***.

**Credit VaR** measures the ***tail losses that drive economic capital and risk appetite***.

| Expected Loss | Credit VaR |
|---|---|
| "What is our average credit loss?" | "What loss could we face in a severe event?" |
| Driven by exposure, PD and recovery | Driven by exposure, PD and **default dependence** |
| Used to estimate the expected cost of counterparty risk | Used to size capital against extreme losses |

> The key uncertainty is not only whether counterparties default — it is whether they *default together*.

---

# The Business Questions Behind the Tail

- **How much** could we lose in a severe counterparty event?
- **Which counterparties** contribute most to that tail?
- **Where are we concentrated** (e.g., sector, region, etc.)?
- **How much diversification do we *actually* have** versus assume?
- **How resilient are we** if several counterparties fail together?

> Credit risk is difficult not because defaults are mysterious, but because they can *occur together*. Every model in this deck represents a different view of that dependence.

---

# Where it Sits in the Risk Stack

- **CCR models** *size* exposure.
- **Credit VaR** turns exposure + default + dependence into a *loss distribution*.
- Credit risk comes from the possibility that an OTC contract has positive value when a counterparty defaults.

```text
   Trades
     │
     ▼
   CCR Exposure Engine        valuation · netting · collateral · future exposure
     │
     ▼
   Counterparty Exposure  ──►  potential loss if they default (EAD)
     │
     ▼
   Credit VaR Engine           default simulation · dependence · aggregation
     │
     ▼
   Loss Distribution      ──►  EL · VaR · ES · risk capital
```

---


<!-- _class: lead -->

# Part 2 · Jump-to-Default

## *Turning counterparty defaults into a portfolio loss distribution*

> Defaults are discrete events.
> Risk is the distribution of what those events can cost us.

---

# Jump-to-Default (In a Nutshell)

**The event:** a counterparty is healthy today, defaults tomorrow, and our positive exposure to them is lost instantly; the "jump."

We don't lend principal. Our credit risk is the **OTC book**: firms owing us mark-to-market on unrealised trades.

**The loop, per scenario:**

1. Exposure at default → 2. Does the name default? → 3. Apply recovery → 4. Sum across the book → 5. Repeat millions of times → **loss distribution**

> The whole modelling choice reduces to step 2×4: **when one name defaults, how likely is the next?**

---

# The One Choice That Moves Capital

| Set by marginals (PD · exposure · recovery) | Set by dependence |
|---|---|
| **Expected Loss** | **The tail: VaR & ES** |
| Barely moves across models | Moves *enormously* across models |

Example; €100m exposure, 40% recovery → **€60m loss if it defaults.**

The marginal models agree on how often that costs us on average. They violently disagree on how often it happens *alongside twenty others.*

> Get the marginals right for pricing. Get the **dependence** right for survival.

---

# The Model Staircase: Eight Models, One Question

Each rung removes one simplification about ***defaults happening together.***

| # | Model | What it adds |
|---|---|---|
| 1 | Forced Default | A worst-case ceiling |
| 2 | Independent Bernoulli | Probability; the diversified baseline |
| 3 | Beta-Binomial | A shared credit cycle |
| 4 | Stress-Regime Bernoulli | An explicit crisis state |
| 5 | Gaussian Copula | A common economy (industry standard) |
| 6 | Basel Correlation | Regulatory alignment |
| 7 | Student-t Copula | Crisis contagion; fat tails |
| 8 | Multi-Factor Gaussian | Real sector & geographic concentration |

> Same book. Same average loss. The tail fattens all the way up.

---

# 1 · Forced Default: *"assume everything fails all at once"*

**Intuition:** every counterparty defaults at once. No probability, no diversification.

**Answers:** *what is the absolute most we could lose?*

| Strengths | Weaknesses |
|---|---|
| Needs almost no data | Not a real scenario |
| A hard, un-arguable ceiling | Useless as a capital number |

**Use it for:** a bounding sanity check, and brand-new books with no history.

> The ceiling every other model must sit beneath.

---

# 2 · Independent Bernoulli; *"unrelated coin flips"*

**Intuition:** each counterparty is a separate weighted coin flip. One failing tells you nothing about the next.

**Answers:** *what does the loss look like if defaults are unrelated?*

| Strengths | Weaknesses |
|---|---|
| Simple, fast, full distribution | Forbids clustering entirely |
| Captures single-name concentration | **Understates the tail; flatters diversification** |

**Use it for:** granular, homogeneous books; the baseline everything else is measured against.

> Its diversification story is too good to be true; because counterparties share an economy.

---

# 3 · Beta-Binomial; *"good years and bad years"*

**Intuition:** the default rate for the whole book is itself uncertain. In a bad year, *everyone's* odds rise together.

**Answers:** *what happens as the credit environment deteriorates?*

| Strengths | Weaknesses |
|---|---|
| Captures the cycle with one dial | Treats the book as one pool |
| Honestly fatter tail than independence | No sector / geography detail |

**Use it for:** homogeneous pools and through-the-cycle provisioning.

> First model to admit co-movement; cheaply, with a single intuitive parameter.

---

# 4 · Stress-Regime Bernoulli; *"calm, or storm"*

**Intuition:** two states of the world. Most of the time it's calm; occasionally a crisis hits and every default probability jumps.

**Answers:** *what does a severe-but-plausible stress do to us?*

| Strengths | Weaknesses |
|---|---|
| The recession is explicit & nameable | Reality isn't a clean on/off switch |
| Ideal for ICAAP / scenario narratives | Needs judgement on odds & severity |

**Use it for:** stress testing and board-facing crisis storytelling.

> Twin-peaked losses: a benign hump, and a distinct, alarming storm hump.

---

# 5 · Gaussian Copula; *the industry standard*

**Intuition:** each name is part idiosyncratic, part tied to a **shared economy.** When the economy dips, names weaken together and defaults cluster.

**Answers:** *how much diversification do we really have once counterparties share drivers?*

| Strengths | Weaknesses |
|---|---|
| Market & regulatory standard | One economy; single factor |
| Realistic clustering & diversification | Bell-curve tails understate joint crashes |

**Use it for:** economic capital on the bilateral book; everyday CCR.

> The workhorse; and the base the last two rungs improve on.

---

<!-- # 6 · Basel Correlation; *the regulator's version*

**Intuition:** the same Gaussian machinery, but correlations are **prescribed by formula**, not estimated from our book.

**Answers:** *how does our internal view compare to the regulatory benchmark?*

| Strengths | Weaknesses |
|---|---|
| Regulatory alignment & comparability | One-size correlations ignore our mix |
| Nothing to estimate or defend | **Blind to concentration by construction** |

**Use it for:** capital reconciliation and cross-institution benchmarking.

> A reconciliation tool, not a risk model; run it *beside* the internal one to explain the gap.

--- -->

# 6 · Student-t Copula; *"crises are contagious"*

**Intuition:** same shared economy, but when it turns bad it turns bad for *everyone at once*; harder than a bell curve allows.

**Answers:** *how bad is the extreme, everything-fails-together scenario?*

| Strengths | Weaknesses |
|---|---|
| Captures tail dependence / contagion | One more parameter to calibrate |
| Realistic crisis behaviour | Heavier compute; looks conservative in calm times |

**Use it for:** severe stress, wrong-way risk, the CRO's tail view.

> 2008 was the year diversification vanished. This is the model that expects it to.

---

<!-- # 8 · Multi-Factor Gaussian; *"many economies"*

**Intuition:** not one economy but several; energy, rates, banks, regions. Each name loads onto the ones that actually move it.

**Answers:** *where is our true concentration; and true diversification?*

| Strengths | Weaknesses |
|---|---|
| Sees sector & geographic concentration | Data-hungry: needs factor loadings |
| Rewards *genuine* diversification | Most calibration & governance |

**Use it for:** counterparty limits, concentration monitoring, portfolio steering.

> The single-factor model can't tell an oil shock from a tech shock. This one can.

--- -->

# Comparison; No Row Wins Every Column

| Model | Realism | Transparency | Tail honesty | Data need | Regulatory fit |
|---|---|---|---|---|---|
| Forced Default | – | High | – | Minimal | Low |
| Independent Bernoulli | Low | High | Low | Low | Low |
| Beta-Binomial | Med | High | Med | Low | Med |
| Stress-Regime | Med | High | Med-High | Med | Med-High |
| Gaussian Copula | High | Med | Med | Med-High | **High** |
| Student-t Copula | **High** | Med | **High** | Med-High | Med-High |
<!-- | Basel Correlation | Med | High | Med | Med | **High** | -->
<!-- | Multi-Factor Gaussian | **High** | Low-Med | High | **High** | Med-High | -->

> Transparency and realism pull in opposite directions. The clearest models understate the tail; the honest ones are hardest to explain. That is why you need a **suite**.

---

# Which Model for Which Decision

| The decision in front of you | The rung |
|---|---|
| Bound the absolute worst case | Forced Default |
| Fast view on a granular book | Independent Bernoulli |
| Provision through the cycle | Beta-Binomial |
| Tell the crisis story for ICAAP | Stress-Regime Bernoulli |
| Economic capital on the bilateral book | Gaussian Copula |
| Protect against contagion | Student-t Copula |
<!-- | Reconcile to regulatory capital | Basel Correlation |
| Set counterparty & sector limits | Multi-Factor Gaussian | -->

> The sophisticated move isn't picking one. It's running several and reading the *spread between them as model risk*; quantified, not hidden.

---

<!-- _class: lead -->

# Part 3 · Reading the Output
## *What the engine hands you; and how to act on it*

> EL | VaR | ES
> Different models produce different tails.
> Understanding why they disagree is as important as the loss number itself.

---

# The Metrics Every Model Returns

The same battery on every model's loss distribution. Models agree near the middle and separate in the tail.

| Metric | Reads as | Feeds |
|---|---|---|
| **Expected Loss** | the average; cost of doing business | pricing · provisions |
| **Loss volatility (σ)** | how dispersed outcomes are | first tail signal |
| **VaR** | the loss we won't breach *x%* of the time | risk-appetite limits |
| **Expected Shortfall (ES)** | the *average* loss once we're past VaR | economic capital |

> VaR is a *line*; a single point on the curve. ES is the *territory beyond it*; the average of everything worse.

Reported side-by-side per model: EL generally stable; VaR and ES fan out with the dependence assumption.

---

# Incremental VaR & ES; *"what does this name add?"*

Hold the portfolio, remove one counterparty, re-measure the tail. The gap is its *incremental risk*.

> `incremental_risk(i) = risk(portfolio) − risk(portfolio excl. i)`

**Answers:** does this counterparty *add* to our tail or diversify it? What's the tail cost of the next trade?

| Property | Consequence |
|---|---|
| Directional; drop-one / add-one | The **new-trade & limit-headroom** metric |
| Defined for both VaR and ES | Ordinary re-run difference; well-conditioned |
| **Not additive** | Won't sum to the total; it double-counts shared diversification |

**Use it for:** pre-deal pricing, limit checks, novate / unwind calls on a single name.

---

# Contribution ES

A different question from incremental: not "what if we drop it," but *how is today's tail split across the book*; an additive allocation that sums **exactly** to portfolio ES.

> `contribution_ES(i)` = average loss of `i` across the worst `1−α` scenarios.

> `portfolio_ES = sum_i contribution_ES(i)`

---

<!-- _class: lead -->

# The End


<!-- # Why There's No Contribution VaR

**Why ES decomposes cleanly and VaR doesn't:**

| Contribution ES | Contribution VaR |
|---|---|
| Averages over the **whole tail** {L ≥ VaR}; thousands of scenarios | Needs each name's loss *where* {L = VaR}; a **knife-edge** point |
| Stable even on lumpy jump-to-default losses | Almost no scenario lands there → the split is **noise** |
| ES is coherent → allocation is additive & sensible | VaR isn't subadditive → contributions can turn **perverse / negative** |

> You can't average what happens *on* the line; you can average the territory beyond it, and split the bill by counterparty. **That split is our concentration- and limit-setting engine.**

--- -->

<!-- # From Metric to Decision

**Capital & limits**
Economic capital *is* the tail. Size counterparty and sector limits on **marginal tail contribution**, not gross notional or current exposure.

**Pricing; CVA / XVA**
Charge each trade for the counterparty capital it consumes. Surface **wrong-way risk**: exposure that rises exactly as the name weakens.

**Strategy**
Does novating to a CCP diversify our tail or concentrate it? Which loan sale or unwind sheds the *most* tail per euro? Which counterparty cluster is our reverse-stress break point?

> One engine feeds pricing, limits, capital and stress; so every number reconciles.

--- -->

<!-- # The Case in One Number

On an **identical book**; same counterparties, exposures, PDs, LGDs, and Expected Loss:

> Expected Loss barely moves across every model.
> The **99% loss spans roughly 50%**; Independent Bernoulli at the bottom, Student-t copula at the top.

That entire spread is the **price of our dependence assumption**; and in most firms it is invisible and undocumented.

### Recommendation

Adopt CreditVaR as the firm's **counterparty-risk platform.** Start on today's data, climb the staircase deliberately, and run models **in parallel** so the disagreement between them becomes a governed, reported measure of model risk; not a hidden bet. -->
