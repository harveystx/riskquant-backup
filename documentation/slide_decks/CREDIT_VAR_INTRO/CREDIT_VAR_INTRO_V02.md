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

## *Measuring the losses that matter when counterparties fail together*

---

<!-- _class: lead -->

# Part 1 · Credit VaR

## *When one default becomes many*

> Credit VaR measures how assumptions about default dependence change the losses we must withstand.
---

# What Credit VaR Measures

Credit VaR answers one question: *if OTC counterparties default, what losses could our trading portfolio experience?*

> **Expected Loss tells us what we expect. Credit VaR measures what we must survive.**

| Component | Question | Determines |
|---|---|---|
| **Exposure** | How much value is at risk if a counterparty defaults? | The size of each loss |
| **Default probability** | How likely is each counterparty to default? | The average loss |
| **Default dependence** | How likely are counterparties to default together? | **The extreme losses (tail)** |

> **Exposure determines the size of the loss. Default probability determines the frequency of loss. *Default dependence determines the severity of the tail.***
---

# Why Credit VaR Exists

**Expected Loss** measures the ***average counterparty credit loss***.

**Credit VaR** measures the ***tail losses that drive internal capital and risk appetite***.

| Expected Loss | Credit VaR |
|---|---|
| "What is our average credit loss?" | "What loss could we face in a severe event?" |
| Driven by exposure, PD and recovery | Driven by exposure, PD and **default dependence** |
| Measures the average impact of counterparty defaults | Measures the extreme outcomes that drive risk appetite and capital |

> The key uncertainty is not only whether counterparties default, but whether they *default together*.

---

# The Business Questions Behind the Tail

- **How much** could we lose in a severe counterparty event?
- **Which counterparties** contribute most to that tail?
- **Where are we concentrated** by counterparty, sector, region, or factor?
- **How much diversification do we *actually* have** versus assume?
- **How resilient are we** if several counterparties fail together?

> Credit risk is difficult not because defaults are mysterious, but because they can *occur together*. Every model in this deck represents a different view of that dependence.

---

# Where it Sits in the Risk Stack

- **CCR models** *size* exposure.
- **Credit VaR** turns exposure + default + dependence into a *loss distribution*.
- Credit risk comes from the possibility that a derivative has positive replacement value when the counterparty defaults.

```text
   Trades
     │
     ▼
   CCR Exposure Engine        valuation · netting · collateral · future exposure
     │
     ▼
   Counterparty Exposure  ──►  value at risk if they default
     │
     ▼
   Credit VaR Engine           default simulation · dependence · aggregation
     │
     ▼
   Loss Distribution      ──►  EL · VaR · ES · risk capital
```

---


<!-- _class: lead -->

# Part 2 · Jump-to-Default (JTD)

## *From individual defaults to portfolio tail losses*

> Defaults happen one name at a time.
> Credit risk emerges when they happen together.

---

# Jump-to-Default (In a Nutshell)

**The event:** a counterparty defaults, and the positive value of our OTC portfolio becomes a credit loss.

We don't lend principal. Our credit risk is the **OTC book**: firms owing us mark-to-market on unrealised trades.

**The loop, per scenario:**

1. Exposure at default → 2. Does the name default? → 3. Apply recovery → 4. Sum across the book → 5. Repeat millions of times → **loss distribution**

> The whole modelling choice reduces to step 2×4: **when one name defaults, how likely is the next?**

---

# The One Choice That Moves Capital

| Driven by individual assumptions | Driven by dependence assumptions |
|---|---|
| **Expected Loss** | **The tail: VaR & ES** |
| Barely moves across models | Moves *enormously* across models |

Example: €100m exposure, 40% recovery → **€60m loss if it defaults.**

The marginal models agree on how often that costs us on average. They violently disagree on how often it happens *alongside twenty others.*

> Get the exposure and default assumptions right for average losses. Get the **dependence** right for tail risk.

---
# The Same Portfolio. Different Views of the Tail.

Each model answers the same question: **When one counterparty defaults, how much does it change the likelihood that others default too?**

| # | Model | Assumption added |
|---|---|---|
| 1 | Forced Default | Everything fails together |
| 2 | Independent Bernoulli | Defaults are unrelated |
| 3 | Beta-Binomial | Default rates can move higher or lower together |
| 4 | Stress-Regime Bernoulli | The market can enter a crisis state |
| 5 | Gaussian Copula | Counterparties share common risk drivers |
| 6 | Student-t Copula | Extreme tail dependence |

> Same portfolio. Same Expected Loss. The difference is the tail.

---

# 1 · Forced Default: *"assume everything fails all at once"*

**Intuition:** every counterparty defaults at once. No probability, no diversification.

**Answers:** *what is the absolute most we could lose?*

| Strengths | Weaknesses |
|---|---|
| Needs almost no data | Not a real scenario |
| A hard, un-arguable ceiling | Not suitable as a realistic capital estimate |

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

> Its diversification benefit is overstated because counterparties share common drivers.

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

**Use it for:** portfolio tail risk measurement and counterparty risk capital.

> The workhorse; and the base the last two rungs improve on.

---

# 6 · Student-t Copula; *"crises are contagious"*

**Intuition:** same shared economy, but when it turns bad it turns bad for *everyone at once*; harder than a bell curve allows.

**Answers:** *how bad is the extreme, everything-fails-together scenario?*

| Strengths | Weaknesses |
|---|---|
| Captures tail dependence / contagion | One more parameter to calibrate |
| Realistic crisis behaviour | Heavier compute; looks conservative in calm times |

**Use it for:** severe stress, wrong-way risk, the CRO's tail view.

> Crisis periods reveal that diversification can weaken exactly when it matters most (e.g., 2008).

---

# Comparing the Models: No Single Model Wins Everywhere

| Model | Realism | Transparency | Tail honesty | Data need | Business fit |
|---|---|---|---|---|---|
| Forced Default | – | High | – | Minimal | Stress bound |
| Independent Bernoulli | Low | High | Low | Low | Baseline view |
| Beta-Binomial | Med | High | Med | Low | Portfolio stress |
| Stress-Regime | Med | High | Med-High | Med | Scenario analysis |
| Gaussian Copula | High | Med | Med | Med-High | Core risk view |
| Student-t Copula | **High** | Med | **High** | Med-High | Extreme tail view |

> Transparency and realism pull in opposite directions. Simpler models are easier to explain; richer models capture more complex dependence. That is why you need a **suite**.

---

# Which Model Answers Which Question?

| The question we need to answer | Model |
|---|---|
| What is the absolute loss ceiling? | Forced Default |
| What if defaults are independent? | Independent Bernoulli |
| What if the portfolio default rate is higher or lower than expected? | Beta-Binomial |
| What happens in a defined stress scenario? | Stress-Regime Bernoulli |
| What if counterparties are exposed to the same economic drivers? | Gaussian Copula |
| What if extreme events create unusually strong clustering? | Student-t Copula |

> The objective is not to find the "right" model. It is to understand how much of the tail is created by our assumptions about dependence.

---

<!-- _class: lead -->

# Part 3 · Reading the Output

## *From loss distribution to risk decisions*

> The loss number is only the beginning. The real value is understanding what drives the tail and where risk comes from.

---

# The Metrics Every Model Returns

The same battery on every model's loss distribution. Models agree near the middle and separate in the tail.

| Metric | Reads as | Feeds |
|---|---|---|
| **Expected Loss (EL)** | average counterparty default loss | expected credit cost |
| **Loss volatility (σ)** | how dispersed outcomes are | first tail signal |
| **VaR** | the loss we won't breach *x%* of the time | risk-appetite limits |
| **Expected Shortfall (ES)** | the *average* loss once we're past VaR | internal capital |

> VaR is a *line*; a single point on the curve. ES is the *territory beyond it*; the average of everything worse.

Across models, Expected Loss (EL) stays almost unchanged.

The difference appears in the tail.

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


# RiskQuant · Credit VaR

> Defaults create losses. *Dependence creates tail risk.*
