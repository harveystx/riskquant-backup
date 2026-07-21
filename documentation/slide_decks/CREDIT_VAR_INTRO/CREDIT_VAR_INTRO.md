---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  /* ── RESET & BASE ─────────────────────────── */
  section {
    background-color: #181A1F;
    color: #B0B3B8;
    font-size: 24px;
    padding: 70px 90px;
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
  }

  /* ── HEADINGS ────────────────────────────── */
  h1 {
    color: #FFFFFF;
    font-size: 42px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 12px;
  }
  h2 {
    color: #FFFFFF;
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 24px;
    padding-bottom: 0;
    border-bottom: none;
  }
  h3 { color: #CCCCCC; font-size: 22px; font-weight: 500; }

  /* ── BODY ────────────────────────────────── */
  p, li { line-height: 1.6; color: #B0B3B8; }
  strong { color: #FFFFFF; font-weight: 600; }
  em { color: #E30076; font-style: normal; }

  /* ── TABLES ──────────────────────────────── */
  table {
    width: 100%;
    font-size: 18px;
    color: #E8E8E8;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 8px;
    overflow: hidden;
  }
  thead th {
    background: #E30076;
    color: #FFFFFF;
    font-weight: 600;
    font-size: 16px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 14px 16px;
    border: none;
  }
  tbody td {
    background: rgba(255, 255, 255, 0.03);
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover td { background: rgba(227, 0, 118, 0.08); }

  /* ── CALLOUT BOX ─────────────────────────── */
  blockquote {
    border-left: none;
    margin: 28px 0;
    padding: 20px 28px;
    background: #FFFFFF;
    border-radius: 8px;
    color: #181A1F;
    font-size: 20px;
    font-style: normal;
    position: relative;
  }
  /* Small magenta top accent */
  blockquote::before {
    content: '';
    position: absolute;
    top: 0;
    left: 28px;
    width: 40px;
    height: 2px;
    background: #FFFFFF;
    border-radius: 1px;
  }

  /* ── CODE ────────────────────────────────── */
  code {
    background: #252830;
    color: #E30076;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 19px;
    font-family: 'Fira Code', 'Consolas', monospace;
  }

  /* ── LINKS ───────────────────────────────── */
  a { color: #E30076; text-decoration: none; border-bottom: 1px solid transparent; }
  a:hover { border-bottom-color: #E30076; }

  /* ── LEAD SLIDE ──────────────────────────── */
  section.lead {
    text-align: left;
    padding-top: 160px;
  }
  section.lead h1 {
    font-size: 52px;
    line-height: 1.15;
    margin-bottom: 16px;
  }
  section.lead p {
    font-size: 26px;
    color: #888;
  }

  /* ── PAGE NUMBERS ────────────────────────── */
  section::after {
    color: rgba(255, 255, 255, 0.2);
    font-size: 14px;
    right: 40px;
    bottom: 30px;
  }

  /* ── ACCENT BAR (bottom of each slide) ───── */
  section::before {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 60px;
    height: 3px;
    background: #E30076;
  }
---

<!-- _class: lead -->

# RiskQuant | Credit VaR Solution Suite

**A scalable platform for measuring how much we could really lose — and why the answer depends on the model you choose.**

Prepared for: Head of Risk · Head of Credit Risk · CRO direct reports · Senior Portfolio Managers · Credit Committee

<!--
Presenter notes:

Open by framing this as a business conversation, not a modelling seminar. The audience already runs the bank's credit risk; they know PD, LGD, EAD, Expected Loss and Basel capital cold. What they do not yet have is a single, coherent platform that tells them how those inputs translate into the *shape of the loss distribution* — and how that shape changes the strategic answers they give the board.

Set the tone with the one sentence that runs through every slide: expected loss is a planning number, but the bank is not managed on averages — it is protected against the tail. This deck is about owning the tail deliberately rather than by accident.

Flag the punchline early so the audience knows where we are heading: on the very same portfolio, the average loss barely moves across every model we will show — but the 1-in-100 loss swings by roughly fifty percent depending on the correlation assumption. That single fact is the entire business case.

Suggested visual: full-bleed dark navy title with a faint loss-distribution curve watermark, the tail shaded gold.

Transition: "Let's start with the questions this suite is built to answer — the ones that land on your desk, not the modelling team's."
-->

---

## What is Credit Value-at-Risk (Credit VaR)?

> How much could we lose from credit defaults over a given time horizon, and how much capital should we hold against those losses?


---

## The Questions That Land On Your Desk

The Credit Committee and the board don't ask about probability distributions. They ask:

- **How much could we lose** in a genuinely bad year — not an average one?
- **How likely** is a loss large enough to threaten capital?
- **Which portfolios** are quietly consuming the most economic capital?
- **Where are we concentrated** — by name, sector, or geography?
- **How much diversification** do we *actually* have, versus how much we assume?
- **What happens in a recession**, when borrowers fail *together*?

> Every one of these is a question about the **tail** of the loss distribution — not the average.

<!--
Presenter notes:

This slide reframes the whole discussion in the audience's own language. None of these questions are answered by Expected Loss. They are all questions about severity, likelihood of severity, and co-movement — the tail.

Walk through them as a set of decisions rather than metrics. "How much could we lose" sets economic capital and risk appetite. "Which portfolios consume capital" drives limit allocation and RAROC. "Where are we concentrated" and "how much diversification do we really have" are the same question asked twice — and the honest answer is almost always "less than the desk thinks," because diversification benefit is a correlation assumption, not a fact. "What happens in a recession" is the stress-testing and ICAAP mandate.

Make the point that today these questions are often answered with a patchwork: a regulatory capital number here, a spreadsheet stress there, a gut-feel concentration view somewhere else. The CreditVaR suite answers all of them from one consistent engine, so the numbers reconcile and the story holds together in front of the regulator and the board.

Suggested visual: six question-cards arranged in a grid, each tagged with the business decision it feeds (Economic Capital, Risk Appetite, Limits, Concentration, Diversification, Stress).

Transition: "The reason these questions are hard is that our default headline metric — Expected Loss — is structurally silent on all of them."
-->

---

## Why Expected Loss Is Not Enough

Expected Loss tells us the **cost of doing business** — the average we should price for and provision against.

But Expected Loss is a **single number**. It says nothing about:

| Expected Loss answers | Expected Loss cannot answer |
|---|---|
| What do we lose *on average*? | What do we lose in a *bad* year? |
| What should we *provision*? | How much *capital* do we need? |
| What's the *cost* of the book? | How *volatile* is that cost? |

> A bank never fails because of its **average** loss. It fails because of the year the losses **cluster**.

Capital exists to absorb the gap between the average and the extreme. **That gap is what CreditVaR measures.**

<!--
Presenter notes:

This is the pivot of Part 1. Expected Loss is necessary and the audience uses it every day — for pricing, for provisioning, for the P&L cost of credit. Nothing here diminishes it. The argument is that EL is the *centre* of the distribution and capital lives in the *tail*, and the two are governed by completely different drivers.

Draw the distinction sharply: EL is driven by the marginal PD, LGD and EAD of each name. The tail is driven by how those defaults *co-move*. You can hold every marginal input fixed — same PDs, same LGDs, same exposures, same average loss to the penny — and still change the capital requirement dramatically by changing only the correlation assumption. We will show exactly this empirically in a few slides.

The banking-crisis framing lands well with this audience: no institution has ever been brought down by realising its expected loss. They fail when a common shock makes many borrowers default in the same quarter, and the realised loss lands far out in the tail that EL never described.

Suggested visual: a single loss-distribution curve. Mark the mean (EL) near the peak with a small label, then shade the far right tail gold and label it "Unexpected Loss = Economic Capital." The visual should make it obvious the mean sits nowhere near the tail.

Transition: "If the tail is driven by co-movement, then measuring it well depends entirely on the information we feed the models."
-->

---

<!-- _class: lead -->

# Part 2
## The Information That Powers the Models

*Richer portfolio information unlocks richer — and more truthful — portfolio models.*

<!--
Presenter notes:

Short divider slide. Signal the shift from "why" to "what we need." Keep it to one line of spoken framing: the sophistication of the answer is capped by the richness of the input. A model cannot see a concentration you never told it about.

Transition: "Let's look at what each layer of data buys us — in business terms, not data-engineering terms."
-->

---

## Data Is the Ceiling on Insight

Each layer of information unlocks a more truthful view of risk:

- **PD · LGD · EAD** — the foundation. Enough to compute Expected Loss and a naïve loss distribution.
- **Rating** — lets us separate investment-grade from high-yield behaviour, and stress them differently.
- **Sector & Geography** — the raw material of concentration. Without them, the model *cannot see* that half the book is energy.
- **Obligor Hierarchy** — parents, subsidiaries and guarantors that default *as one*. Missing this hides true single-name risk.
- **Correlations & Systematic Factors** — how borrowers are tied to the *same economy*. This is what turns a list of loans into a portfolio.

> The models don't get smarter on their own. **Better data is what makes a smarter model possible.**

<!--
Presenter notes:

Frame data maturity as an investment ladder that runs in parallel to the modelling ladder we are about to present. Each rung of data unlocks the next rung of model. You cannot run a multi-factor concentration model if you cannot tag obligors to sectors and regions — so the data conversation and the model conversation are the same conversation.

Emphasise obligor hierarchy specifically, because it is the one most often under-appreciated by senior audiences. If a parent and three subsidiaries are modelled as four independent names, the engine will report false diversification and understate a genuinely large single-name exposure. Getting the hierarchy right is often a bigger tail correction than upgrading the correlation model.

Make the systematic-factor point in plain language: correlation is simply the statement that our borrowers share an economy. When that economy turns, they turn together. Everything sophisticated in Part 3 is just increasingly honest ways of representing that shared economy.

Position this slide as governance-relevant: the bank should know, for each portfolio, which rung of data it currently stands on, because that determines which models it is *entitled* to trust.

Suggested visual: an ascending staircase, each step a data layer (PD/LGD/EAD → Rating → Sector/Geo → Hierarchy → Correlations), with a parallel faint staircase of models climbing alongside it.

Transition: "With the data laid out, we can walk up the modelling staircase — where each model exists to fix a specific blind spot in the one before it."
-->

---

<!-- _class: lead -->

# Part 3
## The Modelling Staircase

*Eight models. Each one solves a limitation of the last.*

<!--
Presenter notes:

Set expectations for the next eight slides. This is not eight competing products; it is one staircase. Every model earns its place by fixing a specific, nameable weakness in its predecessor. We will climb from a crude certainty at the bottom to a realistic, multi-economy view at the top.

Tell the audience what to listen for on each slide: the *business question it answers* and *who should use it*. They do not need the mathematics — they need to know which rung is appropriate for which decision.

Transition: "We start at the bottom — with the bluntest possible answer."
-->

---

## The Staircase at a Glance

| # | Model | The blind spot it fixes |
|---|---|---|
| 1 | **Forced Default** | *(the starting point)* — a worst-case ceiling |
| 2 | **Independent Bernoulli** | Adds probability: not everyone defaults |
| 3 | **Beta-Binomial** | Adds a shared cycle: bad years hit everyone |
| 4 | **Stress-Regime Bernoulli** | Makes the recession *explicit* |
| 5 | **Gaussian Copula** | Ties each name to a common economy |
| 6 | **Basel Correlation** | Aligns the copula to the regulator's formula |
| 7 | **Student-t Copula** | Lets crises be *contagious* — fat tails |
| 8 | **Multi-Factor Gaussian** | Many economies: real sector & geographic concentration |

> The average loss barely changes as you climb. **The tail changes everything.**

<!--
Presenter notes:

This is the map for the whole of Part 3 — keep it on screen long enough for the audience to internalise the shape of the journey. Read the right-hand column aloud: each row is a limitation being retired.

Reinforce the recurring theme one more time before diving in: across all eight, on an identical book, expected loss is essentially constant. What moves — and moves a lot — is the tail. That is because these models mostly share the same marginal inputs and differ only in how they let defaults cluster. Correlation reshapes the tail; it does not move the mean.

Note that models 5 through 8 are the "portfolio" models — they explicitly represent co-movement — and that is where the real strategic value lives. Models 1 through 4 are stepping stones that build intuition and serve specific narrow purposes.

Suggested visual: the eight-rung staircase, colour-graded from grey (crude) at the bottom to deep navy (realistic) at the top, with a small loss curve beside each step whose tail visibly fattens as you ascend.

Transition: "Rung one: the fire drill that assumes the whole building burns."
-->

---

## 1 · Forced Default

**Business intuition:** The fire drill that assumes *the entire building burns down at once.*

**Core assumption:** Every obligor defaults, with certainty. Probability is ignored entirely.

**Answers:** *What is the absolute maximum we could lose?* — the gross exposure at risk.

| Strengths | Weaknesses |
|---|---|
| Needs almost no data | Wildly over-conservative |
| Fully transparent — no black box | No sense of *likelihood* |
| A hard upper bound / sanity check | Useless as a capital number |

**Typical users:** New-portfolio owners with no history; anyone needing a bounding number.
**Typical portfolios:** Brand-new books; single large exposures under a worst-case lens.

**Suggested visual:** A single bar at the far right — "everything gone" — with the realistic loss curve dwarfed beside it.

<!--
Presenter notes:

Introduce this as the deliberate extreme — the "what if literally everything fails" bound. It requires no PD, no correlation, no history: just add up exposure times loss-given-default across the book. Its value is precisely that it needs nothing and cannot be argued with.

Be candid about why nobody holds capital to this number: it treats a 0.1% name and a 20% name identically, and assumes perfect simultaneous failure, which has never happened. But it is genuinely useful in two situations — a brand-new portfolio where you have no data to do anything smarter, and a sanity check that says "no matter what the fancy model claims, we can never lose more than this." In governance terms it is the ceiling on every other model's output.

Suggested visual described above.

Transition: "The first improvement is obvious — not everyone defaults. Let's introduce probability."
-->

---

## 2 · Independent Bernoulli

**Business intuition:** Every borrower is a separate **weighted coin flip** — heads it defaults, weighted by its PD. Flips are unrelated.

**Core assumption:** Defaults are **independent**. No borrower's failure makes another more likely.

**Answers:** *What is our loss distribution if defaults are unrelated?* — the diversified baseline.

| Strengths | Weaknesses |
|---|---|
| Simple, fast, intuitive | Assumes borrowers **never fail together** |
| Produces a full loss distribution | Systematically **understates the tail** |
| Captures *name* concentration | Diversification looks *too good to be true* |

**Typical users:** Retail and granular-book analysts.
**Typical portfolios:** Large, homogeneous retail / SME pools where independence is roughly plausible.

**Suggested visual:** A grid of coins mid-flip; a narrow, well-behaved loss curve with a thin tail.

<!--
Presenter notes:

This is the natural first "real" model and the baseline against which everything else is measured. Each name defaults independently with its own probability; sum the losses and you get a distribution. It already does something Forced Default cannot — it captures *name concentration*, because one enormous exposure dominates the tail even when flips are independent.

The critical weakness is the word "independent." In the real world borrowers share an economy; in a recession they default together. This model explicitly forbids that. As a result it produces the thinnest tail of any model and the most flattering diversification story — which is exactly why relying on it for capital is dangerous. It answers "what if defaults were unrelated," and the honest business answer is "but they're not."

It remains genuinely useful for very granular, homogeneous retail pools where the independence assumption is least wrong and where a fast, transparent number is worth more than a perfect one.

Transition: "So the next question is: what if the whole book has good years and bad years? What if the coin's bias itself moves?"
-->

---

## 3 · Beta-Binomial

**Business intuition:** The coin's bias is **itself uncertain.** Some years are simply bad years — and in a bad year, *everyone's* default rate rises together.

**Core assumption:** A single default rate is drawn for the whole pool each year. One dial, moving everyone at once.

**Answers:** *What if our portfolio's default rate swings from year to year?*

| Strengths | Weaknesses |
|---|---|
| Captures the credit **cycle** with one intuitive dial | One homogeneous pool only |
| Fatter, more honest tail than independence | No sector / geography detail |
| Still simple and fast | A single shared driver — coarse |

**Typical users:** Retail portfolio managers; provisioning and IFRS 9 teams.
**Typical portfolios:** Homogeneous pools — credit cards, mortgages — driven by one common cycle.

**Suggested visual:** Two loss curves overlaid — the thin independent curve vs. the visibly fatter Beta-Binomial curve.

<!--
Presenter notes:

This is the first model that admits co-movement, and it does so with a single, beautifully intuitive idea: the default rate for the whole pool is not fixed — it is drawn fresh each year. Most years it lands near the average; occasionally it lands high, and when it does, *every* borrower in the pool is more likely to default at once. That shared draw is correlation in disguise, achieved with one parameter and no need for sector or factor data.

For this audience the value is that it captures the credit cycle cheaply. It is a strong fit for homogeneous retail pools and for provisioning work, where a single common driver is a reasonable description of reality and the extra realism over independence directly fattens the tail in the right way.

Its limit is homogeneity: it treats the whole pool as one undifferentiated group, so it cannot tell you that your energy names and your tech names move differently. That differentiation is what the copula models add later.

Note the recurring theme in the visual: the mean is unchanged from the independent model, but the tail is already fatter. Same average, more capital.

Transition: "Beta-Binomial blends good and bad years smoothly. But executives often want the bad year named explicitly — the recession scenario."
-->

---

## 4 · Stress-Regime Bernoulli

**Business intuition:** Two weather states — **calm** and **storm.** Most of the time it's calm. But a storm can arrive, and in a storm every default rate jumps.

**Core assumption:** With some probability the whole book enters a **stress regime** that multiplies every PD.

**Answers:** *What happens to us specifically in a recession?*

| Strengths | Weaknesses |
|---|---|
| The recession scenario is **explicit and nameable** | Reality isn't a clean on/off switch |
| Ideal for stress narratives & ICAAP | Requires judgement on stress odds & severity |
| Transparent and easy to govern | Still limited cross-sectional structure |

**Typical users:** Stress-testing teams; ICAAP authors; the CRO's scenario function.
**Typical portfolios:** Cyclical, macro-sensitive books.

**Suggested visual:** A bimodal loss curve — a tall calm peak plus a distinct, alarming second hump in the tail (the storm).

<!--
Presenter notes:

This model reframes co-movement as a scenario rather than a smooth statistical draw, which is exactly how executives and regulators like to think. There are two states of the world — normal and stressed — and in the stressed state every borrower's default probability is scaled up by a stress multiplier. You choose how likely the storm is and how severe it is, and those two dials are directly interpretable and defensible in an ICAAP narrative.

Its great strength is governance and communication. When the board asks "what does a recession do to us," this model answers in the board's own language, and the assumptions are laid bare rather than buried in a correlation matrix. It produces a distinctive twin-peaked loss shape — a benign everyday hump and a separate, larger stress hump — that makes the tail viscerally obvious.

The trade-off is that reality is not a clean switch, and the answer is only as good as the judgement on stress probability and severity. It also still treats the book fairly coarsely. It is best used *alongside* the copula models — as the narrative scenario layer over a statistically richer engine.

Transition: "Regimes and shared draws are intuitive, but the market standard for portfolio credit risk works differently — it ties every single name to a common economy through one hidden driver."
-->

---

## 5 · Gaussian Copula — The Industry Standard

**Business intuition:** Every borrower's fortunes are **partly tied to the same economy.** When the economy dips, defaults *cluster* — not because borrowers know each other, but because they share conditions.

**Core assumption:** One hidden systematic factor drives all names; each name's sensitivity to it sets the correlation.

**Answers:** *Given a shared economy, what is our real diversification — and how do losses cluster?*

| Strengths | Weaknesses |
|---|---|
| The market & regulatory **standard** (CreditMetrics lineage) | A single economy — one factor only |
| Captures clustering and realistic diversification | Bell-curve tails **understate** joint crashes |
| Well understood by regulators & auditors | Correlation assumed constant through the cycle |

**Typical users:** Economic-capital teams; virtually every wholesale desk.
**Typical portfolios:** Diversified corporate / wholesale books.

**Suggested visual:** Many borrower "dials" all wired to one central "Economy" node; a loss curve with a clearly heavier tail than Beta-Binomial.

<!--
Presenter notes:

This is the workhorse of portfolio credit risk and the model the audience is most likely to already recognise from CreditMetrics and the Basel lineage. The mechanism, in plain terms: behind each borrower sits a hidden "health" variable that is part common-economy and part idiosyncratic. When the common economy has a bad draw, many borrowers' health drops together and defaults cluster. The strength of that shared component is the asset correlation.

For the business this is the first model that gives a credible, defensible economic-capital number on a diversified wholesale book. It captures genuine clustering, produces a realistic diversification benefit, and — crucially — it is the language regulators and auditors expect, which makes it the natural spine of the platform.

Its two honest limitations set up the final rungs. First, it uses a single systematic factor — one economy — so it cannot distinguish an energy shock from a tech shock. Second, its Gaussian (bell-curve) core means that in truly extreme scenarios it *under-predicts* how tightly defaults bunch together — real crises are more contagious than the bell curve allows. Those two gaps are precisely what Multi-Factor and Student-t fix.

Transition: "There are two directions to improve the standard copula. The first is to stop estimating correlation ourselves and adopt the regulator's own version."
-->

---

## 6 · Basel Correlation

**Business intuition:** The **regulator's standardised** version of the same economy-linkage — correlations set by a prescribed formula rather than estimated from our data.

**Core assumption:** The Gaussian copula, but with **Basel's prescribed asset correlations** (PD-dependent, by formula).

**Answers:** *How does our capital compare to the regulatory model — and can we reconcile the two?*

| Strengths | Weaknesses |
|---|---|
| **Regulatory alignment** and comparability | One-size-fits-all correlations ignore *our* mix |
| No correlation to estimate or defend | Assumes an infinitely granular book |
| Highly defensible with supervisors | **Blind to name & sector concentration** |

**Typical users:** Regulatory-capital & Pillar 1 reconciliation teams.
**Typical portfolios:** Regulatory reporting; cross-bank benchmarking.

**Suggested visual:** Two capital bars side by side — "Internal Economic Capital" vs. "Regulatory (Basel) Capital" — with the gap annotated.

<!--
Presenter notes:

Position this not as a better risk model but as a *reconciliation* tool. It is the same Gaussian copula machinery, but instead of estimating correlations from our own portfolio we plug in the correlations the Basel IRB formula prescribes. That buys two things: perfect comparability with the regulatory capital number, and a correlation input nobody can argue with because the supervisor wrote it.

The strategic use is bridging. When internal economic capital and regulatory capital diverge — and they always do — this model lets us decompose *why*. Is the gap driven by our correlation estimates, by concentration the regulator's formula ignores, or by portfolio mix? That decomposition is exactly the conversation the CRO wants to have with the board and the regulator.

Be clear about the built-in blindness: the regulatory formula assumes an infinitely fine-grained, perfectly diversified book, so by construction it *cannot see* single-name or sector concentration. That is a feature for comparability and a bug for real risk management — which is the strongest possible argument for also running the internal multi-factor model in parallel.

Transition: "The regulatory version trades realism for comparability. The other direction of travel is the opposite — sacrifice nothing on realism and make the tail more honest about contagion."
-->

---

## 7 · Student-t Copula

**Business intuition:** Same shared economy — but **crashes are contagious.** When conditions turn bad, they turn bad for *everyone at once*, more violently than a bell curve predicts.

**Core assumption:** A Gaussian-style copula with **fat tails** — extreme joint defaults are far more likely.

**Answers:** *How bad is the truly extreme, everything-fails-together scenario?*

| Strengths | Weaknesses |
|---|---|
| Captures **tail dependence** / contagion | One extra parameter to calibrate & defend |
| Far more realistic crisis behaviour | Heavier to compute |
| One dial (tail-fatness) tunes severity | Can look "too conservative" in calm times |

**Typical users:** Tail-risk & CRO functions; severe stress testing.
**Typical portfolios:** Concentrated or systemically linked books; crisis-sensitive exposures.

**Suggested visual:** Gaussian vs. Student-t loss curves overlaid — identical bodies, but the t-curve's tail extends visibly further right.

<!--
Presenter notes:

This is the single most important upgrade for anyone whose job is protecting the bank against the extreme. The Gaussian copula's weakness was that its bell-curve core lets correlations effectively relax in the extremes — it quietly assumes that even in a crisis, defaults don't *all* pile up together. History says otherwise: 2008 was precisely the event where diversification evaporated and everything defaulted at once.

The Student-t copula fixes exactly this with one intuitive dial — tail-fatness. It keeps the same body as the Gaussian model, so ordinary years look the same, but it makes the joint extreme dramatically more likely. This is our house's empirical headline: on an identical book, moving from Independent Bernoulli to the t-copula leaves expected loss essentially unchanged while the 99% credit VaR widens by roughly fifty percent. Same average, materially more capital — because the t-copula refuses to pretend crises are polite.

The cost is honest: one more parameter to calibrate and defend, more computation, and a number that will look conservative when times are calm — which is exactly when the board is least inclined to believe it and most needs to. Framing that tension is part of the CRO's job, and this model gives them the ammunition.

Transition: "Fat tails fix *how violently* names cluster. The final rung fixes *which* names cluster — by giving the portfolio more than one economy."
-->

---

## 8 · Multi-Factor Gaussian Copula

**Business intuition:** Not *one* economy but **many overlapping ones** — tech, energy, US, Europe. Each borrower loads onto several. This is where **real concentration** finally becomes visible.

**Core assumption:** Multiple systematic factors (sectors, regions); each name's loadings define its true correlations.

**Answers:** *Where is our real concentration and diversification — across sectors and geographies?*

| Strengths | Weaknesses |
|---|---|
| Sees **sector & geographic** concentration directly | Data-hungry — needs factor loadings |
| Rewards *genuine* cross-sector diversification | Calibration & governance complexity |
| The engine for **limit-setting** & optimisation | Most sophisticated to run and explain |

**Typical users:** Portfolio managers; concentration-risk & strategic-planning teams.
**Typical portfolios:** Large, diversified, multi-sector / multi-geography wholesale books.

**Suggested visual:** A web of borrowers wired to *several* economy nodes (Energy, Tech, US, EU); a heat-map of sector concentration lighting up the true hot-spot.

<!--
Presenter notes:

This is the top of the staircase and the most strategically valuable rung. The single-factor copula ties every name to one economy; reality has many. A US energy producer and a European software firm do not share a fate — but a single-factor model is forced to correlate them the same as any other pair, which simultaneously *overstates* diversification within a concentrated sector and *understates* it across genuinely unrelated ones.

The multi-factor model gives the portfolio the several economies it actually has. Each borrower loads onto a market factor plus its sector and region. Now the engine can *see* that eighty percent of the book moves with energy, and it will price that concentration into the tail — while correctly rewarding a book that is genuinely spread across uncorrelated sectors with real capital relief. This is the model that turns credit VaR from a reporting number into a *management* tool: it is what you run to set sector limits, to price concentration into loans, and to decide whether an acquisition adds diversification or doubles down on a hidden bet.

Be honest about the cost: it is data-hungry (you need reliable factor loadings), it demands the most calibration governance, and it is the hardest to explain to a non-technical board. That is precisely why it sits at the top of the staircase — it is the destination, not the starting point.

Transition: "Eight models, one staircase. The obvious executive question is: which one is *right*? Let's compare them head to head."
-->

---

<!-- _class: lead -->

# Part 4
## Comparing the Approaches

*There is no single "best" model — only the right model for the decision.*

<!--
Presenter notes:

Brief divider. Prime the audience for the key message of Part 4: this is a trade-off space, not a leaderboard. Simplicity and realism pull in opposite directions, and the right choice depends entirely on the decision being supported and the governance appetite.

Transition: "Here is the whole suite scored on the dimensions executives actually weigh."
-->

---

## Executive Comparison — The Whole Suite

| Model | Realism | Simplicity | Transparency | Compute | Data Need | Tail Honesty | Regulatory Fit |
|---|---|---|---|---|---|---|---|
| Forced Default | ▁ | ██ | ██ | ▁ | ▁ | n/a | ▁ |
| Independent Bernoulli | ▂ | ██ | ██ | ▂ | ▂ | ▂ | ▂ |
| Beta-Binomial | ▄ | █▊ | █▊ | ▂ | ▂ | ▄ | ▃ |
| Stress-Regime Bernoulli | ▄ | █▊ | ██ | ▂ | ▃ | ▅ | ▅ |
| Gaussian Copula | ▆ | ▄ | ▄ | ▅ | ▅ | ▅ | ██ |
| Basel Correlation | ▅ | ▅ | ▆ | ▅ | ▄ | ▄ | ██ |
| Student-t Copula | █▊ | ▃ | ▃ | ▆ | ▅ | ██ | ▆ |
| Multi-Factor Gaussian | ██ | ▂ | ▃ | ██ | ██ | ▆ | ▆ |

> Read it as a trade-off, not a ranking: **realism and transparency pull in opposite directions.** More bars = stronger on that dimension.

<!--
Presenter notes:

Walk the audience down the two dominant diagonals. Simplicity and transparency are highest at the top of the table and fall as you descend; realism, data need and compute do the reverse. That opposition is the entire governance dilemma in one picture: the models the board most easily understands are the ones that most understate the tail, and the models that capture the tail honestly are the hardest to explain and defend.

Draw out the specific sweet spots. The Gaussian and Basel rows are the regulatory-fit champions — that is where reconciliation and Pillar 1 work belongs. The Student-t row is the tail-honesty champion — that is the CRO's severe-stress and economic-capital anchor. The Multi-Factor row is the realism and concentration champion — that is the portfolio manager's limit-setting and optimisation engine. No single row wins every column, which is the whole point.

Use this slide to pre-empt the "just tell us which one" question. The answer is deliberately "it depends on the decision," and the platform's value is that it lets us run several and triangulate rather than betting the bank on one assumption.

Suggested visual: this heat-scored table, ideally with the cells colour-graded so the two opposing diagonals are visible at a glance.

Transition: "If no model wins outright, how does an executive choose? By starting from the decision, not the model."
-->

---

## Which Model, For Which Decision?

| The decision in front of you | The right rung |
|---|---|
| Bound the absolute worst case | **Forced Default** |
| Fast view on a granular retail pool | **Independent Bernoulli** |
| Provisioning through the cycle | **Beta-Binomial** |
| Tell the recession story for ICAAP | **Stress-Regime Bernoulli** |
| Economic capital on a wholesale book | **Gaussian Copula** |
| Reconcile to regulatory capital | **Basel Correlation** |
| Protect against crisis contagion | **Student-t Copula** |
| Set sector & geographic limits | **Multi-Factor Gaussian** |

> The suite's power is running **several at once** and reading the *spread* between them as a measure of model risk.

<!--
Presenter notes:

This is the practical decoder ring. It inverts the framing deliberately: don't start by asking "which model is best," start by naming the decision, and the appropriate rung follows. Reading the table aloud lets every person in the room locate their own mandate — the provisioning team lands on Beta-Binomial, the ICAAP author on Stress-Regime, the capital team on Gaussian and Basel, the CRO on Student-t, the portfolio manager on Multi-Factor.

Then deliver the higher-order insight: the most sophisticated use of the platform is not picking one model but running the relevant several and treating the *gap between their answers* as a quantified measure of model risk. If Independent Bernoulli and the t-copula disagree by fifty percent on capital, that gap is not noise — it is the price of the correlation assumption, made visible and auditable. A single-model shop can never see that; a platform shop manages it.

Transition: "So how does all of this translate into decisions the bank actually makes with money? That's Part 5."
-->

---

<!-- _class: lead -->

# Part 5
## From Metric to Decision

*A number no one acts on is overhead. Here's where CreditVaR earns its keep.*

<!--
Presenter notes:

Divider. Signal the shift from "how we measure" to "what we decide." The credibility of the whole investment rests on this part landing — the audience must see the metric flowing into real money decisions, not into a report that gathers dust.

Transition: "Start with the decisions closest to the balance sheet."
-->

---

## Capital, Appetite & Limits

**Setting the size of the bank's shock absorber:**
- **Economic Capital** — the tail loss *is* the capital number. CreditVaR sizes the buffer directly.
- **ICAAP** — a defensible, model-backed capital adequacy story for the supervisor.
- **Risk Appetite** — translate the board's tolerance into a hard loss threshold and monitor against it.

**Deciding who gets to consume it:**
- **Large-Exposure & Single-Name limits** — sized to *marginal* tail contribution, not gross notional.
- **Sector & Geographic limits** — set where the multi-factor model shows concentration *actually* bites.

> Capital and limits are the same decision viewed from two ends: *how big is the buffer,* and *who is allowed to spend it.*

<!--
Presenter notes:

These are the decisions closest to the board and the regulator, so lead with them. Economic capital is the most direct application: the unexpected-loss tail that CreditVaR produces is, definitionally, the economic capital number. That flows straight into ICAAP, where the value is a capital adequacy narrative that is model-backed and reconcilable rather than asserted.

Risk appetite is where the abstraction becomes operational: the board's qualitative tolerance ("we should survive a severe-but-plausible downturn") becomes a specific loss threshold at a specific confidence level, which the CreditVaR engine monitors continuously. That closes the loop between the boardroom statement and the daily risk report.

On limits, make the sharp point that gross notional is the wrong ruler. A large exposure that diversifies the book may deserve a *higher* limit than a smaller one that doubles a concentration. Limits should be set on *marginal contribution to the tail*, which only a portfolio model can compute — and the multi-factor model is what tells you which sectors and geographies are the real binding constraints rather than the ones that merely look big.

Transition: "Once capital and limits are set, the same engine sharpens every pricing and portfolio-shaping decision."
-->

---

## Pricing, Return & Portfolio Shaping

- **Loan Pricing** — charge for the *capital a deal consumes*, not just its expected loss. A concentrating loan should cost more.
- **RAROC** — put every deal and desk on a level, risk-adjusted playing field. Reward return *per unit of tail*, not per unit of notional.
- **Portfolio Optimisation** — actively steer origination toward exposures that *diversify* the tail.
- **Concentration Monitoring** — a live dashboard of where the book is quietly betting on one economy.

> The same engine that *measures* risk can *shape* the portfolio — turning credit VaR from a report into a steering wheel.

<!--
Presenter notes:

This slide is where the finance and origination side of the audience — the senior portfolio managers and credit committee members — should lean in. The unifying idea is that every deal has two costs: its expected loss, which the bank already prices, and its *contribution to the tail*, which most banks price poorly or not at all. A loan that piles onto the energy concentration consumes far more capital than an identical loan that diversifies the book, and pricing that fails to reflect this quietly subsidises concentration.

RAROC operationalises this: return over risk-adjusted capital puts a boutique high-margin concentrated deal and a thin-margin diversifying deal on the same footing, and often flips the intuitive ranking. That is the conversation the credit committee should be having, and it requires a portfolio credit model underneath — expected loss alone cannot produce a marginal-capital number.

Portfolio optimisation and concentration monitoring are the forward-looking, active-management uses: steering new origination toward what improves the tail, and watching in near-real-time where the book is accumulating hidden common-factor bets. This is what elevates CreditVaR from a backward-looking capital report to a steering wheel for the franchise.

Transition: "And at the strategic level, the same tail lens informs the biggest moves the bank makes."
-->

---

## Strategic & Franchise Decisions

- **Stress Testing** — the tail *is* the stress result; regulatory and internal scenarios run through one consistent engine.
- **Acquisitions** — will a target book *diversify* our tail, or double down on a bet we already hold?
- **Loan Sales & Securitisation** — shed the exposures that contribute *most* to the tail, keep the ones that dampen it.
- **Strategic Planning** — align multi-year growth ambitions with the capital the tail will actually demand.

> The biggest decisions the bank makes are, at heart, decisions about the **shape of its future loss tail.**

<!--
Presenter notes:

This is the slide that speaks to the CRO's seat at the strategy table. Frame acquisitions vividly: a target that looks attractive on standalone metrics can be value-destroying if it concentrates the combined book on an economy the bank is already heavily exposed to — and value-creating if it diversifies. Only a portfolio tail model can answer that, because the question is fundamentally about correlation with the *existing* book, not the target's standalone risk. The same lens tells you which loans to sell or securitise: shed the biggest tail contributors, and the capital relief is far larger than the notional shed would suggest.

Stress testing ties the whole platform together for the supervisor: rather than a disconnected stress exercise living in its own spreadsheet, the stress result is simply the tail of the same engine that produces economic capital, run under a prescribed scenario. That consistency is itself a governance win — one engine, one set of assumptions, reconcilable numbers across capital, stress and planning.

Land the closing line hard: the largest strategic moves — who we buy, what we sell, how fast we grow — are all, underneath, decisions about the shape of the future loss tail. A bank that can see that shape decides better.

Transition: "Which brings us to the recommendation — and it is deliberately not 'buy the most complex model.'"
-->

---

<!-- _class: lead -->

# Part 6
## Recommendation

<!--
Presenter notes:

Divider into the close. Keep the room's attention on the fact that the recommendation is a platform and a maturity path, not a single model purchase. Transition straight into the "no best model" slide.
-->

---

## There Is No Single "Best" Model

- **Simple models** buy **transparency** — easy to explain, easy to govern, impossible to argue with. But they *understate the tail.*
- **Sophisticated models** buy **realism** — they see concentration and contagion. But they demand *data, calibration and governance.*

> The right level of complexity is set by the **decision being supported** — not by the ambition of the modelling team.

Match the model to the mandate:
**transparency where you must explain · realism where you must protect.**

<!--
Presenter notes:

This is the intellectual honesty that earns the audience's trust. Resist the temptation — and the vendor cliché — to declare the most complex model the winner. The simple models are not primitive; they are transparent, and transparency is a genuine governance asset when you are standing in front of a board or a supervisor. Their flaw is a thin tail. The sophisticated models are not simply "better"; they are more realistic at the cost of data hunger and governance burden, and a realistic model you cannot calibrate or explain is worse than useless — it is a false comfort.

The resolution is to match the model to the mandate. Where the primary requirement is to *explain and reconcile* — regulatory capital, board communication — favour transparency. Where the primary requirement is to *protect the franchise* — economic capital, severe stress, concentration limits — favour realism and pay the governance price. The bank needs both, which is precisely why the answer is a suite, not a model.

Transition: "And a suite that spans the whole staircase lets the bank climb at its own pace."
-->

---

## The Platform: Start Simple, Climb Deliberately

The CreditVaR suite is **one engine spanning the entire staircase** — so the bank adopts complexity at the pace its *data, governance and decisions* justify.

- **Today** — stand up transparent models on the portfolios where data is thin. Immediate, defensible numbers.
- **Next** — introduce the Gaussian copula for economic capital and regulatory reconciliation on wholesale books.
- **Then** — add Student-t for crisis honesty and Multi-Factor for concentration-aware limits and pricing.

> One consistent engine means every number **reconciles** — from the retail provisioning pool to the board's capital plan.

<!--
Presenter notes:

Sell the platform, not a point solution. The strategic value of a single engine spanning all eight models is that adoption becomes a *managed journey* rather than a series of disconnected tool purchases. The bank starts where its data currently stands — transparent models on thin-data portfolios — and climbs the staircase as data maturity, governance capability and business need justify each step. Nothing is thrown away; each rung builds on the last within one framework.

The reconciliation point is the clincher for a risk audience that lives with fragmented systems. When the retail provisioning number, the wholesale economic-capital number, the regulatory reconciliation and the board's stress result all come out of *one* engine with *one* set of shared inputs, they foot to each other and tell a single coherent story. That consistency is worth as much to the supervisor relationship as any individual model's accuracy.

Frame the phased roadmap as de-risking the investment: value lands at the first phase, and each subsequent phase is optional, sequenced and governed rather than a big-bang bet.

Suggested visual: the staircase again, now with a "you are here" marker and three forward phases highlighted — a roadmap, not a menu.

Transition: "Let me close on the one number that makes the whole case."
-->

---

<!-- _class: lead -->

## The Case in One Number

On an **identical portfolio**, holding every PD, LGD and exposure fixed:

**Expected Loss barely moves across every model.**
**The 99% loss swings by roughly 50% — driven entirely by the correlation assumption.**

> That gap is not noise. It is the **capital cost of what we assume about how borrowers fail together** — and today, in most banks, that assumption is invisible.

**The CreditVaR suite makes it visible, governable, and decision-ready.**

### Recommendation
Adopt CreditVaR as the bank's **portfolio credit-risk platform** — start on today's data, climb the staircase deliberately, and run models in parallel to price model risk itself.

<!--
Presenter notes:

Close on the single most powerful fact in the entire deck, because it is empirical and it is ours. On one and the same book — same borrowers, same PDs, same LGDs, same exposures, same expected loss to the penny — the models we have shown produce a 99% credit VaR that ranges across roughly a fifty-percent spread, from the thin-tailed independent baseline to the fat-tailed t-copula. Expected loss is essentially constant throughout. Every dollar of that spread is the capital consequence of a correlation assumption, and in most institutions that assumption is buried, undocumented and unchallenged.

Make the ask explicit and confident: adopt CreditVaR as the bank's portfolio credit-risk platform. Begin on the data the bank already has, climb the staircase as maturity allows, and — the sophisticated move — run several models in parallel so the *disagreement between them* becomes a managed, reported measure of model risk rather than a hidden liability.

End on the elevation: this is not about buying a more complex number. It is about making the bank's single largest hidden assumption — how its borrowers fail together — visible, governable and decision-ready. Then invite questions.

Suggested visual: two loss curves overlaid — Independent Bernoulli and Student-t — sharing an identical body and mean, with a bold gold bracket spanning the gap between their 99% points, labelled "the price of correlation."
-->
