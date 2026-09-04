# RiskQuant Copilot Governance Manifest

This manifest describes the intended `.github` AI-governance surface.

## Global

| File | Purpose |
|---|---|
| `copilot-instructions.md` | Global engineering constitution and cross-cutting invariants |

## Path-specific instructions

| File | Scope | Purpose |
|---|---|---|
| `instructions/python.instructions.md` | `**/*.py` | Python baseline |
| `instructions/numerical-core.instructions.md` | valuation/simlab/credit/sensitivity_pnl + tests | Numerical architecture and JIT discipline |
| `instructions/testing.instructions.md` | all test paths | Falsification/testing standards |
| `instructions/solutions.instructions.md` | solution/API contract surfaces | Pydantic/API/catalogue/provenance |
| `instructions/evidence-and-change-control.instructions.md` | `**/*` | Evidence states and change control |
| `instructions/observations.instructions.md` | observations + tests | Observation methodology and data integrity |
| `instructions/market-state.instructions.md` | market_state + tests | Identity/state/relabel semantics |
| `instructions/dependence-stress.instructions.md` | dependence_stress + tests + demos | Normative dependence-stress methodology |
| `instructions/tail-risk.instructions.md` | VaR/ES/quantile/risk-measure names + tests | Tail-risk semantics |
| `instructions/sensitivity-pnl.instructions.md` | sensitivity_pnl + tests | Approved sensitivity-PnL methodology |
| `instructions/credit.instructions.md` | credit + tests | Credit methodology/control standards |
| `instructions/valuation.instructions.md` | valuation + tests | Valuation methodology/control standards |
| `instructions/simlab.instructions.md` | simlab + tests | Simulation/RNG/scenario standards |
| `instructions/repository-research.instructions.md` | `**/*` | Verified repository reconnaissance |
| `instructions/methodology.instructions.md` | methodology/docs | Source-of-truth methodology artefacts |

## Agents

| Agent | Authority |
|---|---|
| `agents/riskquant-lead.agent.md` | orchestration/release workflow |
| `agents/riskquant-researcher.agent.md` | repository facts/reconnaissance |
| `agents/riskquant-architect.agent.md` | architecture/boundaries |
| `agents/quant-modeler.agent.md` | mathematical/statistical methodology |
| `agents/implementation-engineer.agent.md` | production implementation |
| `agents/test-engineer.agent.md` | independent testing/falsification |
| `agents/performance-engineer.agent.md` | benchmark/performance evidence |
| `agents/model-validator.agent.md` | independent model validation |
| `agents/riskquant-auditor.agent.md` | final release gate |

## Operating model

The constitution is global. Path-specific files are cumulative constraints. Agent files define role authority and separation of duties.

Where instruction coverage depends on repository layout, verify the actual workspace tree rather than assuming the path from documentation or conversation history.

## Recommended methodology artefact location

Use `.github/methodology/` for compact approved quantitative decision records when the repository does not already have a stronger canonical location. Existing approved methodology artefacts elsewhere remain authoritative according to the global authority hierarchy.
