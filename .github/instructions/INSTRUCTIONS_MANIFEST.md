# RiskQuant Instruction Layer

These path-specific instructions layer onto `.github/copilot-instructions.md`. VS Code applies both repository-wide and matching path-specific instruction files.

| File | Scope | Purpose |
|---|---|---|
| `python.instructions.md` | `**/*.py` | Python engineering standard |
| `numerical-core.instructions.md` | valuation / simlab / credit / sensitivity_pnl | Numerical core, JIT, memory, DOD/SoA |
| `testing.instructions.md` | test files | Falsification, independent oracles, exactness |
| `solutions.instructions.md` | solutions | API, policy, provenance, golden masters |
| `evidence-and-change-control.instructions.md` | repository-wide | Evidence states, change control, performance claims, ledgers |
| `observations.instructions.md` | observations | Empirical observations and return transformations |
| `market-state.instructions.md` | market_state | Canonical market state and identity mapping |
| `dependence-stress.instructions.md` | dependence_stress + related tests/scripts | Normative dependence-stress methodology and invariants |
| `tail-risk.instructions.md` | VaR / ES / quantile-related modules | Tail-risk methodology |
| `sensitivity-pnl.instructions.md` | sensitivity_pnl + tests | Taylor-expansion PnL and scenario alignment |
| `credit.instructions.md` | credit + tests | Credit methodology and numerical boundaries |
| `valuation.instructions.md` | valuation + tests | Pricing methodology and numerical architecture |
| `simlab.instructions.md` | simlab + tests | Scenario generation, reproducibility, memory |

## Governance

Keep global architectural law in `.github/copilot-instructions.md`. Keep domain-specific mathematical or operational contracts here. Keep role behaviour in `.github/agents/*.agent.md`. Avoid duplicating methodology across agents.
