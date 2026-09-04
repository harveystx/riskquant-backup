---
applyTo: "src/riskquant/simlab/**/*.py,tests/simlab/**/*.py"
---
# Simulation-Lab Standards

Simulation is a contract over scenarios, randomness, ordering, and aggregation—not merely an array generator.

## Compile/scenario separation

All scenario-invariant mapping, validation, dispatch, interpolation, and state preparation must occur before scenario execution.

## RNG

Every stochastic component must define RNG ownership, seed/state semantics, stream independence, and random-number consumption order where reproducibility depends on it.

## Streaming

Streaming/generator APIs are permitted only where the public contract explicitly requires bounded-memory processing or incremental consumption semantics.

Do not introduce `yield` solely to avoid deciding ownership or output-shape contracts.

## Parallelism

Parallel execution requires evidence that the algorithm is safe, deterministic as contracted, and materially faster. Avoid implicit shared state.

## Scenario indexing

Prefer integer scenario indices and gathers. Scenario order is part of the contract whenever downstream attribution, reproducibility, or exact reconstruction depends on it.
