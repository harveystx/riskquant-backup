---
applyTo: "src/riskquant/solutions/**/*.py,tests/solutions/**/*.py,src/riskquant/**/{schemas,catalog,recipes,run,api}.py"
---
# Solution-Layer Standards

The solution layer owns public contracts and orchestration. It does not perform the numerical work that belongs in CORE.

## Canonical structure

A public solution package should normally contain:

- `schemas.py` — Pydantic v2 boundary contracts;
- `catalog.py` — registered models/methods and catalogue integrity;
- `recipes.py` — approved configurations/policy bundles;
- `run.py` — controlled orchestration;
- `api.py` — sole public surface.

Do not create every file mechanically. Use the structure that the contract requires.

## Pydantic

Use Pydantic v2. For strict public contracts, prefer:

```python
model_config = ConfigDict(extra="forbid")
```

Do not allow arbitrary extra fields to silently enter a public contract.

## Policy ownership

The solution layer resolves model choice, methodology recipe, mappings, validation policy, provenance, and dispatch. It then passes compact canonical state to CORE.

## Provenance

Material outputs should retain sufficient provenance to reconstruct the methodology/version/configuration used, subject to the public API contract.

## Testing

Maintain both structural contract tests and frozen numerical masters where deterministic numerical outputs are part of the contract.
