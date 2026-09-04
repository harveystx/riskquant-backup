---
applyTo: "**/*.py"
---
# Python Engineering Standards

## Module form

Every ordinary Python module begins with:

```python
# riskquant/<relative/path>.py
from __future__ import annotations
```

`__init__.py` is the sole exception and must be byte-empty.

Use explicit, concrete typing. Do not introduce `Any` unless the boundary genuinely requires it and the reason is documented. Do not use `# type: ignore`.

## Data models

Prefer immutable, slotted dataclasses for compact internal value objects. Use `Final` for stable module-level constants and `Literal` for closed vocabularies.

Do not use generic tuples or dictionaries as quasi-structs when a named typed model would make the contract clearer.

## APIs

Parameters, return values, units, ordering, mutability, ownership, and failure semantics must be explicit.

Magic defaults are prohibited where a default changes mathematical or business semantics.

## Exceptions and validation

- Never use `assert` for user/domain input validation.
- Do not catch broad `Exception` unless translating at an explicit boundary and preserving the cause.
- Do not silently coerce malformed input into a different semantic state.
- Validation belongs at the narrowest stable boundary that can state the contract correctly.

## Imports and side effects

Modules must not perform business work at import time. Avoid registration side effects, filesystem writes, global state mutation, and hidden environment inspection.

## Structure

Keep modules cohesive. Avoid generic `utils.py`, `helpers.py`, `manager.py`, or `common.py` dumping grounds. Prefer domain-named modules.

Do not add inheritance merely to share implementation. Composition and explicit functions are preferred where they preserve clearer contracts.
