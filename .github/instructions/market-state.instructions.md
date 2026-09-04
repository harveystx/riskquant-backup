---
applyTo: "src/riskquant/market_state/**/*.py,tests/market_state/**/*.py"
---
# Market-State Standards

Market state is canonical numerical state plus identity/mapping semantics required to interpret it.

## Identity

Preserve the distinction:

`*_id` → external identifier → `*_uid` → global business identity → `*_uix` → compact numerical position.

Do not conflate identifiers with array positions.

## State contract

Define explicitly:

- universe/order;
- effective date/time;
- source/version;
- dtype/shape;
- ownership;
- missingness;
- mapping semantics.

## Relabelling versus history

A futures/curve/factor roll or relabelling can change which canonical factor an exposure points to without constituting a new economic market observation. Do not silently rewrite historical observations merely because the current identity mapping changed.

The implementation must distinguish identity/mapping changes from historical-data transformations.

## Performance

Resolve UID→UIX mappings before numerical kernels. Prefer compact integer arrays in the core.
