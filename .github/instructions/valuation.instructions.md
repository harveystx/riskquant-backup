---
applyTo: "src/riskquant/valuation/**/*.py,tests/valuation/**/*.py"
---
# Valuation Standards

Valuation kernels implement an approved pricing contract; they do not decide which contract applies.

## Product/model separation

Resolve product/model dispatch, flags, market conventions, and valid parameter domains before kernel entry.

## Forward pricing

Where the governing contract specifies Black (1976) forward-option valuation, implement that contract exactly, including forward price construction, discounting stage, option direction/type semantics, and numerical domains.

## FX and discounting

Cross-currency relationships and discounting must be explicit. Where covered-interest-parity forward construction is part of the approved methodology, preserve that convention rather than relying on implicit library behaviour.

## Kernel requirements

Primitive arrays/scalars only; scenario-independent setup outside the kernel; no string/object dispatch; explicit finite/domain validation.

## Verification

Test limiting cases and independent pricing references, not only frozen output examples.
