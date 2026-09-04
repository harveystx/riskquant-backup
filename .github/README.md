# RiskQuant AI Engineering Governance

The `.github` directory contains the repository-wide Copilot constitution, path-specific engineering instructions, specialist agents, and methodology governance.

The intended control loop is:

`Research → Architecture → Quant Methodology → Implementation → Tests → Performance → Independent Validation → Audit`

The system is designed so that:

- methodology is explicit before implementation;
- architecture owns boundaries and data flow;
- numerical cores remain primitive, deterministic, and compact;
- tests falsify rather than merely confirm;
- performance claims require measurements;
- validation is independent of implementation;
- audit is the final release gate.

Do not treat this directory as a substitute for approved model-risk governance outside the repository. It operationalises repository engineering controls.
