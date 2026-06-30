# Production Roadmap

SymTorch production readiness means contracts, not claims.

## Current Foundations

- Typed tensor/autograd tests and finite-difference checks.
- Policy bundles with deterministic hashes.
- Local development bundle signing alpha.
- Structured signed-bundle diagnostics and local admission assessment.
- Typed domain contracts for policy inputs.
- Machine-readable production readiness report for all ten hardening tracks.
- Versioned decision trace snapshots with optional ledger state.
- Browser workbench state, policy libraries, and migrations.
- Decision ledgers with replay verification.
- Checked-in policy fixtures with expected decision assertions.
- Browser build, smoke, and E2E gates.

## Ten Production Tracks

1. Typed domains for entities, facts, and policy inputs.
2. Policy bundle signing and trusted key management.
3. Durable persistence adapters beyond local browser state.
4. Expected result snapshots for decisions, traces, and ledgers.
5. Runtime limits across parsing, evaluation, tensors, replay, and browser workloads.
6. Typed error taxonomy for parse, validation, predicate, replay, resource, and backend failures.
7. CPU/GPU backend parity and gradient parity gates.
8. Public API stability snapshots and deprecation policy.
9. Security model for untrusted policy source and imported bundles.
10. Real example apps that demonstrate local-first explainable policy execution.

## Current Non-Claims

The signing alpha uses deterministic local signatures for fixture and development workflows. It is not cryptographic production signing. Durable multi-user storage, remote policy registries, sandboxed execution, and WebGPU backend dispatch remain future work.

## 0.30.0 Contract Layer

The ten tracks are now represented by `symtorch.productionReadiness.v1` through `getProductionReadinessReport()`. Policy bundles can be checked with `assessPolicyBundleSecurity()` before local admission, and decisions can be frozen with `symtorch.decisionTraceSnapshot.v1` snapshots. These are contract foundations; they are not a substitute for real cryptographic signing, sandboxed execution, durable databases, or CPU/GPU parity gates.
