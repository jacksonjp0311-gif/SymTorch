# Production Roadmap

SymTorch production readiness means contracts, not claims.

## Current Foundations

- Typed tensor/autograd tests and finite-difference checks.
- Policy bundles with deterministic hashes.
- Local development bundle signing alpha.
- Typed domain contracts for policy inputs.
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
