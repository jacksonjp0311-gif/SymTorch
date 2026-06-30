# Production Hardening

SymTorch `0.21.0` adds first-pass production hardening contracts across runtime limits, error taxonomy, policy packaging, persistence, and backend dispatch.

This is still not a production deployment claim. It is a set of boundaries that make SymTorch safer to embed in applications and easier to inspect under failure.

## Runtime Limits

Core tensor allocation can be bounded:

```ts
import { configureRuntimeLimits } from "@symtorch/core";

configureRuntimeLimits({ maxTensorElements: 1_000_000 });
```

Rule parsing and evaluation can be bounded:

```ts
import { FuzzyRuleEngine, RuleProgram } from "@symtorch/logic";

const program = new RuleProgram(source, {
  limits: {
    maxRuleSourceLength: 8_000,
    maxRules: 100,
    maxPredicatesPerRule: 16
  }
});

const engine = new FuzzyRuleEngine(registry, {
  limits: { maxEntitiesPerEvaluation: 1_000 }
});
```

Agent batches and replay can be bounded:

```ts
const agent = new RuleAgent(program, engine, 0.5, {
  limits: { maxEntitiesPerBatch: 1_000 }
});

verifyDecisionLedgerReplay(snapshot, replay, {
  limits: { maxReplayEntries: 10_000 }
});
```

## Error Taxonomy

The core package now exports:

- `SymTorchError`
- `ResourceLimitError`
- `BackendExecutionError`

The logic package now exports:

- `RuleParseError`
- `RuleValidationError`
- `PredicateEvaluationError`

Existing error messages remain human-readable, while new error classes let embedding apps distinguish resource, backend, predicate, and validation failures.

## Policy Bundles

Policy bundles package rules, predicate metadata, version metadata, and a deterministic hash:

```ts
import { createPolicyBundle, verifyPolicyBundleHash } from "@symtorch/logic";

const bundle = createPolicyBundle({
  name: "Escalation Policy",
  version: "2026.06.30",
  rules: "escalate(X) :- high_risk(X), not approved(X).",
  predicates: [
    { kind: "threshold", name: "high_risk", valueKey: "risk", threshold: 0.7, slope: 10 },
    { kind: "fact", name: "approved" }
  ],
  metadata: { owner: "risk" }
});

console.log(verifyPolicyBundleHash(bundle));
```

Bundles use `symtorch.policyBundle.v1`. The current hash is an integrity guard for local/package workflows, not a cryptographic signature.

## Append-Oriented Ledgers

Node now exposes `AppendFileDecisionLedgerSink` from `@symtorch/agent/node`.

It stores one JSON ledger entry per line and reconstructs a `symtorch.decisionLedger.v1` snapshot on read. This is useful for simple append-only operator logs without rewriting the whole ledger snapshot after every decision.

## Backend Dispatch Boundary

`@symtorch/core` now exposes `runBackendKernel()`, `BackendKernelName`, and backend `execute()` support.

CPU remains the correctness oracle. WebGPU tensors still do not execute core tensor ops through dispatch yet; instead, the dispatch boundary now fails with `BackendExecutionError` when an operation reaches a backend that has no kernel implementation.

## Current Limits

- Runtime limits are opt-in except tensor allocation configuration.
- Policy bundle hashing is deterministic but not a security signature.
- Append file ledgers are not databases and do not provide retention, locking, or compaction.
- Backend dispatch exists as a contract; GPU execution remains explicit through `@symtorch/webgpu`.
