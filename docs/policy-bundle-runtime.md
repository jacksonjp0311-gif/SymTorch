# Policy Bundle Runtime

SymTorch `0.25.0` turns policy bundles into executable runtime artifacts and lets the browser workbench use the same bundle-backed path, including a local versioned policy library and explicit migration helpers for saved workbench artifacts.

The flow is now:

```text
policy bundle -> verify hash -> materialize predicates -> RuleProgram -> FuzzyRuleEngine -> RuleAgent -> decision trace
```

## Load a Bundle

```ts
import { createPolicyBundle, loadPolicyBundle } from "@symtorch/logic";

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

const { program, registry, engine } = loadPolicyBundle(bundle);
```

`loadPolicyBundle()` verifies the `symtorch.policyBundle.v1` schema and hash before materializing:

- `fact` predicates into `FactPredicate`
- `threshold` predicates into `ThresholdPredicate`
- `linear` predicates into `LinearPredicate`

## Create an Agent

```ts
import { createPolicyAgent } from "@symtorch/agent";

const agent = createPolicyAgent(bundle, {
  threshold: 0.5,
  limits: {
    maxRuleSourceLength: 1_000,
    maxEntitiesPerBatch: 100
  }
});

agent.memory.observeEntity("case-1", { risk: 0.9, approved: 0.1 });
const decision = agent.decideEntityTrace("case-1");
```

The result is a JSON-safe agent decision with a versioned explanation trace.

## Demo

```powershell
pnpm demo:policy
pnpm demo:golden-policy
```

The demo creates a bundle, verifies it, loads it, evaluates a case, records a ledger entry, and verifies replay.

## Current Scope

- Bundle loading is local and synchronous.
- Bundle hashes are deterministic integrity checks, not cryptographic signatures.
- Predicate materialization currently supports fact, threshold, and linear predicates.
- Bundle runtime does not yet include a remote registry, policy promotion workflow, or migration runner.
