# Policy Fixture Suite

SymTorch `0.27.0` adds expected decision checks to the checked-in policy fixture corpus.

The fixture suite is a production-readiness guardrail. Each policy bundle in `examples/policies/` is treated as an executable artifact, not just sample JSON.

## Current Fixtures

- `escalation.policy.json`
- `fraud-review.policy.json`
- `support-routing.policy.json`
- `expected-decisions.json`

## Verification Path

The fixture tests and `pnpm demo:policy-fixtures` verify that each policy can:

- pass `symtorch.policyBundle.v1` schema and hash checks
- materialize through `loadPolicyBundle()`
- execute through `createPolicyAgent()`
- produce entity decisions
- match expected rank, entity, action, acceptance, and score ranges
- record accepted decisions into a ledger
- replay-check the ledger without drift
- load into a migratable workbench policy library

## Run

```powershell
pnpm demo:policy-fixtures
```

The fixture suite is intentionally small. Its job is to keep the public demo policies honest while the runtime grows. The sidecar expectation file is separate from policy bundles so bundle metadata can stay primitive-only and hash-stable under the public bundle contract.
