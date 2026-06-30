# Policy Replay CLI

SymTorch `0.19.0` adds a command-line replay gate for decision ledgers.

The CLI reads a `symtorch.decisionLedger.v1` snapshot, rebuilds a fact-predicate policy, replays each recorded context, and exits nonzero if current policy behavior diverges from the recorded decision.

```powershell
pnpm ledger:replay -- `
  --ledger ./ledger.json `
  --program "escalate(X) :- high_risk(X), not approved(X)." `
  --predicates high_risk,approved `
  --threshold 0.5
```

For machine-readable output:

```powershell
pnpm ledger:replay -- --ledger ./ledger.json --program "escalate(X) :- high_risk(X)." --predicates high_risk --json
```

Tolerance for floating point drift is explicit:

```powershell
pnpm ledger:replay -- --ledger ./ledger.json --program "escalate(X) :- high_risk(X)." --predicates high_risk --atol 0.001 --rtol 0.0001
```

## Current Scope

- fact predicates only
- one rule program supplied as CLI text
- entity and non-entity ledger entries
- exact action/acceptance/head matching
- configurable score and threshold tolerance

## Non-Goals

The CLI is not a policy registry, auth system, database, or sandbox for untrusted rule sources. It is a CI/operator gate for detecting decision drift against a recorded ledger.
