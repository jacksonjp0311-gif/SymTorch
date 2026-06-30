# Persistence Adapter Alpha

SymTorch `0.6.0` introduces a versioned boundary for decision-ledger persistence.

The runtime still ships only an in-memory `DecisionLedger`, but ledgers can now be exported and restored through `symtorch.decisionLedger.v1` snapshots. This is the contract storage adapters should use for file, IndexedDB, SQLite, or service-backed replay.

```ts
import {
  DecisionLedger,
  isSerializedDecisionLedger,
  loadDecisionLedger,
  serializeDecisionLedger
} from "@symtorch/agent";

const snapshot = serializeDecisionLedger(agent.ledger);
const json = JSON.stringify(snapshot);
const parsed = JSON.parse(json);

if (!isSerializedDecisionLedger(parsed)) {
  throw new Error("Invalid decision ledger snapshot.");
}

const restored = loadDecisionLedger(new DecisionLedger(), parsed);
```

## Contract

```json
{
  "schemaVersion": "symtorch.decisionLedger.v1",
  "entries": []
}
```

Each entry contains:

- deterministic `id`
- ISO `createdAt`
- `kind` of `agent` or `entity`
- JSON-safe context snapshot
- versioned `symtorch.agentDecision.v1` decision payload

## Current Limits

- No built-in file, IndexedDB, or SQLite adapter yet.
- No migrations are needed yet because this is the first ledger schema.
- Snapshot validation is structural, not a security sandbox.
- Storage adapters should still enforce size limits, retention, encryption, and access control outside SymTorch.
