# Persistence Adapter Alpha

SymTorch `0.6.0` introduced a versioned boundary for decision-ledger persistence. SymTorch `0.17.0` adds a Node filesystem sink and replay verification on top of that boundary.

The runtime still uses an in-memory `DecisionLedger`, but ledgers can be exported and restored through `symtorch.decisionLedger.v1` snapshots. This is the contract storage adapters should use for file, IndexedDB, SQLite, or service-backed replay.

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

Node.js file persistence is available through a separate subpath:

```ts
import { FileDecisionLedgerSink } from "@symtorch/agent/node";

const sink = new FileDecisionLedgerSink("./ledger.json");
await sink.write(agent.ledger.snapshot());
const restoredSnapshot = await sink.read();
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

- A Node file sink exists. IndexedDB, SQLite, and service-backed adapters are not built in yet.
- No migrations are needed yet because this is the first ledger schema.
- Snapshot validation is structural, not a security sandbox.
- Storage adapters should still enforce size limits, retention, encryption, and access control outside SymTorch.
