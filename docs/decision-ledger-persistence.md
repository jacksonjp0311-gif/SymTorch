# Decision Ledger Persistence and Replay

SymTorch `0.17.0` adds a small production-shaped persistence boundary for agent decisions.

## What It Provides

- `DecisionLedger.snapshot()` creates a versioned JSON-safe ledger snapshot.
- `FileDecisionLedgerSink` persists that snapshot in Node.js through `@symtorch/agent/node`.
- `verifyDecisionLedgerReplay()` compares recorded decisions with decisions recomputed from recorded contexts.

```ts
import { verifyDecisionLedgerReplay } from "@symtorch/agent";
import { FileDecisionLedgerSink } from "@symtorch/agent/node";

const sink = new FileDecisionLedgerSink("./ledger.json");
await sink.write(agent.ledger.snapshot());

const snapshot = await sink.read();
const report = verifyDecisionLedgerReplay(snapshot, (entry) => {
  // Rebuild the same policy, feed entry.context, and return decideTrace().
  return replayDecision(entry);
});
```

## What It Detects

Replay verification detects drift between a recorded decision and current policy behavior for the same recorded context. This is useful after rule, predicate, or threshold changes.

## What It Does Not Provide

- database durability
- access control
- encryption
- retention policy
- distributed consistency
- proof that an external action was safe or authorized

The ledger is an audit and replay primitive. Applications remain responsible for storage operations, privacy controls, and deployment policy.
