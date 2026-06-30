# Observability Hooks

SymTorch `0.20.0` adds lightweight observer hooks for rule evaluation, agent decisions, ledger appends, and replay checks.

The hooks are intentionally dependency-free. They emit structured event objects that can be forwarded to logs, metrics, traces, tests, or an application-specific monitoring layer.

## Logic Events

```ts
import { FactPredicate, FuzzyRuleEngine, PredicateRegistry, RuleProgram } from "@symtorch/logic";

const events: unknown[] = [];
const program = new RuleProgram("escalate(X) :- high_risk(X).");
const registry = new PredicateRegistry().register(new FactPredicate("high_risk"));
const engine = new FuzzyRuleEngine(registry, {
  observer: {
    onRuleEvaluate: (event) => events.push(event),
    onProgramEvaluate: (event) => events.push(event)
  }
});

engine.evaluateProgramGrouped(program, { high_risk: 0.9 });
```

Current logic events:

- `rule.evaluate`: emitted after one fuzzy rule evaluates.
- `program.evaluate`: emitted after grouped program evaluation completes.

Events include stable context keys, rule/head identifiers, scores, counts, and `durationMs`.

## Agent Events

```ts
import { RuleAgent } from "@symtorch/agent";

const agent = new RuleAgent(program, engine, 0.5, {
  observer: {
    onDecision: (event) => console.log(event),
    onLedgerAppend: (event) => console.log(event),
    onReplay: (event) => console.log(event)
  }
});
```

Current agent events:

- `agent.decision`: emitted when a JSON-safe decision trace is produced.
- `ledger.append`: emitted after a decision is appended to the in-memory ledger through `RuleAgent`.
- `ledger.replay`: emitted after `verifyDecisionLedgerReplay()` completes when an observer is supplied through the replay options.

Replay observer example:

```ts
verifyDecisionLedgerReplay(snapshot, replay, {
  atol: 0.001,
  observer: {
    onReplay: (event) => console.log(event)
  }
});
```

## Current Scope

- Hooks are synchronous and lightweight.
- Event payloads are JSON-safe by convention.
- Hooks are best-effort operator signals; throwing inside a hook currently propagates to the caller.
- `durationMs` is for observability, not deterministic replay.

## Non-Goals

These hooks are not a metrics backend, distributed tracing implementation, durable audit log, or security boundary. Persist decision ledgers separately when replay or audit retention matters.
