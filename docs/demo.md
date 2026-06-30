# 30-Second Demo

This demo shows the core SymTorch loop: readable rule, fuzzy tensor execution, trainable predicate, and explanation-preserving decision output.

For executable verification, run:

```powershell
pnpm demo:gradients
pnpm demo:rule
pnpm demo:ledger
pnpm demo:all
pnpm ledger:replay -- --ledger ./ledger.json --program "escalate(X) :- high_risk(X)." --predicates high_risk
```

- `demo:gradients` checks finite-difference sanity for reductions, cross entropy, and LayerNorm.
- `demo:rule` trains a readable fuzzy rule and prints a versioned explanation trace.
- `demo:ledger` ranks entity decisions, filters accepted results, persists a ledger snapshot, and verifies replay.
- `ledger:replay` turns a persisted ledger into a command-line drift gate.

```ts
import { tensor } from "@symtorch/core";
import {
  decisionTrace,
  FuzzyRuleEngine,
  PredicateRegistry,
  RuleProgram,
  RuleTrainer,
  ThresholdPredicate
} from "@symtorch/logic";

const program = new RuleProgram(`
  escalate(X) :- high_risk(X), not approved(X).
`);

const highRisk = new ThresholdPredicate("high_risk", "risk", 0.9, 10);
const registry = new PredicateRegistry()
  .register(highRisk)
  .fixed("approved", (_call, context) => tensor(context.approved as number));

const engine = new FuzzyRuleEngine(registry);
const trainer = new RuleTrainer(engine, program.rules[0]!, registry, { learningRate: 0.2 });

trainer.fit([
  { risk: 0.15, approved: 0.05, label: 0 },
  { risk: 0.72, approved: 0.05, label: 1 },
  { risk: 0.90, approved: 0.95, label: 0 }
], { epochs: 100 });

const result = trainer.predict({ risk: 0.82, approved: 0.08 });

console.log(result.score.item());
console.log(decisionTrace(result));
```

Example trace shape:

```json
{
  "schemaVersion": "symtorch.explanation.v1",
  "type": "rule",
  "head": "escalate(X)",
  "score": 0.74,
  "predicates": [
    {
      "name": "high_risk(X)",
      "negated": false,
      "kind": "learnable"
    },
    {
      "name": "not approved(X)",
      "negated": true,
      "kind": "fixed"
    }
  ]
}
```
