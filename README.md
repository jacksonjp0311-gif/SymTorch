# SymTorch

[![CI](https://github.com/jacksonjp0311-gif/SymTorch/actions/workflows/ci.yml/badge.svg)](https://github.com/jacksonjp0311-gif/SymTorch/actions/workflows/ci.yml)
![TypeScript](https://img.shields.io/badge/TypeScript-strict-blue)
![pnpm](https://img.shields.io/badge/pnpm-11.7.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**Differentiable tensors, trainable symbolic rules, and explainable agent decisions in TypeScript.**

SymTorch is a JavaScript-native research and engineering project for building neuro-symbolic AI systems in the browser, Node.js, and edge-style runtimes. It combines a PyTorch-inspired eager tensor/autograd core with a fuzzy-logic rule engine, learnable predicates, working memory, and agent-facing explanations.

The bet is simple: developers should be able to write rules humans can read, compile them into differentiable tensor programs, train them with gradient descent, and still ask, "why did the system decide that?"

```prolog
escalate(X) :- high_risk(X), not approved(X).
escalate(X) :- customer_vip(X).
```

Those rules can produce one aggregated `escalate(X)` score, train their predicates from feedback, and preserve an explanation trace for every contributing rule.

## 30-Second Demo

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

The rule stays readable, `high_risk(X)` can train, and the final output remains a versioned JSON-safe explanation trace.

## Why SymTorch

Most JavaScript ML tools focus on neural models alone. Most rule engines are discrete and brittle. SymTorch is building the missing middle: a tensor system where symbolic rules are first-class differentiable programs.

Use it for:

- explainable agent policies
- trainable business rules
- browser or Node.js ML experiments without a Python service
- fuzzy reasoning over observations, facts, and learned predicates
- LLM-assisted rule authoring where SymTorch executes, trains, and explains

## Why this is not just JS tensors

Tensors and autograd are the substrate, not the point. SymTorch uses them to make readable symbolic policies executable, differentiable, and trainable without discarding the rule that a human wrote.

The intended path is:

```text
readable rule -> fuzzy tensor execution -> trainable predicate -> explanation trace -> agent decision
```

That means a policy such as `escalate(X) :- high_risk(X), not approved(X).` can stay readable while `high_risk(X)` becomes a learnable predicate, the rule score remains differentiable, and the final decision still carries a trace of which predicates and rules contributed. SymTorch is early and foundation-first; it is not claiming parity with mature tensor frameworks. The work right now is to make the math and explanations credible enough to build on.

## Current Capabilities

- Eager `Tensor` API with CPU `float32` typed-array storage.
- Backend registry and storage model with CPU available and WebGPU registered as an explicit placeholder acceleration target.
- Reverse-mode autograd with `.backward()` and gradient accumulation.
- Core ops including `matmul`, broadcasting, reductions, activations, `softmax`, `logSoftmax`, and `logsumexp`.
- Vector-symbolic binding ops with differentiable circular convolution and circular correlation.
- `nn.Module`, `Parameter`, `Linear`, `LayerNorm`, `Sequential`, `SGD`, `Adam`, MSE, cross-entropy, and BCE-with-logits.
- Prolog-like fuzzy rules with product-t-norm conjunction and probabilistic-OR aggregation.
- Fixed, fact-backed, threshold, and linear predicates.
- `RuleTrainer` for fitting differentiable rules against labeled examples.
- `FactStore` working memory for observations and entity-scoped facts.
- Same-head rule aggregation with per-rule explanations.
- Versioned, JSON-safe explanation traces for agent integrations.
- Versioned, JSON-safe agent decision contracts with structural validators.
- Entity batch evaluation and ranking over fact stores.
- Agent loop primitives for observation -> decision -> serialized explanation, including entity batches.
- Versioned decision ledger snapshots for replay and persistence adapters.
- Holographic memory primitive for binding, superposing, and recalling vector symbols.
- Browser policy workbench with scenario selection, trainable predicates, import/export, smoke tests, and Playwright E2E coverage.
- Versioned `symtorch.scenario.v1` contracts for JSON-safe policy scenarios.
- Versioned `symtorch.trainingRun.v1` records for browser-side training history.
- WebGPU package with runtime detection, buffer pooling, explicit tensor residency, same-shape elementwise kernel prototypes, and a browser parity gate when WebGPU is available.

## Quickstart

```powershell
pnpm install
pnpm typecheck
pnpm test
pnpm build
```

Run the demos:

```powershell
pnpm example:linear
pnpm example:routing
pnpm example:trainable-routing
pnpm example:batch-routing
```

## Executable Demos

```powershell
pnpm demo:gradients
pnpm demo:rule
pnpm demo:ledger
pnpm demo:all
```

These are short verification scripts. They are meant to show the core claims directly: gradient sanity, trainable readable rules, and JSON-safe agent ledger replay.

## Trainable Rule Example

```ts
import { tensor } from "@symtorch/core";
import {
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
console.log(result.explanation);
```

## Rule Authoring Validation

```ts
import { FactPredicate, PredicateRegistry, validateProgram, validatePrograms } from "@symtorch/logic";

const validation = validateProgram("escalate(X) :- high-risk(X).");

if (!validation.ok) {
  console.log(validation.error.line, validation.error.column);
  console.log(validation.error.message);
}

const registry = new PredicateRegistry()
  .register(new FactPredicate("high_risk"))
  .register(new FactPredicate("approved"));

const drafts = validatePrograms([
  { id: "draft-a", source: "escalate(X) :- high_risk(X), not approved(X)." },
  { id: "draft-b", source: "escalate(X) :- customer_vip(X)." }
], { registry });

console.log(drafts.map((draft) => [draft.id, draft.result.ok]));
```

## Monorepo Layout

```text
packages/
  core/      tensors, ops, shape utilities, CPU execution, autograd
  nn/        modules, layers, optimizers, and losses
  logic/     fuzzy rules, predicates, fact stores, trainers, explanations
  agent/     working memory and rule-based decision loops
  webgpu/    WebGPU detection and future accelerated backend

examples/
  linear-regression/
  neuro-symbolic-routing/
  trainable-routing/
```

## Documentation

- [Architecture](docs/architecture.md)
- [Public API surface](docs/api-surface.md)
- [30-second demo](docs/demo.md)
- [Browser playground prep](docs/browser-playground.md)
- [Holographic memory](docs/holographic-memory.md)
- [Persistence adapter alpha](docs/persistence.md)
- [Backend abstraction alpha](docs/backend-abstraction.md)
- [GPU backend plan](docs/gpu-backend-plan.md)
- [WebGPU residency prototype](docs/webgpu-residency.md)
- [WebGPU add kernel prototype](docs/webgpu-add-kernel.md)
- [WebGPU browser parity gate](docs/webgpu-browser-parity.md)
- [WebGPU same-shape elementwise kernels](docs/webgpu-elementwise-kernels.md)
- [Production readiness alpha](docs/production-readiness.md)
- [Limitations](docs/limitations.md)
- [Changelog](CHANGELOG.md)

## Design Principles

- **CPU first, WebGPU second.** CPU semantics are the correctness oracle.
- **Eager and debuggable.** Start with dynamic execution before graph capture.
- **Rules stay readable.** Learning should not erase interpretability.
- **Explanations are data.** Decision traces should be structured, renderable, and testable.
- **Agent-scale performance.** Optimize for realistic browser/Node agent workloads before chasing massive-model training.

## Roadmap

Near term:

- backend dispatch foundation for future WebGPU kernels
- typed domains and guarded grounding
- scenario contract loaders beyond the playground
- richer explanation renderers and decision cards
- more tensor ops and gradient checks
- policy replay against decision ledgers

Mid term:

- WebGPU elementwise, reduction, and matmul kernels
- sparse or masked relational execution
- attention and transformer blocks
- LLM-assisted rule authoring workflows

Long term:

- vector-symbolic memory operations
- kernel fusion and graph capture
- federated or edge-oriented agent loops

## Status

SymTorch is early, active, and intentionally foundation-first. The `0.13.0` workspace line is the WebGPU unary elementwise kernel set: CPU remains the correctness oracle, explicit WebGPU kernels now cover `add`, `sub`, `mul`, `div`, `neg`, `abs`, `exp`, `log`, `relu`, `sigmoid`, `sqrt`, and `tanh`, and browser parity now gates the elementwise set when WebGPU is available.

The version labels in this repository are engineering checkpoints for the private workspace. They are not production deployment, autonomous authority, or npm stability claims. See [CHANGELOG.md](CHANGELOG.md) for the seal history and [Production Readiness Alpha](docs/production-readiness.md) for the current gate.
