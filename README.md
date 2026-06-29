# SymTorch

**Differentiable tensors, trainable symbolic rules, and explainable agent decisions in TypeScript.**

SymTorch is a JavaScript-native research and engineering project for building neuro-symbolic AI systems in the browser, Node.js, and edge-style runtimes. It combines a PyTorch-inspired eager tensor/autograd core with a fuzzy-logic rule engine, learnable predicates, working memory, and agent-facing explanations.

The bet is simple: developers should be able to write rules humans can read, compile them into differentiable tensor programs, train them with gradient descent, and still ask, “why did the system decide that?”

```prolog
escalate(X) :- high_risk(X), not approved(X).
escalate(X) :- customer_vip(X).
```

Those rules can produce one aggregated `escalate(X)` score, train their predicates from feedback, and preserve an explanation trace for every contributing rule.

## Why SymTorch

Most JavaScript ML tools focus on neural models alone. Most rule engines are discrete and brittle. SymTorch is building the missing middle: a tensor system where symbolic rules are first-class differentiable programs.

Use it for:

- explainable agent policies
- trainable business rules
- browser or Node.js ML experiments without a Python service
- fuzzy reasoning over observations, facts, and learned predicates
- LLM-assisted rule authoring where SymTorch executes, trains, and explains

## Current Capabilities

- Eager `Tensor` API with CPU `float32` typed-array storage.
- Reverse-mode autograd with `.backward()` and gradient accumulation.
- Core ops including `matmul`, broadcasting, reductions, activations, `softmax`, `logSoftmax`, and `logsumexp`.
- `nn.Module`, `Parameter`, `Linear`, `LayerNorm`, `Sequential`, `SGD`, `Adam`, MSE, cross-entropy, and BCE-with-logits.
- Prolog-like fuzzy rules with product-t-norm conjunction and probabilistic-OR aggregation.
- Fixed, fact-backed, threshold, and linear predicates.
- `RuleTrainer` for fitting differentiable rules against labeled examples.
- `FactStore` working memory for observations and entity-scoped facts.
- Same-head rule aggregation with per-rule explanations.
- Agent loop primitives for observation -> decision -> explanation.
- WebGPU package shell with runtime detection and backend planning.

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
```

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

## Design Principles

- **CPU first, WebGPU second.** CPU semantics are the correctness oracle.
- **Eager and debuggable.** Start with dynamic execution before graph capture.
- **Rules stay readable.** Learning should not erase interpretability.
- **Explanations are data.** Decision traces should be structured, renderable, and testable.
- **Agent-scale performance.** Optimize for realistic browser/Node agent workloads before chasing massive-model training.

## Roadmap

Near term:

- richer explanation renderers and decision cards
- typed domains and guarded grounding
- batched rule evaluation over fact stores
- more tensor ops and gradient checks
- browser demos

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

SymTorch is early, active, and intentionally foundation-first. The current implementation is useful for small differentiable-rule experiments and agent-policy prototypes, while the tensor and backend layers continue to harden.

