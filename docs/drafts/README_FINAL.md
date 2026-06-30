# SymTorch

[![CI](https://github.com/jacksonjp0311-gif/SymTorch/actions/workflows/ci.yml/badge.svg)](https://github.com/jacksonjp0311-gif/SymTorch/actions/workflows/ci.yml)
![TypeScript](https://img.shields.io/badge/TypeScript-strict-blue)
![pnpm](https://img.shields.io/badge/pnpm-11.7.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**Differentiable tensors, trainable symbolic rules, and explainable agent decisions in TypeScript.**

SymTorch is a JavaScript-native research and engineering project for building **neuro-symbolic AI systems** in the browser, Node.js, and edge-style runtimes.

It combines:

- a PyTorch-inspired eager **tensor + autograd** core
- a Prolog-like **fuzzy rule engine** with learnable predicates
- **working memory** over observations and entity facts
- **versioned, JSON-safe explanation traces** and decision ledgers (designed for agents)

The bet is simple: developers should be able to write rules humans can read, compile them into differentiable tensor programs, train them with gradient descent, and still ask: **“why did the system decide that?”**

```prolog
escalate(X) :- high_risk(X), not approved(X).
escalate(X) :- customer_vip(X).
```

Those rules can produce one aggregated `escalate(X)` score, train their predicates from feedback, and preserve an explanation trace for every contributing rule.

---

## 30-second demo

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

---

## The product loop (workbench prototype)

The current browser workbench demonstrates the end-to-end loop SymTorch is moving toward:

```text
edit readable policy
-> package hashed policy bundle
-> verify bundle
-> save/load local policy library
-> materialize RuleAgent
-> evaluate entity decisions
-> inspect explanation trace
-> record decision ledger
-> replay-check behavior
```

Run it locally:

```powershell
pnpm playground:browser
```

The workbench is intentionally **local-first**. It proves the contracts and runtime path before adding remote registries, auth, or multi-user storage.

---

## Why SymTorch

Most JavaScript ML tools focus on neural models alone. Most rule engines are discrete and brittle.

SymTorch is building the missing middle: **a tensor system where symbolic rules are first-class differentiable programs**.

Use it for:

- explainable agent policies
- trainable business rules
- browser or Node.js ML experiments without a Python service
- fuzzy reasoning over observations, facts, and learned predicates
- LLM-assisted rule authoring where SymTorch executes, trains, and explains

---

## Install

### Option A: from npm (after first public publish)

```powershell
pnpm add @symtorch/core @symtorch/logic @symtorch/agent
```

### Option B: from source (today)

```powershell
git clone https://github.com/jacksonjp0311-gif/SymTorch
cd SymTorch
pnpm install
pnpm test
pnpm build
```

---

## Quickstart (repo gates)

```powershell
pnpm install
pnpm typecheck
pnpm test
pnpm build
```

Run examples:

```powershell
pnpm example:linear
pnpm example:routing
pnpm example:trainable-routing
pnpm example:batch-routing
```

Run verification demos:

```powershell
pnpm demo:gradients
pnpm demo:rule
pnpm demo:ledger
pnpm demo:policy
pnpm demo:golden-policy
pnpm demo:policy-fixtures
pnpm demo:all
```

---

## Current capabilities (detailed)

This section is intentionally exhaustive: it’s the factual capability surface of the repo as of `0.30.0`.

- Eager `Tensor` API with CPU `float32` typed-array storage.
- Backend registry and storage model with CPU available and WebGPU registered as an explicit placeholder acceleration target.
- Typed production-hardening errors, runtime limits, and an explicit backend dispatch boundary.
- Reverse-mode autograd with `.backward()` and gradient accumulation.
- Core ops including `matmul`, broadcasting, reductions, activations, `softmax`, `logSoftmax`, and `logsumexp`.
- Vector-symbolic binding ops with differentiable circular convolution and circular correlation.
- `nn.Module`, `Parameter`, `Linear`, `LayerNorm`, `Sequential`, `SGD`, `Adam`, MSE, cross-entropy, and BCE-with-logits.
- Prolog-like fuzzy rules with product-t-norm conjunction and probabilistic-OR aggregation.
- Fixed, fact-backed, threshold, and linear predicates.
- Typed domain contracts for policy input validation.
- `RuleTrainer` for fitting differentiable rules against labeled examples.
- `FactStore` working memory for observations and entity-scoped facts.
- Same-head rule aggregation with per-rule explanations.
- Versioned, JSON-safe explanation traces for agent integrations.
- Versioned, JSON-safe agent decision contracts with structural validators.
- Entity batch evaluation and ranking over fact stores.
- Agent loop primitives for observation -> decision -> serialized explanation, including entity batches.
- Versioned decision ledger snapshots for replay and persistence adapters.
- Versioned policy bundle contracts with deterministic integrity hashes.
- Local development policy bundle signing alpha.
- Structured bundle verification diagnostics and policy security assessment for local admission gates.
- HMAC-SHA256 policy bundle signing alpha with trusted-key metadata.
- Policy admission reports for sandbox-style import gates.
- Versioned production-readiness reports covering the ten production hardening tracks.
- Versioned decision trace snapshots for freezing decisions and optional ledger state as fixtures.
- Checked-in golden trace corpus fixtures for behavior drift checks.
- Typed domain grounding helpers for accepted/rejected entity contexts.
- Durable ledger adapter descriptors, retention helpers, and operational event sinks.
- Executable policy bundle runtime that materializes predicates and creates `RuleAgent` instances.
- Browser policy bundle import/export with hash verification and bundle-backed decisions.
- Versioned browser policy libraries for saving, loading, exporting, and importing local bundles.
- Explicit workbench migration helpers for old saved state and policy-library artifacts.
- Checked-in policy fixture suite for escalation, fraud review, and support routing.
- Expected decision fixtures that lock policy rank, entity, action, acceptance, and score ranges.
- Append-oriented Node ledger persistence for newline-delimited decision entries.
- Dependency-free observer hooks for rule evaluation, decisions, ledger appends, and replay summaries.
- Holographic memory primitive for binding, superposing, and recalling vector symbols.
- Browser policy workbench with scenario selection, trainable predicates, import/export, smoke tests, and Playwright E2E coverage.
- Versioned `symtorch.scenario.v1` contracts for JSON-safe policy scenarios.
- Versioned `symtorch.trainingRun.v1` records for browser-side training history.
- WebGPU package with runtime detection, buffer pooling, explicit tensor residency, same-shape elementwise kernel prototypes, and a browser parity gate when WebGPU is available.

---

## Validation gates (release-manifest)

The `0.30.0` release manifest documents the “production contract corpus alpha” gates:

```text
pnpm install --frozen-lockfile
pnpm typecheck
pnpm test
pnpm playground:test
pnpm build
pnpm playground:build
pnpm exec tsx scripts/smoke-browser-playground.ts
pnpm playground:e2e
pnpm demo:policy-fixtures
pnpm demo:all
```

---

## Non-claims (important)

The version labels in this repository are engineering checkpoints for a private workspace.

SymTorch is:

- not a full PyTorch replacement
- not a full Prolog or Datalog engine
- not a production authorization system
- not a high-performance GPU tensor runtime yet
- not a durable database or retention system
- not an npm stability guarantee

---

## Monorepo layout

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
  policies/
```

---

## Documentation

- [Architecture](docs/architecture.md)
- [Public API surface](docs/api-surface.md)
- [30-second demo](docs/demo.md)
- [Browser playground prep](docs/browser-playground.md)
- [Holographic memory](docs/holographic-memory.md)
- [Persistence adapter alpha](docs/persistence.md)
- [Decision ledger persistence and replay](docs/decision-ledger-persistence.md)
- [Policy replay CLI](docs/policy-replay-cli.md)
- [Observability hooks](docs/observability.md)
- [Production hardening](docs/production-hardening.md)
- [Production roadmap](docs/production-roadmap.md)
- [Policy bundle runtime](docs/policy-bundle-runtime.md)
- [Policy fixture suite](docs/policy-fixtures.md)
- [Workbench migrations](docs/workbench-migrations.md)
- [Backend abstraction alpha](docs/backend-abstraction.md)
- [GPU backend plan](docs/gpu-backend-plan.md)
- [WebGPU residency prototype](docs/webgpu-residency.md)
- [WebGPU add kernel prototype](docs/webgpu-add-kernel.md)
- [WebGPU browser parity gate](docs/webgpu-browser-parity.md)
- [WebGPU same-shape elementwise kernels](docs/webgpu-elementwise-kernels.md)
- [Limitations](docs/limitations.md)
- [Changelog](CHANGELOG.md)
- [Production readiness alpha](docs/production-readiness.md)

---

## Status

SymTorch is early, active, and intentionally foundation-first.

The `0.30.0` workspace line is the **production contract corpus alpha** line: the ten production tracks are represented in machine-readable readiness reports, policy admission reports, HMAC signing metadata, typed domain contracts, decision trace snapshots, golden trace fixtures, expected decision fixtures, migrations, replay checks, operational event sinks, API stability snapshots, WebGPU parity metadata, and browser gates.

Durable database adapters, audited key management, hardened sandboxing, and WebGPU backend dispatch remain explicit future work.


---

## Scripts (from original README)

The following command blocks are preserved verbatim from the initial README for continuity.


```text
pnpm playground:browser
```


```text
pnpm install
pnpm typecheck
pnpm test
pnpm build
```


```text
pnpm example:linear
pnpm example:routing
pnpm example:trainable-routing
pnpm example:batch-routing
```


```text
pnpm demo:gradients
pnpm demo:rule
pnpm demo:ledger
pnpm demo:policy
pnpm demo:golden-policy
pnpm demo:policy-fixtures
pnpm demo:all
```
