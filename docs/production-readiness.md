# Production Readiness Alpha

SymTorch `0.18.0` is the batched matmul and replay tolerance line. This does not mean the project is production-ready. It means the repository now has the contract discipline, validation gates, replay boundaries, explicit backend intent, no-hidden-sync storage policy, initial WebGPU upload/readback boundary, a broader explicit GPU kernel set, scalar reduction composition, a first numerical-stability primitive, file-backed ledger snapshots for Node, replay verification with configurable tolerance, batched tensor operations, and a neural regularization layer needed to keep moving toward production without blurring research claims into deployment claims.

## What Is Production-Shaped

- Versioned explanation traces via `symtorch.explanation.v1`.
- Versioned agent decisions via `symtorch.agentDecision.v1`.
- Versioned decision ledger snapshots via `symtorch.decisionLedger.v1`.
- Versioned browser playground state via `symtorch.playground.v1`.
- Versioned scenario contracts via `symtorch.scenario.v1`.
- Versioned training runs via `symtorch.trainingRun.v1`.
- JSON-safe decision and explanation payloads.
- Entity decision batches and replayable in-memory ledger snapshots.
- Node filesystem ledger persistence through `FileDecisionLedgerSink`.
- Decision replay verification for detecting policy drift against recorded ledger entries.
- Configurable replay tolerance (`atol`, `rtol`) for float-drift detection after predicate retraining.
- Batched `matmul` with gradient support for rank-3+ tensors.
- `Dropout` layer with inverted scaling for neural regularization.
- Local browser build, smoke, and Playwright interaction gates.
- Executable demos for gradients, trainable rules, agent ledgers, and holographic memory.
- Backend descriptors for CPU and future WebGPU dispatch.
- Explicit tensor storage types and readback boundaries.
- WebGPU tensor upload/readback prototype in `@symtorch/webgpu`.
- Same-shape WebGPU elementwise kernel prototypes.
- Scalar WebGPU `sumAll` reduction prototype.
- Composed scalar WebGPU `meanAll`.
- Stable scalar WebGPU `logSumExpAll`.
- Browser parity gate for the explicit kernel set when WebGPU is available.

## What Is Still Not Production-Ready

- WebGPU kernels are still narrow: same-shape `float32` elementwise, scalar `sumAll`, composed `meanAll`, and scalar `logSumExpAll` only.
- WebGPU is a registered placeholder backend, not an execution backend.
- The decision ledger is still an in-memory runtime primitive; the file sink persists snapshots but is not a database or retention system.
- Rule evaluation does not yet implement full unification, joins, quantifiers, or relational grounding.
- There is no security sandbox for executing untrusted rule sources.
- There are no persistence adapters beyond the Node file sink, no migration runners, or service-level observability hooks.
- Package versions are private workspace checkpoints, not npm stability guarantees.
- `Dropout` does not save or restore its training/eval mode across serialization boundaries.

## Required Local Gate

```powershell
pnpm install --frozen-lockfile
pnpm typecheck
pnpm test
pnpm playground:test
pnpm build
pnpm playground:build
pnpm exec tsx scripts/smoke-browser-playground.ts
pnpm playground:e2e
pnpm demo:all
```

## Release Manifest

The machine-readable release manifest lives at [release-manifest.json](release-manifest.json). Tests assert that its version and schema strings match exported runtime constants. This keeps public-facing documentation, browser contracts, ledger contracts, and agent contracts aligned as the system evolves.

## Operator Guidance

Treat SymTorch decisions as explainable policy signals, not autonomous authority. Downstream systems should record the serialized decision, explanation trace, input context, model or rule version, and any human or service action that consumed the signal.
