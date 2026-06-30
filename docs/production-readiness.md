# Production Readiness Alpha

SymTorch `0.6.0` is the persistence-adapter alpha line. This does not mean the project is production-ready. It means the repository now has the contract discipline, validation gates, and replay boundaries needed to keep moving toward production without blurring research claims into deployment claims.

## What Is Production-Shaped

- Versioned explanation traces via `symtorch.explanation.v1`.
- Versioned agent decisions via `symtorch.agentDecision.v1`.
- Versioned decision ledger snapshots via `symtorch.decisionLedger.v1`.
- Versioned browser playground state via `symtorch.playground.v1`.
- Versioned scenario contracts via `symtorch.scenario.v1`.
- Versioned training runs via `symtorch.trainingRun.v1`.
- JSON-safe decision and explanation payloads.
- Entity decision batches and replayable in-memory ledger snapshots.
- Local browser build, smoke, and Playwright interaction gates.
- Executable demos for gradients, trainable rules, agent ledgers, and holographic memory.

## What Is Still Not Production-Ready

- WebGPU tensor kernels are not implemented yet.
- The decision ledger is in-memory only, with a versioned snapshot/load boundary for storage adapters.
- Rule evaluation does not yet implement full unification, joins, quantifiers, or relational grounding.
- There is no security sandbox for executing untrusted rule sources.
- There are no persistence adapters, migration runners, or service-level observability hooks.
- Package versions are private workspace checkpoints, not npm stability guarantees.

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
