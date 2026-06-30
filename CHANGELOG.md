# Changelog

SymTorch uses "seal" checkpoints as internal engineering milestones. `0.8.0` is the CPU backend dispatch alpha line, not a production stability claim.

## v0.8.0 CPU Backend Dispatch Alpha

- Added explicit `CpuStorage`, `GpuStorage`, and `TensorStorage` types to `@symtorch/core`.
- Added a `TensorBackend` interface with storage creation and synchronous read boundaries.
- Routed tensor construction through registered backend storage.
- Preserved CPU tensor behavior while making WebGPU placeholder tensors impossible to read silently.
- Added async `Tensor.read()` and `Tensor.toCPU()` APIs for future explicit readback.
- Added storage/no-hidden-sync tests.
- Added a GPU backend plan covering backend shape, storage, readback policy, and parity tolerances.
- Bumped private workspace package versions to `0.8.0`.

## v0.7.0 Backend Abstraction Alpha

- Added backend descriptors and registry helpers to `@symtorch/core`.
- Added default-device routing with `getDefaultDevice()`, `setDefaultDevice()`, and `withDefaultDevice()`.
- Registered `cpu` as the available correctness backend.
- Registered `webgpu` as a placeholder acceleration target before kernels are implemented.
- Added backend routing tests for device intent and scoped defaults.
- Added backend abstraction documentation.
- Bumped private workspace package versions to `0.7.0`.

## v0.6.0 Persistence Adapter Alpha

- Added the `symtorch.decisionLedger.v1` schema version for serialized ledger snapshots.
- Added `DecisionLedger.snapshot()` and `DecisionLedger.load()` for replayable persistence boundaries.
- Added `SerializedDecisionLedger`, `DecisionLedgerSink`, `serializeDecisionLedger()`, `loadDecisionLedger()`, and `isSerializedDecisionLedger()`.
- Added ledger restore tests that preserve deterministic decision IDs after reload.
- Raised the CI timeout to account for browser build, smoke, and Playwright gates.
- Bumped private workspace package versions to `0.6.0`.

## v0.5.0 Production Readiness Alpha

- Added the `symtorch.agentDecision.v1` schema version for serialized agent decisions.
- Added structural validators for serialized agent and entity decisions.
- Added a machine-readable release manifest for schema versions, validation gates, and non-claims.
- Added production-readiness documentation with explicit operational boundaries.
- Added manifest tests that keep package version and runtime schema constants aligned.
- Bumped private workspace package versions to `0.5.0`.

## v0.4.0 Vector-Symbolic Memory Alpha

- Added differentiable circular convolution and circular correlation to `@symtorch/core`.
- Added `bind` and `unbind` aliases for vector-symbolic binding workflows.
- Added `HolographicMemory` and `vectorSymbol` to `@symtorch/agent`.
- Added finite-difference gradient coverage for circular binding ops.
- Added executable holographic memory demo and included it in `demo:all`.
- Bumped private workspace package versions to `0.4.0`.

## v0.3.0 Trainable Policy Workbench Alpha

- Added the `symtorch.trainingRun.v1` training-run contract.
- Persisted and exported structured training history with playground state.
- Added before/after loss movement and sampled loss-history rendering in the browser workbench.
- Added training-run round-trip tests and browser E2E coverage.
- Bumped the private workspace package versions to `0.3.0`.

## v0.2.0 Policy Runtime Alpha

- Added the `symtorch.scenario.v1` scenario contract.
- Added scenario contract export, parse, and validation helpers.
- Added browser import support for standalone scenario JSON.
- Added bundled scenario contract diagnostics and tests.
- Bumped the private workspace package versions to `0.2.0`.

## v0.1.23 Agent Policy Scenario Seal

- Added bundled case escalation, fraud review, and support routing scenarios.
- Added a browser scenario selector and scenario-aware state export/import.
- Added scenario validation and browser interaction coverage.

## v0.1.22 Browser Interaction E2E Seal

- Added Playwright browser interaction coverage for the playground.
- Verified training, state export/import, invalid-rule recovery, and ledger recording through the UI.
- Added the browser interaction gate to CI.
- Added a CI timeout and Chromium headless-shell install for the browser interaction gate.

## v0.1.21 Browser Preview Smoke Seal

- Added a dependency-free Vite preview smoke test for the browser playground.
- Verified production HTML, CSS, and JS assets load from the built app.
- Added the browser preview smoke gate to CI.

## v0.1.20 Browser CI Gate Seal

- Added focused browser playground regression tests to CI.
- Added browser playground production build validation to CI.

## v0.1.19 Browser Training Dataset Seal

- Moved browser training examples into the playground model API.
- Added editable in-browser training examples for risk, approval, and label.
- Added coverage for caller-provided rule training examples.

## v0.1.18 Playground Import/Export Seal

- Added explicit browser playground state import/export.
- Added readable versioned state JSON for rule source, facts, and trained threshold.
- Added export round-trip coverage.

## v0.1.17 Browser State Persistence Seal

- Added versioned browser playground state serialization.
- Persisted editable rule source, entity facts, and trained threshold in local storage.
- Added state round-trip and invalid-state tests.

## v0.1.16 Browser Playground Test Seal

- Extracted browser playground policy behavior into a reusable model module.
- Added regression tests for rule validation, entity decisions, ledger replay, and browser-side rule training.
- Added `playground:test` for focused playground validation.

## v0.1.15 Browser Training Panel Seal

- Added an in-browser `ThresholdPredicate` training panel.
- Added threshold, score, and loss movement display.
- Preserved explanation trace output after browser-side training.

## v0.1.14 Browser Playground Prep Seal

- Added a minimal browser playground example.
- Added rule editing, fact sliders, validation diagnostics, entity decision ranking, and ledger replay.
- Added `playground:browser` and `playground:build` package scripts.
- Added browser playground documentation.

## v0.1.13 Demo Script Seal

- Added executable gradient sanity demo.
- Added executable trainable rule demo.
- Added executable agent ledger demo.
- Added `demo:gradients`, `demo:rule`, `demo:ledger`, and `demo:all` package scripts.

## v0.1.12 Public API Surface Seal

- Added `docs/api-surface.md`.
- Documented supported and experimental exports for each package.
- Linked public API boundaries from README and architecture docs.

## v0.1.11 Release Hygiene Seal

- Added public release hygiene docs.
- Moved seal history out of the README status flow.
- Clarified current package version policy.

## v0.1.10 Batch Authoring Throughput Seal

- Added `validatePrograms()` for validating many rule drafts in one call.
- Added stable IDs, per-draft results, and diagnostics for authoring workflows.

## v0.1.9 Predicate Binding Diagnostics Seal

- Added registry-aware validation for missing predicate bindings.
- Added `PredicateRegistry.has()` and `PredicateRegistry.names()`.

## v0.1.8 Rule Authoring Helpers Seal

- Added non-throwing `validateProgram()`.
- Added rule authoring validation examples.

## v0.1.7 Rule Parser Diagnostics Seal

- Added `RuleParseError`.
- Added line, column, snippet, and caret parser diagnostics.

## v0.1.6 Decision Ledger Seal

- Added append-only in-memory decision ledger.
- Added timestamped JSON-safe decision records with context snapshots.

## v0.1.5 Rule Evaluation Options Seal

- Added deterministic entity decision ordering.
- Added `topK`, `minScore`, and accepted-only filtering.

## v0.1.4 Entity Decision Batch Seal

- Added ranked, JSON-safe entity decision batches.
- Preserved traces for accepted and below-threshold candidates.

## v0.1.3 Agent Decision Contract Seal

- Added serialized agent decision contracts.
- Added `RuleAgent.decideTrace()`.

## v0.1.2 Explanation Schema Seal

- Added versioned, JSON-safe explanation traces.
- Added `decisionTrace()` and `serializeExplanation()`.

## v0.1.1 Gradient Correctness Seal

- Fixed axis-reduction gradients when `keepDims=false`.
- Added finite-difference coverage for reductions, neural losses, and LayerNorm.
- Added explanation-preserving rule-training tests.
