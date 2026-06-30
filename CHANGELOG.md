# Changelog

SymTorch uses "seal" checkpoints as internal engineering milestones. `0.23.0` is the policy workbench runtime line, not a production stability claim.

## v0.23.0 Policy Workbench Runtime

- Routed browser playground decisions through policy bundles and `createPolicyAgent()`.
- Added browser policy bundle export/import for `symtorch.policyBundle.v1`.
- Added a policy health panel with schema, hash, verification, rule count, predicate count, decision status, and replay status.
- Added a checked-in golden escalation policy at `examples/policies/escalation.policy.json`.
- Added `pnpm demo:golden-policy` to load the golden bundle, run decisions, record ledger entries, and replay-check them.
- Added tests for bundle import/export, tamper rejection, materialized-runtime parity, and browser bundle round-trip.
- Bumped private workspace package versions to `0.23.0`.

## v0.22.0 Policy Bundle Runtime

- Added `loadPolicyBundle()` to verify bundles and materialize executable rule programs, predicate registries, and engines.
- Added `createPolicyAgent()` for one-call bundle-to-agent runtime construction.
- Restores fact, threshold, and linear predicates from bundle metadata.
- Added `pnpm demo:policy` for bundle creation, decision, ledger recording, and replay verification.
- Added tests for executable bundles, tamper rejection, restored predicates, JSON-safe agent decisions, limits, and observer propagation.
- Bumped private workspace package versions to `0.22.0`.

## v0.21.0 Production Hardening

- Added core `SymTorchError`, `ResourceLimitError`, and `BackendExecutionError` taxonomy.
- Added tensor allocation limits and an explicit backend kernel dispatch boundary.
- Added logic runtime limits for rule source length, rule count, predicates per rule, and entity evaluation count.
- Added `PredicateEvaluationError`, policy bundle contracts, deterministic bundle hashing, and bundle validation.
- Added agent limits for entity batches and replay entries.
- Added `AppendFileDecisionLedgerSink` for newline-delimited append-oriented ledger persistence.
- Added hardening tests and documentation, and bumped private workspace package versions to `0.21.0`.

## v0.20.0 Observability Hooks

- Added dependency-free logic observer hooks for rule and grouped program evaluation.
- Added agent observer hooks for serialized decisions and ledger appends.
- Added replay observer summaries for `verifyDecisionLedgerReplay()`.
- Added tests for event payload shape, stable context keys, and replay summaries.
- Added observability documentation and bumped private workspace package versions to `0.20.0`.

## v0.19.0 Policy Replay CLI

- Added `pnpm ledger:replay` for replaying persisted decision ledgers from the command line.
- Supports explicit rule program text, fact-predicate registration, threshold, JSON output, `atol`, and `rtol`.
- Exits nonzero on decision drift so replay checks can run in CI or release gates.
- Added CLI tests for pass, fail, JSON output, and tolerance handling.
- Bumped private workspace package versions to `0.19.0`.

## v0.18.0 Batched Matmul and Replay Tolerance

- Added batched `matmul` support for rank-3 tensors in `@symtorch/core`.
- Added `matmul2D` and `batchedMatmul` internal dispatch paths with full gradient support.
- Added finite-difference gradient verification for batched matmul.
- Added `Dropout` layer to `@symtorch/nn` with inverted scaling and training/eval mode.
- Added `DecisionReplayTolerance` type and `tolerance` parameter to `verifyDecisionLedgerReplay()` in `@symtorch/agent`.
- Replay tolerance supports `atol` (absolute) and `rtol` (relative) thresholds for detecting float drift after predicate retraining.
- Expanded core tests with 6 new batched matmul test cases.
- Expanded nn tests with 5 new Dropout test cases.
- Expanded agent tests with 1 new replay tolerance test case.
- Updated API surface documentation for all affected packages.
- Bumped private workspace package versions to `0.18.0`.

## v0.17.0 Ledger Persistence and Replay

- Added browser-safe decision ledger replay verification with `verifyDecisionLedgerReplay()`.
- Added `DecisionReplayFn`, `DecisionReplayReport`, and `DecisionReplayMismatch` contracts.
- Added Node-only `FileDecisionLedgerSink` under `@symtorch/agent/node`.
- Expanded tests and the agent ledger demo to cover file persistence and replay verification.
- Bumped private workspace package versions to `0.17.0`.

## v0.16.0 WebGPU Stable LogSumExp

- Added stable scalar `logSumExpAll` support with `WEBGPU_LOG_SUM_EXP_ALL_WGSL`.
- Added `logSumExpAllTensor()` and `WebGPUContext.logSumExpAll()`.
- Tested normal and large-value inputs against CPU stable log-sum-exp oracles.
- Expanded browser parity coverage for the exported log-sum-exp shader.
- Bumped private workspace package versions to `0.16.0`.

## v0.15.0 WebGPU Mean Composition

- Added scalar-shaped WebGPU storage creation with `scalarTensor()` and `WebGPUContext.scalar()`.
- Added `meanAllTensor()` and `WebGPUContext.meanAll()` by composing `sumAll` with scalar division.
- Expanded fake-device and browser parity coverage for scalar reduction composition.
- Kept CPU as the correctness oracle and kept WebGPU execution explicit through `@symtorch/webgpu`.
- Bumped private workspace package versions to `0.15.0`.

## v0.14.0 WebGPU Sum Reduction Kernel

- Added explicit WebGPU `sumAll` reduction support that returns scalar-shaped storage.
- Added `WEBGPU_SUM_ALL_WGSL`, `WebGPUContext.sumAll()`, and `sumAllTensor()`.
- Expanded fake-device tests and browser parity coverage to include the first reduction kernel.
- Kept CPU as the correctness oracle and kept WebGPU execution explicit through `@symtorch/webgpu`.
- Bumped private workspace package versions to `0.14.0`.

## v0.13.0 WebGPU Unary Elementwise Kernels

- Added explicit same-shape WebGPU unary kernels for `abs`, `exp`, `log`, `relu`, `sigmoid`, `sqrt`, and `tanh`.
- Kept `@symtorch/webgpu` as an explicit upload/execute/readback package; these kernels are not wired into core tensor dispatch yet.
- Expanded fake-device kernel tests and browser WebGPU parity coverage for the elementwise kernel set.
- Bumped private workspace package versions to `0.13.0`.

## v0.12.0 WebGPU Same-Shape Elementwise Kernels

- Added same-shape `float32` WebGPU kernels for `sub`, `mul`, `div`, and `neg`.
- Added `WebGPUContext.sub()`, `mul()`, `div()`, and `neg()`.
- Added standalone `subTensors()`, `mulTensors()`, `divTensors()`, and `negTensor()`.
- Generalized the WebGPU compute pipeline cache by shader source.
- Expanded fake-device tests to compare the elementwise kernel set against CPU oracles.
- Added elementwise kernel documentation.
- Bumped private workspace package versions to `0.12.0`.

## v0.11.0 WebGPU Browser Parity Gate

- Added a Playwright browser parity gate for the same-shape WebGPU add kernel.
- The parity gate compiles the exported WGSL shader in a browser when WebGPU is available.
- The test compares GPU output against a CPU oracle using documented tolerance.
- The parity gate skips cleanly when `navigator.gpu` or an adapter is unavailable.
- Added WebGPU browser parity documentation.
- Bumped private workspace package versions to `0.11.0`.

## v0.10.0 WebGPU Add Kernel Prototype

- Added the first WGSL compute kernel: same-shape `float32` elementwise add.
- Added `WebGPUContext.add()` and standalone `addTensors()`.
- Added per-device compute pipeline caching for the add kernel.
- Added CI-safe fake-device tests that exercise upload, bind group dispatch, readback, and CPU oracle comparison.
- Added WebGPU add-kernel documentation.
- Bumped private workspace package versions to `0.10.0`.

## v0.9.0 WebGPU Residency Prototype

- Added `WebGPUContext` for explicit WebGPU tensor upload/readback workflows.
- Added `WebGPUTensorStorage`, `WebGPUDType`, and default parity tolerance constants.
- Added `uploadTensor()` and `readTensor()` primitives backed by `GPUBuffer` copy/readback operations.
- Integrated upload/dispose paths with `BufferPool`.
- Added CI-safe fake-device tests for WebGPU residency contracts.
- Added WebGPU residency documentation.
- Bumped private workspace package versions to `0.9.0`.

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
