# Changelog

SymTorch currently uses "seal" checkpoints as internal engineering milestones. Package versions remain `0.1.0` until an npm-facing release is cut.

## v0.1.22 Browser Interaction E2E Seal

- Added Playwright browser interaction coverage for the playground.
- Verified training, state export/import, invalid-rule recovery, and ledger recording through the UI.
- Added the browser interaction gate to CI.
- Used the official Playwright CI image to avoid slow browser dependency installation.

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
