# Changelog

SymTorch currently uses "seal" checkpoints as internal engineering milestones. Package versions remain `0.1.0` until an npm-facing release is cut.

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
