# SymTorch Architecture

SymTorch is a TypeScript-native differentiable rule engine for explainable agents.

```text
readable rule
  -> fuzzy tensor execution
  -> trainable predicate
  -> explanation trace
  -> serialized agent decision
  -> entity batch / ledger / validation loop
```

## Package Boundaries

- `@symtorch/core`: eager tensors, CPU execution, shape utilities, and reverse-mode autograd.
- `@symtorch/nn`: modules, parameters, layers, optimizers, and losses.
- `@symtorch/logic`: rule parsing, fuzzy evaluation, predicates, explanations, training, and validation.
- `@symtorch/agent`: working memory, serialized decisions, entity batches, filtering, and decision ledger.
- `@symtorch/webgpu`: WebGPU capability detection and future accelerated backend work.

See [Public API Surface](api-surface.md) for the currently supported exports and experimental boundaries.

## Runtime Flow

1. A developer writes a readable rule such as `escalate(X) :- high_risk(X), not approved(X).`
2. `RuleProgram` parses it into an AST.
3. `FuzzyRuleEngine` evaluates each predicate through a resolver or `PredicateRegistry`.
4. Predicate scores are combined with fuzzy logic operations backed by `@symtorch/core` tensors.
5. Learnable predicates expose `Parameter` objects and can be trained with optimizers from `@symtorch/nn`.
6. Evaluation returns both a tensor score and structured explanation data.
7. `RuleAgent` wraps grouped results into serialized decision contracts.
8. Entity decisions can be ranked, filtered, and recorded into an append-only in-memory ledger.

## Correctness Invariants

- CPU behavior is the correctness oracle.
- WebGPU is an acceleration path, not a required runtime.
- Public decision and explanation data must be JSON-safe.
- Runtime contract schema versions must be listed in the release manifest and covered by tests.
- Rule authoring APIs should support both throwing and non-throwing validation paths.
- Tests should protect mathematical behavior and agent-facing contracts.

## Near-Term Direction

- Browser policy playground.
- Browser playground training panel.
- Public API surface documentation.
- Production-readiness contract manifest.
- Typed domains and guarded grounding.
- Broader gradient checks and tensor op coverage.
- WebGPU kernels with CPU parity tests.
