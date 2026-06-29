# Public API Surface

This document defines the intended public API surface for the current `0.1.0` package line. SymTorch is still early, so this is a stability guide rather than a semantic-versioning guarantee.

## Stability Levels

- **Supported:** Intended for examples, tests, and early user code.
- **Experimental:** Exported and usable, but likely to change as the runtime hardens.
- **Internal by convention:** Exported today because packages are small, but not recommended as a stable integration point.

## `@symtorch/core`

Supported:

- `Tensor`
- `tensor`, `fromArray`, `zeros`, `ones`, `full`, `randn`
- `add`, `sub`, `mul`, `div`, `neg`
- `exp`, `log`, `abs`, `pow`, `sqrt`, `tanh`, `clip`
- `relu`, `sigmoid`
- `sum`, `mean`, `max`
- `matmul`, `transpose`, `reshape`
- `logsumexp`, `softmax`, `logSoftmax`
- `sizeOf`
- `DType`, `Device`, `TensorOptions`, `TensorLike`

Notes:

- CPU execution is the correctness oracle.
- `Device` includes `"webgpu"`, but WebGPU tensor execution is not implemented yet.

## `@symtorch/nn`

Supported:

- `Parameter`
- `Module`
- `Linear`, `Sequential`, `ReLU`, `Sigmoid`, `LayerNorm`
- `mseLoss`
- `binaryCrossEntropy`, `binaryCrossEntropyWithLogits`
- `crossEntropyLoss`
- `Optimizer`, `SGD`, `Adam`

Notes:

- Layers are intentionally minimal and eager.
- The optimizer API is small and may expand as parameter groups and schedules are added.

## `@symtorch/logic`

Supported:

- Rule AST and parser types: `Term`, `PredicateCall`, `RuleAst`, `RuleProgram`
- Parser diagnostics: `RuleParseError`
- Validation: `RuleValidationResult`, `RuleDiagnostic`, `BatchRuleValidationItem`, `RuleValidationOptions`, `RuleValidationInput`, `parseProgram`, `parseRule`, `validateProgram`, `validatePrograms`
- Predicate contracts: `Predicate`, `PredicateContext`, `PredicateResolver`, `PredicateResolution`
- Predicates and registry: `PredicateRegistry`, `FixedPredicate`, `FactPredicate`, `ThresholdPredicate`, `LinearPredicate`
- Evaluation: `FuzzyRuleEngine`, `RuleResult`, `AggregatedRuleResult`, `EntityRuleResult`, `RankedEntityResult`
- Training: `LabeledRuleExample`, `RuleTrainerOptions`, `RuleTrainerHistoryItem`, `RuleTrainerResult`, `RuleTrainer`
- Explanations: `EXPLANATION_SCHEMA_VERSION`, `ExplanationSchemaVersion`, `RuleExplanation`, `PredicateTrace`, `AggregatedRuleExplanation`, `SerializedPredicateTrace`, `SerializedRuleExplanation`, `SerializedAggregatedRuleExplanation`, `SerializedExplanation`
- Rendering and serialization: `renderRuleExplanation`, `renderAggregatedExplanation`, `decisionCard`, `decisionTrace`, `serializeExplanation`
- Fuzzy ops: `productAnd`, `probabilisticOr`, `fuzzyNot`
- Formatting: `formatPredicate`

Experimental:

- Parser and validation diagnostics may gain additional fields and diagnostic codes.
- Fuzzy operations may expand beyond product t-norm and probabilistic OR.

Current semantic limit:

- Terms and variables are represented in the AST, but evaluation does not yet perform full unification, joins, or relational grounding.

## `@symtorch/agent`

Supported:

- `Observation`
- `AgentDecision`
- `SerializedAgentDecision`
- `SerializedEntityDecision`
- `EntityDecisionOptions`
- `DecisionLedgerEntry`
- `DecisionLedger`
- `WorkingMemory`
- `RuleAgent`

Notes:

- `RuleAgent.decide()` returns live tensor-backed results.
- `RuleAgent.decideTrace()` and entity trace methods return JSON-safe decision contracts.
- `DecisionLedger` is in-memory only. It is an audit primitive, not persistent storage.

## `@symtorch/webgpu`

Supported:

- `WebGPUStatus`
- `detectWebGPU`
- `requestWebGPUDevice`

Experimental:

- `BufferPool`

Notes:

- WebGPU support is currently capability detection and backend planning.
- Tensor kernels, buffer scheduling, and CPU/GPU parity tests are future work.

## Internal By Convention

The repository currently uses single-file package entry points. Helper functions not listed above should be treated as implementation details even if they appear in generated declarations later. New public APIs should be added here when they become part of the supported surface.
