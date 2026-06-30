# Public API Surface

This document defines the intended public API surface for the current `0.30.0` production contract corpus alpha line. SymTorch is still early, so this is a stability guide rather than a semantic-versioning guarantee.

## Stability Levels

- **Supported:** Intended for examples, tests, and early user code.
- **Experimental:** Exported and usable, but likely to change as the runtime hardens.
- **Internal by convention:** Exported today because packages are small, but not recommended as a stable integration point.

## `@symtorch/core`

Supported:

- `Tensor`
- `SymTorchError`, `ResourceLimitError`, `BackendExecutionError`
- `CoreRuntimeLimits`, `configureRuntimeLimits`, `getRuntimeLimits`
- `BackendDescriptor`, `BackendScope`, `BackendStatus`
- `BackendKernelName`, `runBackendKernel`
- `CpuStorage`, `GpuStorage`, `TensorStorage`, `TensorBackend`
- `registerBackend`, `getBackend`, `listBackends`, `getDefaultDevice`, `setDefaultDevice`, `withDefaultDevice`
- `tensor`, `fromArray`, `zeros`, `ones`, `full`, `randn`
- `add`, `sub`, `mul`, `div`, `neg`
- `exp`, `log`, `abs`, `pow`, `sqrt`, `tanh`, `clip`
- `relu`, `sigmoid`
- `sum`, `mean`, `max`
- `matmul` (rank-2 and rank-3+), `transpose`, `reshape`
- `circularConvolve`, `circularCorrelate`, `bind`, `unbind`
- `logsumexp`, `softmax`, `logSoftmax`
- `sizeOf`
- `DType`, `Device`, `TensorOptions`, `TensorLike`

Notes:

- CPU execution is the correctness oracle.
- `matmul` now supports rank-2 (matrix multiply) and rank-3+ (batched matrix multiply) inputs. Rank-1 inputs throw.
- Batched matmul dispatches internally through `batchedMatmul` with full gradient support.
- `Device` includes `"webgpu"`, but WebGPU tensor execution is not implemented yet.
- The `webgpu` backend descriptor is a placeholder for future dispatch and parity gates.
- WebGPU-placeholder tensors cannot be read synchronously; readback must stay explicit.

## `@symtorch/nn`

Supported:

- `Parameter`
- `Module`
- `Linear`, `Sequential`, `ReLU`, `Sigmoid`, `Dropout`, `LayerNorm`
- `mseLoss`
- `binaryCrossEntropy`, `binaryCrossEntropyWithLogits`
- `crossEntropyLoss`
- `Optimizer`, `SGD`, `Adam`

Notes:

- `Dropout` uses inverted scaling (`1 / (1 - p)`) during training. Set `dropout.training = false` for eval mode.
- `Dropout` with `p = 0` passes input through unchanged.
- Layers are intentionally minimal and eager.
- The optimizer API is small and may expand as parameter groups and schedules are added.

## `@symtorch/logic`

Supported:

- Rule AST and parser types: `Term`, `PredicateCall`, `RuleAst`, `RuleProgram`
- Parser diagnostics: `RuleParseError`
- Error taxonomy: `RuleValidationError`, `PredicateEvaluationError`
- Validation: `RuleValidationResult`, `RuleDiagnostic`, `BatchRuleValidationItem`, `RuleValidationOptions`, `RuleValidationInput`, `parseProgram`, `parseRule`, `validateProgram`, `validatePrograms`
- Predicate contracts: `Predicate`, `PredicateContext`, `PredicateResolver`, `PredicateResolution`
- Predicates and registry: `PredicateRegistry`, `FixedPredicate`, `FactPredicate`, `ThresholdPredicate`, `LinearPredicate`
- Evaluation: `FuzzyRuleEngine`, `RuleResult`, `AggregatedRuleResult`, `EntityRuleResult`, `RankedEntityResult`
- Observability: `LogicObserver`, `FuzzyRuleEngineOptions`, `RuleEvaluationEvent`, `ProgramEvaluationEvent`
- Runtime limits and policy bundles: `LogicRuntimeLimits`, `productionRuntimeLimits`, `POLICY_BUNDLE_SCHEMA_VERSION`, `SerializedPolicyBundle`, `PolicyBundlePredicate`, `PolicyBundleInput`, `LoadedPolicyBundle`, `LoadPolicyBundleOptions`, `createPolicyBundle`, `isSerializedPolicyBundle`, `verifyPolicyBundleHash`, `loadPolicyBundle`
- Production contracts: `PRODUCTION_READINESS_SCHEMA_VERSION`, `POLICY_ADMISSION_SCHEMA_VERSION`, `ProductionReadinessReport`, `ProductionReadinessTrack`, `ProductionTrackId`, `ProductionTrackStatus`, `PolicyAdmissionReport`, `PolicyAdmissionDiagnostic`, `getProductionReadinessReport`, `admitPolicyBundle`
- Policy admission alpha: `POLICY_BUNDLE_SIGNATURE_SCHEMA_VERSION`, `PolicyBundleSignature`, `SignedPolicyBundle`, `SignedPolicyBundleVerificationResult`, `TrustedPolicyKey`, `TrustedPolicyKeySet`, `signPolicyBundle`, `signPolicyBundleHmacSha256`, `verifySignedPolicyBundle`, `verifySignedPolicyBundleDetailed`, `verifySignedPolicyBundleHmacSha256`, `PolicyBundleSecurityAssessment`, `assessPolicyBundleSecurity`
- Domain contracts: `DOMAIN_CONTRACT_SCHEMA_VERSION`, `DomainContract`, `DomainEntity`, `DomainField`, `DomainFieldType`, `DomainValidationDiagnostic`, `DomainValidationResult`, `DomainGroundingRecord`, `DomainGroundingResult`, `createDomainContract`, `validateDomainContext`, `groundDomainEntities`
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
- `AGENT_DECISION_SCHEMA_VERSION`
- `AgentDecisionSchemaVersion`
- `DECISION_LEDGER_SCHEMA_VERSION`
- `DecisionLedgerSchemaVersion`
- `DECISION_TRACE_SNAPSHOT_SCHEMA_VERSION`
- `DecisionTraceSnapshotSchemaVersion`
- `SerializedAgentDecision`
- `SerializedEntityDecision`
- `SerializedDecisionLedger`
- `DecisionTraceSnapshot`
- `DecisionLedgerSink`
- `DecisionLedgerAppendSink`
- `DurableLedgerAdapterKind`
- `DurableLedgerAdapterDescriptor`
- `DecisionReplayFn`
- `DecisionReplayReport`
- `DecisionReplayMismatch`
- `DecisionReplayTolerance`
- `DecisionReplayError`
- `AgentObserver`
- `AgentDecisionEvent`
- `DecisionLedgerAppendEvent`
- `DecisionReplayEvent`
- `OperationalEvent`
- `OperationalSummary`
- `RuleAgentOptions`
- `PolicyAgentOptions`
- `AgentRuntimeLimits`
- `EntityDecisionOptions`
- `DecisionLedgerEntry`
- `DecisionLedger`
- `InMemoryOperationalSink`
- `WorkingMemory`
- `HolographicMemory`
- `HolographicMemoryTrace`
- `vectorSymbol`
- `RuleAgent`
- `isSerializedAgentDecision`
- `isSerializedEntityDecision`
- `isSerializedDecisionLedger`
- `serializeDecisionLedger`
- `describeDurableLedgerAdapter`
- `applyLedgerRetention`
- `createDecisionTraceSnapshot`
- `isDecisionTraceSnapshot`
- `loadDecisionLedger`
- `verifyDecisionLedgerReplay`
- `createPolicyAgent`

Node-only subpath:

- `@symtorch/agent/node`
- `FileDecisionLedgerSink`
- `AppendFileDecisionLedgerSink`

Notes:

- `RuleAgent.decide()` returns live tensor-backed results.
- `RuleAgent.decideTrace()` and entity trace methods return JSON-safe decision contracts.
- Serialized decisions are versioned as `symtorch.agentDecision.v1`.
- Serialized ledger snapshots are versioned as `symtorch.decisionLedger.v1`.
- `DecisionLedger` is in-memory, but snapshots can be persisted through `DecisionLedgerSink` adapters.
- `FileDecisionLedgerSink` is Node-only and intentionally exported from `@symtorch/agent/node` so browser bundles do not pull in `node:fs`.
- `verifyDecisionLedgerReplay()` now accepts an optional `tolerance` parameter with `atol` and `rtol` thresholds for detecting float drift after predicate retraining. Without tolerance, replay requires exact JSON match as before.
- `RuleAgent` accepts optional observer hooks for serialized decisions and ledger appends.
- `verifyDecisionLedgerReplay()` accepts an optional replay observer through the tolerance/options object.
- `HolographicMemory` is an experimental vector-symbolic memory primitive. It supports differentiable binding and approximate recall, not guaranteed cleanup memory.

## `@symtorch/webgpu`

Supported:

- `WebGPUStatus`
- `WebGPUDType`
- `WebGPUTensorStorage`
- `WebGPUTolerance`
- `WebGPUParityCase`
- `WebGPUBackendDispatchStatus`
- `WEBGPU_DEFAULT_TOLERANCE`
- `WEBGPU_PARITY_CASES`
- `getWebGPUBackendDispatchStatus`
- `WEBGPU_ADD_WGSL`
- `WEBGPU_SUB_WGSL`, `WEBGPU_MUL_WGSL`, `WEBGPU_DIV_WGSL`, `WEBGPU_NEG_WGSL`
- `WEBGPU_ABS_WGSL`, `WEBGPU_EXP_WGSL`, `WEBGPU_LOG_WGSL`, `WEBGPU_RELU_WGSL`, `WEBGPU_SIGMOID_WGSL`, `WEBGPU_SQRT_WGSL`, `WEBGPU_TANH_WGSL`
- `WEBGPU_SUM_ALL_WGSL`
- `WEBGPU_LOG_SUM_EXP_ALL_WGSL`
- `detectWebGPU`
- `requestWebGPUDevice`
- `WebGPUContext`
- `createWebGPUContext`
- `uploadTensor`
- `readTensor`
- `scalarTensor`
- `addTensors`
- `subTensors`, `mulTensors`, `divTensors`, `negTensor`
- `absTensor`, `expTensor`, `logTensor`, `reluTensor`, `sigmoidTensor`, `sqrtTensor`, `tanhTensor`
- `sumAllTensor`
- `meanAllTensor`
- `logSumExpAllTensor`
- `BufferPool`

Experimental:

- Tensor residency APIs are prototypes until real kernels and hardware parity gates exist.

Notes:

- WebGPU support is currently capability detection, buffer pooling, explicit tensor upload/readback, a same-shape elementwise kernel set, scalar `sumAll`, composed `meanAll`, and stable scalar `logSumExpAll`.
- Same-shape elementwise `add`, `sub`, `mul`, `div`, `neg`, `abs`, `exp`, `log`, `relu`, `sigmoid`, `sqrt`, and `tanh` have prototype kernels. `sumAll` is the first reduction prototype; `meanAll` composes `sumAll` with scalar division; `logSumExpAll` is a scalar stability primitive for future softmax/loss work. Broadcasting, axis reductions, wider tensor kernels, and core dispatch integration are future work.
- The browser parity gate exercises the explicit kernel set when WebGPU is available and skips otherwise.

## Internal By Convention

The repository currently uses single-file package entry points. Helper functions not listed above should be treated as implementation details even if they appear in generated declarations later. New public APIs should be added here when they become part of the supported surface.
