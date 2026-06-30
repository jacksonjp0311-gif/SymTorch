# Limitations

SymTorch is early and foundation-first. The current repository is useful for small differentiable-rule experiments and agent policy prototypes, but it is not a mature tensor framework or a full logic programming engine.

## Not A PyTorch Replacement

SymTorch has a PyTorch-inspired eager tensor/autograd core, but it does not aim to match mature tensor libraries feature-by-feature today. Current execution is CPU-first, with WebGPU still at the backend-planning stage.

The `0.18.0` line adds batched `matmul` support for rank-3 tensors and a `Dropout` layer. It does not add conv2d, pooling, embedding, attention, or parameter groups.

## Not Full Prolog Or Datalog

The rule syntax has predicates, terms, variables, negation, and a Prolog-like shape. The current evaluator does not implement full unification, joins, quantifiers, backtracking, or relational grounding.

Today, predicate calls are resolved against a context through registered predicate implementations. Terms and variables are represented in the AST, but variable binding across fact relations is not yet implemented.

## Not A Production Authorization System

SymTorch can express explainable policies, but it is not a security or authorization framework. Do not use current rule scores as the only control for high-stakes access, finance, safety, or compliance decisions.

The `0.6.0` line adds production-shaped contracts, validation gates, and versioned ledger replay boundaries. It does not add deployment hardening, sandboxing, compliance controls, or autonomous authority.

## Not A High-Performance GPU Runtime Yet

The `@symtorch/webgpu` package currently focuses on detection and backend planning. GPU kernels, memory scheduling, kernel fusion, and CPU/GPU parity testing are future work.

The `0.16.0` WebGPU line can upload tensor data into GPU buffers, run same-shape `float32` binary and unary elementwise kernels, run scalar `sumAll`, compose `meanAll`, run stable scalar `logSumExpAll`, read results back explicitly, and run a browser parity gate for the explicit kernel set when WebGPU is available. It does not yet provide general tensor dispatch, broadcasting, axis reductions, autograd on GPU, full softmax, or broad hardware coverage.

The `0.17.0` ledger line adds a Node filesystem sink and replay verification, but it is not a database, consensus log, authorization audit system, or retention policy. Applications still need their own storage durability, access control, privacy, and lifecycle rules.

The `0.18.0` replay tolerance line adds configurable `atol` and `rtol` thresholds to `verifyDecisionLedgerReplay()`. This helps detect float drift after predicate retraining without false positives from rounding. It does not add selective replay, date-range filtering, or replay with partial context restoration.

The `0.19.0` policy replay CLI adds a command-line drift gate for fact-predicate policies. It does not load arbitrary policy modules, manage policy versions, sandbox untrusted rule source, or provide a policy registry.

The `0.20.0` observability hooks line adds synchronous operator events for rule evaluation, agent decisions, ledger appends, and replay summaries. It does not provide a metrics backend, distributed tracing implementation, durable audit log, or hook isolation; hook errors currently propagate to the caller.

The `0.21.0` production hardening line adds opt-in runtime limits, typed errors, policy bundle hashing, append file ledgers, and a backend dispatch boundary. It does not make policy bundles cryptographically signed, make append files transactional databases, sandbox untrusted rules, or route core tensor execution through WebGPU kernels yet.

The `0.27.0` expected decision fixture line makes local bundles executable in Node, saveable/loadable through a versioned browser policy library, explicitly migratable for supported local artifact shapes, and continuously verified against expected fixture decisions. It does not add a remote policy registry, cryptographic signing, staged promotion workflow, durable multi-user storage, or a general migration runner for arbitrary future schemas.

## Current Best Use Cases

- Explainable agent policy prototypes.
- Trainable fuzzy business rules.
- Browser or Node.js neuro-symbolic experiments.
- LLM-assisted rule drafting and validation.
- Small agent decision ledgers and entity ranking workflows.
- Batched neural inference with dropout regularization.
