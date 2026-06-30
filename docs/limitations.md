# Limitations

SymTorch is early and foundation-first. The current repository is useful for small differentiable-rule experiments and agent policy prototypes, but it is not a mature tensor framework or a full logic programming engine.

## Not A PyTorch Replacement

SymTorch has a PyTorch-inspired eager tensor/autograd core, but it does not aim to match mature tensor libraries feature-for-feature today. Current execution is CPU-first, with WebGPU still at the backend-planning stage.

## Not Full Prolog Or Datalog

The rule syntax has predicates, terms, variables, negation, and a Prolog-like shape. The current evaluator does not implement full unification, joins, quantifiers, backtracking, or relational grounding.

Today, predicate calls are resolved against a context through registered predicate implementations. Terms and variables are represented in the AST, but variable binding across fact relations is not yet implemented.

## Not A Production Authorization System

SymTorch can express explainable policies, but it is not a security or authorization framework. Do not use current rule scores as the only control for high-stakes access, finance, safety, or compliance decisions.

The `0.6.0` line adds production-shaped contracts, validation gates, and versioned ledger replay boundaries. It does not add deployment hardening, sandboxing, compliance controls, or autonomous authority.

## Not A High-Performance GPU Runtime Yet

The `@symtorch/webgpu` package currently focuses on detection and backend planning. GPU kernels, memory scheduling, kernel fusion, and CPU/GPU parity testing are future work.

The `0.15.0` WebGPU line can upload tensor data into GPU buffers, run same-shape `float32` binary and unary elementwise kernels, run scalar `sumAll`, compose `meanAll`, read results back explicitly, and run a browser parity gate for the explicit kernel set when WebGPU is available. It does not yet provide general tensor dispatch, broadcasting, axis reductions, autograd on GPU, or broad hardware coverage.

## Current Best Use Cases

- Explainable agent policy prototypes.
- Trainable fuzzy business rules.
- Browser or Node.js neuro-symbolic experiments.
- LLM-assisted rule drafting and validation.
- Small agent decision ledgers and entity ranking workflows.
