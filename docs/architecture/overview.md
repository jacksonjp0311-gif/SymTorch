# Architecture Overview

SymTorch is organized as a TypeScript monorepo with narrow package boundaries.

## Packages

- `@symtorch/core`: eager tensors, CPU reference kernels, shape utilities, and reverse-mode autograd.
- `@symtorch/nn`: neural-network modules, parameters, layers, optimizers, and losses.
- `@symtorch/logic`: Prolog-like rules compiled to differentiable fuzzy tensor expressions with explanations.
- `@symtorch/webgpu`: WebGPU capability detection and the future accelerated backend.
- `@symtorch/agent`: minimal observation -> decide -> learn agent loop primitives.

## Invariants

- CPU is the correctness oracle.
- WebGPU is an acceleration path, never a required runtime.
- Tensor operations are eager and dynamic first.
- Differentiable ops expose vector-Jacobian-product backward rules.
- Rule execution separates numeric scores from human-readable explanations.

## Roadmap

1. Harden CPU tensors and autograd with gradient checks.
2. Expand neural layers and stable losses.
3. Add WebGPU kernels with CPU parity tests.
4. Expand the rule compiler with typed domains, quantifiers, and learnable predicates.
5. Build browser and Node.js agent demos.

