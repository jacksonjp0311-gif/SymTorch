# SymTorch

SymTorch is a TypeScript-first differentiable tensor library and neuro-symbolic rule runtime for browser, Node.js, and edge-style JavaScript environments.

The project starts with a correct eager CPU tensor/autograd core, then layers on neural-network modules, a WebGPU backend, differentiable fuzzy-logic rules, explanations, and agent loops.

## Current milestone

- Eager `Tensor` API with scalar reverse-mode autograd.
- CPU `float32` storage through typed arrays.
- PyTorch-like creation and ops: `tensor`, `zeros`, `ones`, `randn`, `add`, `sub`, `mul`, `div`, `matmul`, `sum`, `mean`, `relu`, `sigmoid`, `log`, `exp`, `softmax`, `logsumexp`, `reshape`, `transpose`.
- `nn.Module`, `Parameter`, `Linear`, `Sequential`, `SGD`, `Adam`, and `mseLoss`.
- A compact Prolog-like rule compiler for differentiable fuzzy rules with explanation traces.
- Trainable fixed, threshold, and linear predicates for neuro-symbolic rules.
- `RuleTrainer` for fitting differentiable rules against labeled examples.
- WebGPU package shell with capability detection and backend-planning docs.

## Install

```powershell
pnpm install
pnpm typecheck
pnpm test
```

## Examples

```powershell
pnpm example:linear
pnpm example:routing
pnpm example:trainable-routing
```

## Vision

SymTorch is built for neuro-symbolic agents: humans write readable rules, SymTorch compiles them into differentiable tensor programs, and training adjusts learnable predicates while preserving traces that explain why a decision fired.
