# WebGPU Same-Shape Elementwise Kernels

SymTorch `0.13.0` expands the explicit WebGPU kernel set.

Supported prototype kernels:

- `add`
- `sub`
- `mul`
- `div`
- `neg`
- `abs`
- `exp`
- `log`
- `relu`
- `sigmoid`
- `sqrt`
- `tanh`

All kernels are intentionally narrow:

- `float32`
- same-shape tensors only
- explicit upload/readback through `@symtorch/webgpu`
- not wired into `@symtorch/core` tensor dispatch yet

```ts
const left = context.uploadTensor([2, -4, 9, 8], [4]);
const right = context.uploadTensor([1, 2, -3, 4], [4]);

const diff = context.sub(left, right);
const product = context.mul(left, right);
const quotient = context.div(left, right);
const negated = context.neg(left);
const activated = context.relu(left);
const probabilities = context.sigmoid(left);
```

## Current Gate

The kernel set is covered by fake-device tests in CI. The browser parity gate covers the explicit elementwise set when WebGPU is available and skips cleanly on runners without WebGPU.
