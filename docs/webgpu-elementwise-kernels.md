# WebGPU Same-Shape Elementwise Kernels

SymTorch `0.12.0` expands the explicit WebGPU kernel set.

Supported prototype kernels:

- `add`
- `sub`
- `mul`
- `div`
- `neg`

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
```

## Current Gate

The kernel set is covered by fake-device tests in CI. The browser parity gate currently targets the add kernel; the next step is broadening browser parity to the full same-shape elementwise set.
