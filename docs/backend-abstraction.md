# Backend Abstraction Alpha

SymTorch `0.7.0` introduces a small backend registry in `@symtorch/core`.

This is the bridge toward WebGPU without moving faster than the math. CPU remains the correctness oracle. WebGPU is registered as a placeholder acceleration target so code can start carrying explicit device intent before kernels exist.

## Public API

```ts
import {
  getBackend,
  getDefaultDevice,
  listBackends,
  setDefaultDevice,
  tensor,
  withDefaultDevice
} from "@symtorch/core";

console.log(listBackends());
console.log(getBackend("cpu"));
console.log(getBackend("webgpu"));

const cpuTensor = tensor([1, 2, 3]);

const gpuMarkedTensor = withDefaultDevice("webgpu", () => {
  return tensor([1, 2, 3]);
});
```

## Current Behavior

- `cpu` is available and remains the execution backend.
- `webgpu` is registered as a placeholder.
- Tensor creation records device intent.
- Existing tensor operations still execute through the CPU implementation.
- No WebGPU kernels, buffer scheduling, or CPU/GPU parity gates exist yet.

## Why This Comes Before Kernels

A backend registry gives the project a stable place to add:

- upload/download boundaries
- backend capability checks
- CPU to GPU parity tests
- kernel dispatch
- readback observability
- future abortable GPU queues

The goal is to make acceleration modular instead of scattering WebGPU conditionals through tensor operations.
