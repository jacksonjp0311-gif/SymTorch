# WebGPU Residency Prototype

SymTorch `0.9.0` introduces the first real WebGPU residency boundary in `@symtorch/webgpu`.

This is not a full GPU tensor backend yet. It provides the upload/readback infrastructure that future kernels will use.

## Public API

```ts
import { createWebGPUContext, detectWebGPU, requestWebGPUDevice } from "@symtorch/webgpu";

const status = await detectWebGPU();
if (!status.available) throw new Error(status.reason);

const device = await requestWebGPUDevice(status.adapter);
const context = createWebGPUContext(device);

const storage = context.uploadTensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
const values = await context.readTensor(storage);

context.disposeTensor(storage);
context.destroy();
```

## What Exists

- `WebGPUContext`
- `WebGPUTensorStorage`
- `uploadTensor()`
- `readTensor()`
- `BufferPool` integration
- default CPU to GPU parity tolerance constants
- CI-safe fake-device tests for upload/readback contract behavior

## What Does Not Exist Yet

- Tensor operations dispatched from `@symtorch/core` to WebGPU.
- WGSL compute kernels.
- Browser/hardware CPU to GPU parity gates.
- GPU autograd execution.
- Device-lost recovery.

## Next Kernel

The first kernel should be same-shape vector `add`.

That proves the full path:

```text
CPU data -> upload -> WebGPU storage -> WGSL kernel -> explicit readback -> CPU oracle comparison
```
