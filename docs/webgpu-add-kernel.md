# WebGPU Add Kernel Prototype

SymTorch `0.10.0` introduces the first WGSL compute kernel in `@symtorch/webgpu`.

The kernel is intentionally narrow:

- `float32`
- same-shape tensors only
- elementwise add
- explicit upload and readback

```ts
import { createWebGPUContext, detectWebGPU, requestWebGPUDevice } from "@symtorch/webgpu";

const status = await detectWebGPU();
if (!status.available) throw new Error(status.reason);

const device = await requestWebGPUDevice(status.adapter);
const context = createWebGPUContext(device);

const left = context.uploadTensor([1, 2, 3], [3]);
const right = context.uploadTensor([10, 20, 30], [3]);
const result = context.add(left, right);

console.log(await context.readTensor(result));
```

## Why This Matters

This proves the first complete GPU compute path:

```text
CPU data -> upload -> WebGPU storage -> WGSL kernel -> explicit readback -> CPU oracle comparison
```

## Current Limits

- Not wired into `@symtorch/core` tensor dispatch.
- No broadcasting.
- No autograd on GPU.
- No browser hardware parity gate yet.
- Browser parity is covered by the optional WebGPU gate in [WebGPU Browser Parity Gate](webgpu-browser-parity.md).
- Fake-device tests cover API behavior in CI; real hardware parity is the next step.
