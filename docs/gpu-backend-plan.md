# GPU Backend Plan

This note locks the Phase 0 decisions for SymTorch's future WebGPU backend.

## Backend Interface

`@symtorch/core` owns the backend registry. A backend has:

- a public descriptor: `id`, `name`, `status`, `description`
- `createStorage(data, shape, dtype)`
- `readSync(storage)`

CPU is the available backend and correctness oracle. WebGPU is registered as a placeholder acceleration target until real residency and kernels exist.

## Storage Model

Tensor storage is explicit:

- `CpuStorage`: `Float32Array` data plus shape and dtype metadata
- `GpuStorage`: shape, dtype, byte length, and placeholder status

The current line uses single residency. CPU tensors have CPU storage. WebGPU-marked tensors have placeholder GPU storage and cannot be read as CPU data.

Future GPU residency should replace placeholder `GpuStorage` with a real object containing:

- `GPUDevice`
- `GPUBuffer`
- byte length
- usage flags
- dtype and shape
- optional debug label

## No-Surprises Sync Policy

SymTorch should not silently read GPU tensors back to CPU.

Synchronous APIs such as `toArray()`, `item()`, and internal CPU math require CPU-resident tensors. GPU readback must be explicit through async APIs such as `tensor.read()` or `tensor.toCPU()` once real WebGPU storage exists.

This protects training loops from accidental stalls and makes browser performance easier to reason about.

## Parity Tolerance Policy

CPU remains the numerical oracle. Future CPU to GPU parity tests should use:

- default `atol = 1e-5`
- default `rtol = 1e-4`
- softer tolerances for reduction-heavy ops such as softmax and logsumexp when needed

Every tolerance exception should be documented next to the parity test that needs it.

## Phase 1 Entry Criteria

Before implementing WebGPU kernels, the repo should keep passing:

- backend descriptor and device routing tests
- explicit readback/no-hidden-sync tests
- existing CPU autograd and finite-difference tests
- browser playground smoke and E2E gates

The first WebGPU kernel should be same-shape vector `add`, with a parity test that uploads CPU data, executes the kernel, explicitly reads back, and compares against CPU output.

## Phase 1 Progress

The `0.9.0` line adds the first residency prototype:

- `WebGPUContext`
- explicit tensor upload
- explicit tensor readback
- `BufferPool` integration
- fake-device tests for CI-safe upload/readback contract coverage

The `0.10.0` line adds the first WGSL kernel:

- same-shape `float32` add
- explicit upload and readback
- fake-device CI coverage

The next missing piece is a real browser/hardware parity gate for the add kernel.
