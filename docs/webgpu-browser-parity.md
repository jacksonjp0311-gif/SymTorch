# WebGPU Browser Parity Gate

SymTorch `0.11.0` added the first browser-side WebGPU parity gate. SymTorch `0.13.0` expands it to the explicit same-shape elementwise kernel set.

The gate runs same-shape `float32` WGSL kernels in a real browser environment when WebGPU is available. If the browser or runner does not expose `navigator.gpu`, the test skips cleanly and reports the reason.

## What It Proves

- The shader exported by `@symtorch/webgpu` can compile in a browser WebGPU runtime.
- Uploaded CPU data can be processed by binary and unary elementwise kernels.
- Explicit readback can be compared against a CPU oracle.
- Tolerance uses `WEBGPU_DEFAULT_TOLERANCE`.

## What It Does Not Prove

- General GPU tensor dispatch.
- Broadcasting.
- GPU autograd.
- Performance.
- Hardware coverage across vendors.

## Gate

The parity test lives in:

```text
tests/playground/webgpu-add-parity.e2e.ts
```

It is included in:

```powershell
pnpm playground:e2e
```
