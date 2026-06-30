import { expect, test } from "@playwright/test";
import {
  WEBGPU_ABS_WGSL,
  WEBGPU_ADD_WGSL,
  WEBGPU_DEFAULT_TOLERANCE,
  WEBGPU_DIV_WGSL,
  WEBGPU_EXP_WGSL,
  WEBGPU_LOG_SUM_EXP_ALL_WGSL,
  WEBGPU_LOG_WGSL,
  WEBGPU_MUL_WGSL,
  WEBGPU_NEG_WGSL,
  WEBGPU_RELU_WGSL,
  WEBGPU_SIGMOID_WGSL,
  WEBGPU_SQRT_WGSL,
  WEBGPU_SUB_WGSL,
  WEBGPU_SUM_ALL_WGSL,
  WEBGPU_TANH_WGSL
} from "@symtorch/webgpu";

test("webgpu explicit kernels match CPU oracles when WebGPU is available", async ({ page }) => {
  await page.goto("/");
  const available = await page.evaluate(async () => {
    const gpu = navigator.gpu;
    if (!gpu) return { ok: false, reason: "navigator.gpu is not available." };
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: "No WebGPU adapter is available." };
    const device = await adapter.requestDevice();
    device.destroy();
    return { ok: true, reason: "" };
  });

  test.skip(!available.ok, `WebGPU parity skipped: ${available.reason}`);

  const cases = [
    { name: "add", kind: "binary", shader: WEBGPU_ADD_WGSL },
    { name: "sub", kind: "binary", shader: WEBGPU_SUB_WGSL },
    { name: "mul", kind: "binary", shader: WEBGPU_MUL_WGSL },
    { name: "div", kind: "binary", shader: WEBGPU_DIV_WGSL },
    { name: "neg", kind: "unary", shader: WEBGPU_NEG_WGSL },
    { name: "abs", kind: "unary", shader: WEBGPU_ABS_WGSL },
    { name: "exp", kind: "unary", shader: WEBGPU_EXP_WGSL },
    { name: "log", kind: "unary-positive", shader: WEBGPU_LOG_WGSL },
    { name: "relu", kind: "unary", shader: WEBGPU_RELU_WGSL },
    { name: "sigmoid", kind: "unary", shader: WEBGPU_SIGMOID_WGSL },
    { name: "sqrt", kind: "unary-positive", shader: WEBGPU_SQRT_WGSL },
    { name: "tanh", kind: "unary", shader: WEBGPU_TANH_WGSL },
    { name: "sumAll", kind: "reduction", shader: WEBGPU_SUM_ALL_WGSL },
    { name: "meanAll", kind: "reduction", shader: WEBGPU_SUM_ALL_WGSL },
    { name: "logSumExpAll", kind: "reduction", shader: WEBGPU_LOG_SUM_EXP_ALL_WGSL }
  ] as const;

  for (const kernel of cases) {
    const result = await page.evaluate(async ({ kernel, tolerance }) => {
    const adapter = await navigator.gpu!.requestAdapter();
    if (!adapter) throw new Error("Expected a WebGPU adapter after availability check.");
    const device = await adapter.requestDevice();
    const left = new Float32Array([1, -2, 3.5, 4, 9, -8, 0.25, 12]);
    const positive = new Float32Array([0.25, 1, 2, 4, 9, 16, 0.5, 25]);
    const right = new Float32Array([0.5, 2, -1.5, 8, -4, 3, 0.75, -10]);
    const input = kernel.kind === "unary-positive" ? positive : left;
    const expected = expectedValues(kernel.name, input, right);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const outputByteLength = kernel.kind === "reduction" ? Float32Array.BYTES_PER_ELEMENT : input.byteLength;
    const leftBuffer = device.createBuffer({ size: input.byteLength, usage });
    const rightBuffer = device.createBuffer({ size: input.byteLength, usage });
    const outBuffer = device.createBuffer({ size: outputByteLength, usage });
    device.queue.writeBuffer(leftBuffer, 0, input);
    device.queue.writeBuffer(rightBuffer, 0, right);

    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: kernel.shader }),
        entryPoint: "main"
      }
    });
    const entries: GPUBindGroupEntry[] = kernel.kind === "binary"
      ? [
        { binding: 0, resource: { buffer: leftBuffer } },
        { binding: 1, resource: { buffer: rightBuffer } },
        { binding: 2, resource: { buffer: outBuffer } }
      ]
      : [
        { binding: 0, resource: { buffer: leftBuffer } },
        { binding: 1, resource: { buffer: outBuffer } }
      ];
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();

    const readback = device.createBuffer({
      size: outputByteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    encoder.copyBufferToBuffer(outBuffer, 0, readback, 0, outputByteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
    if (kernel.name === "meanAll") {
      actual[0] = (actual[0] ?? 0) / input.length;
    }
    readback.unmap();
    readback.destroy();
    leftBuffer.destroy();
    rightBuffer.destroy();
    outBuffer.destroy();
    device.destroy();

    const maxError = actual.reduce((max, value, index) => Math.max(max, Math.abs(value - (expected[index] ?? 0))), 0);
    const ok = actual.every((value, index) => {
      const target = expected[index] ?? 0;
      return Math.abs(value - target) <= tolerance.atol + tolerance.rtol * Math.abs(target);
    });
    return { name: kernel.name, actual, expected, maxError, ok };

    function expectedValues(name: string, left: Float32Array, right: Float32Array): number[] {
      if (name === "sumAll") return [Array.from(left).reduce((total, value) => total + value, 0)];
      if (name === "meanAll") return [Array.from(left).reduce((total, value) => total + value, 0) / left.length];
      if (name === "logSumExpAll") {
        const max = Math.max(...Array.from(left));
        return [Math.log(Array.from(left).reduce((total, value) => total + Math.exp(value - max), 0)) + max];
      }
      return Array.from(left, (value, index) => {
        const b = right[index] ?? 0;
        if (name === "add") return value + b;
        if (name === "sub") return value - b;
        if (name === "mul") return value * b;
        if (name === "div") return value / b;
        if (name === "neg") return -value;
        if (name === "abs") return Math.abs(value);
        if (name === "exp") return Math.exp(value);
        if (name === "log") return Math.log(value);
        if (name === "relu") return Math.max(value, 0);
        if (name === "sigmoid") return 1 / (1 + Math.exp(-value));
        if (name === "sqrt") return Math.sqrt(value);
        if (name === "tanh") return Math.tanh(value);
        throw new Error(`Unknown kernel ${name}.`);
      });
    }

  }, { kernel, tolerance: WEBGPU_DEFAULT_TOLERANCE });

    expect(result.ok, `${result.name} max error ${result.maxError}`).toBe(true);
  }
});
