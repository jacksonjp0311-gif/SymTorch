import { expect, test } from "@playwright/test";
import { WEBGPU_ADD_WGSL, WEBGPU_DEFAULT_TOLERANCE } from "@symtorch/webgpu";

test("webgpu add kernel matches the CPU oracle when WebGPU is available", async ({ page }) => {
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

  const result = await page.evaluate(async ({ shader, tolerance }) => {
    const adapter = await navigator.gpu!.requestAdapter();
    if (!adapter) throw new Error("Expected a WebGPU adapter after availability check.");
    const device = await adapter.requestDevice();
    const left = new Float32Array([1, -2, 3.5, 4, 9, -8, 0.25, 12]);
    const right = new Float32Array([0.5, 2, -1.5, 8, -4, 3, 0.75, -10]);
    const expected = Array.from(left, (value, index) => value + (right[index] ?? 0));
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const leftBuffer = device.createBuffer({ size: left.byteLength, usage });
    const rightBuffer = device.createBuffer({ size: right.byteLength, usage });
    const outBuffer = device.createBuffer({ size: left.byteLength, usage });
    device.queue.writeBuffer(leftBuffer, 0, left);
    device.queue.writeBuffer(rightBuffer, 0, right);

    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: shader }),
        entryPoint: "main"
      }
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: leftBuffer } },
        { binding: 1, resource: { buffer: rightBuffer } },
        { binding: 2, resource: { buffer: outBuffer } }
      ]
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();

    const readback = device.createBuffer({
      size: left.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    encoder.copyBufferToBuffer(outBuffer, 0, readback, 0, left.byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
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
    return { actual, expected, maxError, ok };
  }, { shader: WEBGPU_ADD_WGSL, tolerance: WEBGPU_DEFAULT_TOLERANCE });

  expect(result.ok, `max error ${result.maxError}`).toBe(true);
  expect(result.actual).toEqual(result.expected);
});
