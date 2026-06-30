import { describe, expect, it } from "vitest";
import {
  absTensor,
  addTensors,
  createWebGPUContext,
  divTensors,
  expTensor,
  logSumExpAllTensor,
  logTensor,
  meanAllTensor,
  mulTensors,
  negTensor,
  reluTensor,
  sigmoidTensor,
  sqrtTensor,
  scalarTensor,
  sumAllTensor,
  subTensors,
  tanhTensor,
  uploadTensor,
  WEBGPU_ADD_WGSL,
  WEBGPU_DEFAULT_TOLERANCE
} from "@symtorch/webgpu";

describe("@symtorch/webgpu", () => {
  it("uploads and reads tensor storage through an explicit context", async () => {
    const device = new FakeGPUDevice();
    const context = createWebGPUContext(device as unknown as GPUDevice);
    const storage = context.uploadTensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const values = await context.readTensor(storage);

    expect(storage).toMatchObject({
      kind: "webgpu",
      dtype: "float32",
      shape: [2, 2],
      size: 4,
      byteLength: 16
    });
    expect(Array.from(values)).toEqual([1, 2, 3, 4]);

    context.disposeTensor(storage);
    context.destroy();
  });

  it("validates upload shape and exposes parity tolerances", () => {
    const device = new FakeGPUDevice();

    expect(WEBGPU_DEFAULT_TOLERANCE).toEqual({ atol: 1e-5, rtol: 1e-4 });
    expect(() => uploadTensor(device as unknown as GPUDevice, [1, 2, 3], [2, 2]))
      .toThrow("does not match shape");
  });

  it("runs the same-shape add kernel against a CPU oracle", async () => {
    const device = new FakeGPUDevice();
    const context = createWebGPUContext(device as unknown as GPUDevice);
    const left = context.uploadTensor([1, -2, 3.5, 4], [4]);
    const right = context.uploadTensor([0.5, 2, -1.5, 8], [4]);

    const result = context.add(left, right);
    const values = await context.readTensor(result);

    expect(WEBGPU_ADD_WGSL).toContain("@compute");
    expect(Array.from(values)).toEqual([1.5, 0, 2, 12]);
    expect(() => addTensors(device as unknown as GPUDevice, left, context.uploadTensor([1, 2], [2])))
      .toThrow("shape mismatch");
  });

  it("runs the same-shape elementwise kernel set against CPU oracles", async () => {
    const device = new FakeGPUDevice();
    const context = createWebGPUContext(device as unknown as GPUDevice);
    const left = context.uploadTensor([2, -4, 9, 8], [4]);
    const right = context.uploadTensor([1, 2, -3, 4], [4]);

    await expectStorage(context, context.sub(left, right), [1, -6, 12, 4]);
    await expectStorage(context, context.mul(left, right), [2, -8, -27, 32]);
    await expectStorage(context, context.div(left, right), [2, -2, -3, 2]);
    await expectStorage(context, context.neg(left), [-2, 4, -9, -8]);
    await expectStorage(context, subTensors(device as unknown as GPUDevice, left, right), [1, -6, 12, 4]);
    await expectStorage(context, mulTensors(device as unknown as GPUDevice, left, right), [2, -8, -27, 32]);
    await expectStorage(context, divTensors(device as unknown as GPUDevice, left, right), [2, -2, -3, 2]);
    await expectStorage(context, negTensor(device as unknown as GPUDevice, left), [-2, 4, -9, -8]);
  });

  it("runs the unary activation and math kernel set against CPU oracles", async () => {
    const device = new FakeGPUDevice();
    const context = createWebGPUContext(device as unknown as GPUDevice);
    const signed = context.uploadTensor([-2, -0.5, 0, 3], [4]);
    const positive = context.uploadTensor([0.25, 1, 4, 9], [4]);

    await expectStorage(context, context.abs(signed), [2, 0.5, 0, 3]);
    await expectStorage(context, context.relu(signed), [0, 0, 0, 3]);
    await expectStorageClose(context, context.sigmoid(signed), Array.from([-2, -0.5, 0, 3], sigmoid));
    await expectStorageClose(context, context.exp(signed), Array.from([-2, -0.5, 0, 3], Math.exp));
    await expectStorageClose(context, context.log(positive), Array.from([0.25, 1, 4, 9], Math.log));
    await expectStorage(context, context.sqrt(positive), [0.5, 1, 2, 3]);
    await expectStorageClose(context, context.tanh(signed), Array.from([-2, -0.5, 0, 3], Math.tanh));

    await expectStorage(context, absTensor(device as unknown as GPUDevice, signed), [2, 0.5, 0, 3]);
    await expectStorage(context, reluTensor(device as unknown as GPUDevice, signed), [0, 0, 0, 3]);
    await expectStorageClose(context, sigmoidTensor(device as unknown as GPUDevice, signed), Array.from([-2, -0.5, 0, 3], sigmoid));
    await expectStorageClose(context, expTensor(device as unknown as GPUDevice, signed), Array.from([-2, -0.5, 0, 3], Math.exp));
    await expectStorageClose(context, logTensor(device as unknown as GPUDevice, positive), Array.from([0.25, 1, 4, 9], Math.log));
    await expectStorage(context, sqrtTensor(device as unknown as GPUDevice, positive), [0.5, 1, 2, 3]);
    await expectStorageClose(context, tanhTensor(device as unknown as GPUDevice, signed), Array.from([-2, -0.5, 0, 3], Math.tanh));
  });

  it("runs the sum-all reduction kernel against a CPU oracle", async () => {
    const device = new FakeGPUDevice();
    const context = createWebGPUContext(device as unknown as GPUDevice);
    const matrix = context.uploadTensor([1, -2, 3.5, 4, 8, -0.5], [2, 3]);

    const result = context.sumAll(matrix);
    expect(result).toMatchObject({
      kind: "webgpu",
      dtype: "float32",
      shape: [],
      size: 1,
      byteLength: 4
    });
    await expectStorage(context, result, [14]);
    await expectStorage(context, sumAllTensor(device as unknown as GPUDevice, matrix), [14]);
  });

  it("composes scalar tensors and mean-all from explicit kernels", async () => {
    const device = new FakeGPUDevice();
    const context = createWebGPUContext(device as unknown as GPUDevice);
    const matrix = context.uploadTensor([1, -2, 3.5, 4, 8, -0.5], [2, 3]);
    const scalar = context.scalar(6);

    expect(scalar).toMatchObject({
      kind: "webgpu",
      dtype: "float32",
      shape: [],
      size: 1,
      byteLength: 4
    });
    await expectStorage(context, scalar, [6]);
    await expectStorage(context, scalarTensor(device as unknown as GPUDevice, 2.5), [2.5]);

    const result = context.meanAll(matrix);
    expect(result).toMatchObject({
      kind: "webgpu",
      dtype: "float32",
      shape: [],
      size: 1,
      byteLength: 4
    });
    await expectStorageClose(context, result, [14 / 6]);
    await expectStorageClose(context, meanAllTensor(device as unknown as GPUDevice, matrix), [14 / 6]);
  });

  it("runs stable log-sum-exp-all against CPU oracles", async () => {
    const device = new FakeGPUDevice();
    const context = createWebGPUContext(device as unknown as GPUDevice);
    const values = context.uploadTensor([-2, 0, 1.5, 4], [4]);
    const large = context.uploadTensor([1000, 1001, 999], [3]);

    await expectStorageClose(context, context.logSumExpAll(values), [stableLogSumExp([-2, 0, 1.5, 4])]);
    await expectStorageClose(context, logSumExpAllTensor(device as unknown as GPUDevice, values), [stableLogSumExp([-2, 0, 1.5, 4])]);
    await expectStorageClose(context, context.logSumExpAll(large), [Math.fround(stableLogSumExp([1000, 1001, 999]))]);
  });
});

async function expectStorage(context: ReturnType<typeof createWebGPUContext>, storage: ReturnType<typeof addTensors>, expected: number[]): Promise<void> {
  await expect(context.readTensor(storage)).resolves.toEqual(new Float32Array(expected));
}

async function expectStorageClose(
  context: ReturnType<typeof createWebGPUContext>,
  storage: ReturnType<typeof addTensors>,
  expected: number[]
): Promise<void> {
  const values = await context.readTensor(storage);
  expect(Array.from(values)).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(values[i]).toBeCloseTo(expected[i] ?? 0, 6);
  }
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function stableLogSumExp(values: number[]): number {
  const max = Math.max(...values);
  return Math.log(values.reduce((total, value) => total + Math.exp(value - max), 0)) + max;
}

class FakeGPUDevice {
  readonly queue = new FakeGPUQueue();

  createBuffer(descriptor: GPUBufferDescriptor): FakeGPUBuffer {
    return new FakeGPUBuffer(descriptor.size);
  }

  createCommandEncoder(): FakeGPUCommandEncoder {
    return new FakeGPUCommandEncoder();
  }

  createShaderModule(descriptor: GPUShaderModuleDescriptor): FakeGPUShaderModule {
    return new FakeGPUShaderModule(descriptor.code);
  }

  createComputePipeline(descriptor: GPUComputePipelineDescriptor): FakeGPUComputePipeline {
    return new FakeGPUComputePipeline((descriptor.compute.module as unknown as FakeGPUShaderModule).code);
  }

  createBindGroup(descriptor: GPUBindGroupDescriptor): FakeGPUBindGroup {
    return new FakeGPUBindGroup(descriptor.entries);
  }
}

class FakeGPUQueue {
  writeBuffer(buffer: FakeGPUBuffer, bufferOffset: number, data: ArrayBuffer, dataOffset = 0, size?: number): void {
    const source = new Uint8Array(data, dataOffset, size ?? data.byteLength - dataOffset);
    new Uint8Array(buffer.bytes).set(source, bufferOffset);
  }

  submit(_commandBuffers: unknown[]): void {}
}

class FakeGPUCommandEncoder {
  beginComputePass(): FakeGPUComputePassEncoder {
    return new FakeGPUComputePassEncoder();
  }

  copyBufferToBuffer(source: FakeGPUBuffer, sourceOffset: number, destination: FakeGPUBuffer, destinationOffset: number, size: number): void {
    const sourceBytes = new Uint8Array(source.bytes, sourceOffset, size);
    new Uint8Array(destination.bytes).set(sourceBytes, destinationOffset);
  }

  finish(): unknown {
    return {};
  }
}

class FakeGPUShaderModule {
  constructor(readonly code: string) {}
}

class FakeGPUComputePipeline {
  constructor(readonly code: string) {}

  getBindGroupLayout(_index: number): unknown {
    return {};
  }
}

class FakeGPUBindGroup {
  constructor(readonly entries: readonly GPUBindGroupEntry[]) {}

  bufferAt(binding: number): FakeGPUBuffer {
    const entry = this.entries.find((item) => item.binding === binding);
    const resource = entry?.resource as { buffer?: FakeGPUBuffer } | undefined;
    if (!resource?.buffer) throw new Error(`Missing fake bind group buffer at binding ${binding}.`);
    return resource.buffer;
  }
}

class FakeGPUComputePassEncoder {
  private pipeline: FakeGPUComputePipeline | null = null;
  private bindGroup: FakeGPUBindGroup | null = null;

  setPipeline(pipeline: FakeGPUComputePipeline): void {
    this.pipeline = pipeline;
  }

  setBindGroup(_index: number, bindGroup: FakeGPUBindGroup): void {
    this.bindGroup = bindGroup;
  }

  dispatchWorkgroups(_x: number): void {
    if (!this.pipeline || !this.bindGroup) throw new Error("Fake compute pass expected a pipeline and bind group.");
    if (this.pipeline.code.includes("total = total + input[i]")) {
      const input = new Float32Array(this.bindGroup.bufferAt(0).bytes);
      const out = new Float32Array(this.bindGroup.bufferAt(1).bytes);
      out[0] = Array.from(input).reduce((total, value) => total + value, 0);
      return;
    }
    if (this.pipeline.code.includes("input[i] - maxValue")) {
      const input = new Float32Array(this.bindGroup.bufferAt(0).bytes);
      const out = new Float32Array(this.bindGroup.bufferAt(1).bytes);
      out[0] = stableLogSumExp(Array.from(input));
      return;
    }
    if (this.pipeline.code.includes("var<storage, read> input")) {
      const input = new Float32Array(this.bindGroup.bufferAt(0).bytes);
      const out = new Float32Array(this.bindGroup.bufferAt(1).bytes);
      for (let i = 0; i < out.length; i++) {
        const value = input[i] ?? 0;
        if (this.pipeline.code.includes("1.0 / (1.0 + exp(-input[i]))")) out[i] = sigmoid(value);
        else if (this.pipeline.code.includes("-input[i]")) out[i] = -value;
        else if (this.pipeline.code.includes("abs(input[i])")) out[i] = Math.abs(value);
        else if (this.pipeline.code.includes("exp(input[i])")) out[i] = Math.exp(value);
        else if (this.pipeline.code.includes("log(input[i])")) out[i] = Math.log(value);
        else if (this.pipeline.code.includes("max(input[i], 0.0)")) out[i] = Math.max(value, 0);
        else if (this.pipeline.code.includes("sqrt(input[i])")) out[i] = Math.sqrt(value);
        else if (this.pipeline.code.includes("tanh(input[i])")) out[i] = Math.tanh(value);
        else throw new Error("Fake compute pass received an unknown unary shader.");
      }
      return;
    }
    const left = new Float32Array(this.bindGroup.bufferAt(0).bytes);
    const right = new Float32Array(this.bindGroup.bufferAt(1).bytes);
    const out = new Float32Array(this.bindGroup.bufferAt(2).bytes);
    for (let i = 0; i < out.length; i++) {
      const a = left[i] ?? 0;
      const b = right[i] ?? 0;
      if (this.pipeline.code.includes("left[i] + right[i]")) out[i] = a + b;
      else if (this.pipeline.code.includes("left[i] - right[i]")) out[i] = a - b;
      else if (this.pipeline.code.includes("left[i] * right[i]")) out[i] = a * b;
      else if (this.pipeline.code.includes("left[i] / right[i]")) out[i] = a / b;
      else throw new Error("Fake compute pass received an unknown shader.");
    }
  }

  end(): void {}
}

class FakeGPUBuffer {
  readonly bytes: ArrayBuffer;
  private mapped = false;

  constructor(readonly size: number) {
    this.bytes = new ArrayBuffer(size);
  }

  async mapAsync(_mode: number): Promise<void> {
    this.mapped = true;
  }

  getMappedRange(offset = 0, size = this.size): ArrayBuffer {
    if (!this.mapped) throw new Error("Buffer is not mapped.");
    return this.bytes.slice(offset, offset + size);
  }

  unmap(): void {
    this.mapped = false;
  }

  destroy(): void {}
}
