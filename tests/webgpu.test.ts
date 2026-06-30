import { describe, expect, it } from "vitest";
import {
  addTensors,
  createWebGPUContext,
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
});

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
    if (!this.pipeline?.code.includes("left[i] + right[i]") || !this.bindGroup) {
      throw new Error("Fake compute pass expected the add kernel and a bind group.");
    }
    const left = new Float32Array(this.bindGroup.bufferAt(0).bytes);
    const right = new Float32Array(this.bindGroup.bufferAt(1).bytes);
    const out = new Float32Array(this.bindGroup.bufferAt(2).bytes);
    for (let i = 0; i < out.length; i++) out[i] = (left[i] ?? 0) + (right[i] ?? 0);
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
