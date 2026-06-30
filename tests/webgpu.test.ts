import { describe, expect, it } from "vitest";
import {
  createWebGPUContext,
  uploadTensor,
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
});

class FakeGPUDevice {
  readonly queue = new FakeGPUQueue();

  createBuffer(descriptor: GPUBufferDescriptor): FakeGPUBuffer {
    return new FakeGPUBuffer(descriptor.size);
  }

  createCommandEncoder(): FakeGPUCommandEncoder {
    return new FakeGPUCommandEncoder();
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
  copyBufferToBuffer(source: FakeGPUBuffer, sourceOffset: number, destination: FakeGPUBuffer, destinationOffset: number, size: number): void {
    const sourceBytes = new Uint8Array(source.bytes, sourceOffset, size);
    new Uint8Array(destination.bytes).set(sourceBytes, destinationOffset);
  }

  finish(): unknown {
    return {};
  }
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
