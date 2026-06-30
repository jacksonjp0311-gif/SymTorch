/// <reference types="@webgpu/types" />

export type WebGPUDType = "float32";

export type WebGPUStatus =
  | { available: true; adapter: GPUAdapter; features: readonly string[]; limits: Record<string, number> }
  | { available: false; reason: string };

export type WebGPUTensorStorage = {
  kind: "webgpu";
  dtype: WebGPUDType;
  shape: readonly number[];
  size: number;
  byteLength: number;
  buffer: GPUBuffer;
};

export type WebGPUTolerance = {
  atol: number;
  rtol: number;
};

export const WEBGPU_DEFAULT_TOLERANCE: WebGPUTolerance = {
  atol: 1e-5,
  rtol: 1e-4
};

export async function detectWebGPU(gpu: GPU | undefined = globalThis.navigator?.gpu): Promise<WebGPUStatus> {
  if (!gpu) return { available: false, reason: "navigator.gpu is not available in this runtime." };
  const adapter = await gpu.requestAdapter();
  if (!adapter) return { available: false, reason: "No compatible WebGPU adapter was found." };
  return {
    available: true,
    adapter,
    features: Array.from(adapter.features.values()),
    limits: extractLimits(adapter.limits)
  };
}

export async function requestWebGPUDevice(adapter: GPUAdapter): Promise<GPUDevice> {
  return adapter.requestDevice();
}

export class WebGPUContext {
  readonly pool: BufferPool;

  constructor(readonly device: GPUDevice) {
    this.pool = new BufferPool(device);
  }

  uploadTensor(data: Float32Array | readonly number[], shape: readonly number[]): WebGPUTensorStorage {
    return uploadTensor(this.device, data, shape, this.pool);
  }

  async readTensor(storage: WebGPUTensorStorage): Promise<Float32Array> {
    return readTensor(this.device, storage);
  }

  disposeTensor(storage: WebGPUTensorStorage): void {
    this.pool.release(storage.buffer, storage.byteLength);
  }

  destroy(): void {
    this.pool.destroy();
  }
}

export function createWebGPUContext(device: GPUDevice): WebGPUContext {
  return new WebGPUContext(device);
}

export function uploadTensor(
  device: GPUDevice,
  data: Float32Array | readonly number[],
  shape: readonly number[],
  pool?: BufferPool
): WebGPUTensorStorage {
  const values = data instanceof Float32Array ? data : new Float32Array(data);
  const size = sizeOf(shape);
  if (values.length !== size) {
    throw new Error(`WebGPU tensor data length ${values.length} does not match shape [${shape.join(", ")}] (${size}).`);
  }
  const byteLength = alignedByteLength(values.byteLength);
  const buffer = pool?.acquire(byteLength, bufferUsage().storageCopy) ?? device.createBuffer({
    size: byteLength,
    usage: bufferUsage().storageCopy
  });
  device.queue.writeBuffer(buffer, 0, values.buffer, values.byteOffset, values.byteLength);
  return {
    kind: "webgpu",
    dtype: "float32",
    shape: [...shape],
    size,
    byteLength,
    buffer
  };
}

export async function readTensor(device: GPUDevice, storage: WebGPUTensorStorage): Promise<Float32Array> {
  const readback = device.createBuffer({
    size: storage.byteLength,
    usage: bufferUsage().mapReadCopy
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(storage.buffer, 0, readback, 0, storage.byteLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapModeValue.READ);
  const mapped = readback.getMappedRange(0, storage.byteLength);
  const values = new Float32Array(mapped.slice(0, storage.size * Float32Array.BYTES_PER_ELEMENT));
  readback.unmap();
  readback.destroy();
  return values;
}

export class BufferPool {
  private readonly free = new Map<number, GPUBuffer[]>();

  constructor(private readonly device: GPUDevice) {}

  acquire(size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const bucket = this.free.get(size);
    const existing = bucket?.pop();
    if (existing) return existing;
    return this.device.createBuffer({ size, usage });
  }

  release(buffer: GPUBuffer, size: number): void {
    const bucket = this.free.get(size) ?? [];
    bucket.push(buffer);
    this.free.set(size, bucket);
  }

  destroy(): void {
    for (const buffers of this.free.values()) {
      for (const buffer of buffers) buffer.destroy();
    }
    this.free.clear();
  }
}

function sizeOf(shape: readonly number[]): number {
  return shape.reduce((acc, n) => acc * n, 1);
}

function alignedByteLength(byteLength: number): number {
  return Math.max(4, Math.ceil(byteLength / 4) * 4);
}

function bufferUsage(): { storageCopy: GPUBufferUsageFlags; mapReadCopy: GPUBufferUsageFlags } {
  const usage = globalThis.GPUBufferUsage;
  return {
    storageCopy: (usage?.STORAGE ?? 0x80) | (usage?.COPY_SRC ?? 0x04) | (usage?.COPY_DST ?? 0x08),
    mapReadCopy: (usage?.MAP_READ ?? 0x01) | (usage?.COPY_DST ?? 0x08)
  };
}

const GPUMapModeValue = {
  READ: globalThis.GPUMapMode?.READ ?? 0x0001
} as const;

function extractLimits(limits: GPUSupportedLimits): Record<string, number> {
  const out: Record<string, number> = {};
  for (const key of Object.keys(limits) as Array<keyof GPUSupportedLimits>) {
    const value = limits[key];
    if (typeof value === "number") out[String(key)] = value;
  }
  return out;
}
