/// <reference types="@webgpu/types" />

export type WebGPUStatus =
  | { available: true; adapter: GPUAdapter; features: readonly string[]; limits: Record<string, number> }
  | { available: false; reason: string };

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

function extractLimits(limits: GPUSupportedLimits): Record<string, number> {
  const out: Record<string, number> = {};
  for (const key of Object.keys(limits) as Array<keyof GPUSupportedLimits>) {
    const value = limits[key];
    if (typeof value === "number") out[String(key)] = value;
  }
  return out;
}
