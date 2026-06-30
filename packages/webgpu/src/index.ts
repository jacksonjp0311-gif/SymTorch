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

export type WebGPUParityCase = {
  op: "add" | "sub" | "mul" | "div" | "neg" | "abs" | "exp" | "log" | "relu" | "sigmoid" | "sqrt" | "tanh" | "sumAll" | "meanAll" | "logSumExpAll";
  shape: readonly number[];
  tolerance: WebGPUTolerance;
};

export type WebGPUBackendDispatchStatus = {
  routedThroughCore: false;
  explicitKernelCount: number;
  parityCases: WebGPUParityCase[];
  remaining: string[];
};

export const WEBGPU_DEFAULT_TOLERANCE: WebGPUTolerance = {
  atol: 1e-5,
  rtol: 1e-4
};

export const WEBGPU_PARITY_CASES: readonly WebGPUParityCase[] = [
  "add",
  "sub",
  "mul",
  "div",
  "neg",
  "abs",
  "exp",
  "log",
  "relu",
  "sigmoid",
  "sqrt",
  "tanh",
  "sumAll",
  "meanAll",
  "logSumExpAll"
].map((op) => ({
  op: op as WebGPUParityCase["op"],
  shape: [2, 3],
  tolerance: WEBGPU_DEFAULT_TOLERANCE
}));

export function getWebGPUBackendDispatchStatus(): WebGPUBackendDispatchStatus {
  return {
    routedThroughCore: false,
    explicitKernelCount: WEBGPU_PARITY_CASES.length,
    parityCases: WEBGPU_PARITY_CASES.map((item) => ({ ...item, shape: [...item.shape], tolerance: { ...item.tolerance } })),
    remaining: [
      "register async WebGPU execution backend in @symtorch/core",
      "add broadcasting and axis reductions",
      "add matmul and autograd parity gates",
      "define browser and Node WebGPU runtime support matrix"
    ]
  };
}

export const WEBGPU_ADD_WGSL = `
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < arrayLength(&out)) {
    out[i] = left[i] + right[i];
  }
}
`;

export const WEBGPU_SUB_WGSL = binaryElementwiseShader("left[i] - right[i]");
export const WEBGPU_MUL_WGSL = binaryElementwiseShader("left[i] * right[i]");
export const WEBGPU_DIV_WGSL = binaryElementwiseShader("left[i] / right[i]");
export const WEBGPU_NEG_WGSL = unaryElementwiseShader("-input[i]");
export const WEBGPU_ABS_WGSL = unaryElementwiseShader("abs(input[i])");
export const WEBGPU_EXP_WGSL = unaryElementwiseShader("exp(input[i])");
export const WEBGPU_LOG_WGSL = unaryElementwiseShader("log(input[i])");
export const WEBGPU_RELU_WGSL = unaryElementwiseShader("max(input[i], 0.0)");
export const WEBGPU_SIGMOID_WGSL = unaryElementwiseShader("1.0 / (1.0 + exp(-input[i]))");
export const WEBGPU_SQRT_WGSL = unaryElementwiseShader("sqrt(input[i])");
export const WEBGPU_TANH_WGSL = unaryElementwiseShader("tanh(input[i])");
export const WEBGPU_SUM_ALL_WGSL = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x > 0u) {
    return;
  }
  var total = 0.0;
  for (var i = 0u; i < arrayLength(&input); i = i + 1u) {
    total = total + input[i];
  }
  out[0] = total;
}
`;
export const WEBGPU_LOG_SUM_EXP_ALL_WGSL = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x > 0u) {
    return;
  }
  let length = arrayLength(&input);
  if (length == 0u) {
    out[0] = -1.0 / 0.0;
    return;
  }
  var maxValue = input[0];
  for (var i = 1u; i < length; i = i + 1u) {
    maxValue = max(maxValue, input[i]);
  }
  var total = 0.0;
  for (var i = 0u; i < length; i = i + 1u) {
    total = total + exp(input[i] - maxValue);
  }
  out[0] = log(total) + maxValue;
}
`;

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

  scalar(value: number): WebGPUTensorStorage {
    return scalarTensor(this.device, value, this.pool);
  }

  async readTensor(storage: WebGPUTensorStorage): Promise<Float32Array> {
    return readTensor(this.device, storage);
  }

  disposeTensor(storage: WebGPUTensorStorage): void {
    this.pool.release(storage.buffer, storage.byteLength);
  }

  add(left: WebGPUTensorStorage, right: WebGPUTensorStorage): WebGPUTensorStorage {
    return addTensors(this.device, left, right, this.pool);
  }

  sub(left: WebGPUTensorStorage, right: WebGPUTensorStorage): WebGPUTensorStorage {
    return subTensors(this.device, left, right, this.pool);
  }

  mul(left: WebGPUTensorStorage, right: WebGPUTensorStorage): WebGPUTensorStorage {
    return mulTensors(this.device, left, right, this.pool);
  }

  div(left: WebGPUTensorStorage, right: WebGPUTensorStorage): WebGPUTensorStorage {
    return divTensors(this.device, left, right, this.pool);
  }

  neg(input: WebGPUTensorStorage): WebGPUTensorStorage {
    return negTensor(this.device, input, this.pool);
  }

  abs(input: WebGPUTensorStorage): WebGPUTensorStorage {
    return absTensor(this.device, input, this.pool);
  }

  exp(input: WebGPUTensorStorage): WebGPUTensorStorage {
    return expTensor(this.device, input, this.pool);
  }

  log(input: WebGPUTensorStorage): WebGPUTensorStorage {
    return logTensor(this.device, input, this.pool);
  }

  relu(input: WebGPUTensorStorage): WebGPUTensorStorage {
    return reluTensor(this.device, input, this.pool);
  }

  sigmoid(input: WebGPUTensorStorage): WebGPUTensorStorage {
    return sigmoidTensor(this.device, input, this.pool);
  }

  sqrt(input: WebGPUTensorStorage): WebGPUTensorStorage {
    return sqrtTensor(this.device, input, this.pool);
  }

  tanh(input: WebGPUTensorStorage): WebGPUTensorStorage {
    return tanhTensor(this.device, input, this.pool);
  }

  sumAll(input: WebGPUTensorStorage): WebGPUTensorStorage {
    return sumAllTensor(this.device, input, this.pool);
  }

  meanAll(input: WebGPUTensorStorage): WebGPUTensorStorage {
    return meanAllTensor(this.device, input, this.pool);
  }

  logSumExpAll(input: WebGPUTensorStorage): WebGPUTensorStorage {
    return logSumExpAllTensor(this.device, input, this.pool);
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

export function scalarTensor(device: GPUDevice, value: number, pool?: BufferPool): WebGPUTensorStorage {
  return uploadTensor(device, [value], [], pool);
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

export function addTensors(
  device: GPUDevice,
  left: WebGPUTensorStorage,
  right: WebGPUTensorStorage,
  pool?: BufferPool
): WebGPUTensorStorage {
  return binaryElementwise("addTensors", device, left, right, WEBGPU_ADD_WGSL, pool);
}

export function subTensors(
  device: GPUDevice,
  left: WebGPUTensorStorage,
  right: WebGPUTensorStorage,
  pool?: BufferPool
): WebGPUTensorStorage {
  return binaryElementwise("subTensors", device, left, right, WEBGPU_SUB_WGSL, pool);
}

export function mulTensors(
  device: GPUDevice,
  left: WebGPUTensorStorage,
  right: WebGPUTensorStorage,
  pool?: BufferPool
): WebGPUTensorStorage {
  return binaryElementwise("mulTensors", device, left, right, WEBGPU_MUL_WGSL, pool);
}

export function divTensors(
  device: GPUDevice,
  left: WebGPUTensorStorage,
  right: WebGPUTensorStorage,
  pool?: BufferPool
): WebGPUTensorStorage {
  return binaryElementwise("divTensors", device, left, right, WEBGPU_DIV_WGSL, pool);
}

export function negTensor(device: GPUDevice, input: WebGPUTensorStorage, pool?: BufferPool): WebGPUTensorStorage {
  return unaryElementwise(device, input, WEBGPU_NEG_WGSL, pool);
}

export function absTensor(device: GPUDevice, input: WebGPUTensorStorage, pool?: BufferPool): WebGPUTensorStorage {
  return unaryElementwise(device, input, WEBGPU_ABS_WGSL, pool);
}

export function expTensor(device: GPUDevice, input: WebGPUTensorStorage, pool?: BufferPool): WebGPUTensorStorage {
  return unaryElementwise(device, input, WEBGPU_EXP_WGSL, pool);
}

export function logTensor(device: GPUDevice, input: WebGPUTensorStorage, pool?: BufferPool): WebGPUTensorStorage {
  return unaryElementwise(device, input, WEBGPU_LOG_WGSL, pool);
}

export function reluTensor(device: GPUDevice, input: WebGPUTensorStorage, pool?: BufferPool): WebGPUTensorStorage {
  return unaryElementwise(device, input, WEBGPU_RELU_WGSL, pool);
}

export function sigmoidTensor(device: GPUDevice, input: WebGPUTensorStorage, pool?: BufferPool): WebGPUTensorStorage {
  return unaryElementwise(device, input, WEBGPU_SIGMOID_WGSL, pool);
}

export function sqrtTensor(device: GPUDevice, input: WebGPUTensorStorage, pool?: BufferPool): WebGPUTensorStorage {
  return unaryElementwise(device, input, WEBGPU_SQRT_WGSL, pool);
}

export function tanhTensor(device: GPUDevice, input: WebGPUTensorStorage, pool?: BufferPool): WebGPUTensorStorage {
  return unaryElementwise(device, input, WEBGPU_TANH_WGSL, pool);
}

export function sumAllTensor(device: GPUDevice, input: WebGPUTensorStorage, pool?: BufferPool): WebGPUTensorStorage {
  const byteLength = Float32Array.BYTES_PER_ELEMENT;
  const outputBuffer = pool?.acquire(byteLength, bufferUsage().storageCopy) ?? device.createBuffer({
    size: byteLength,
    usage: bufferUsage().storageCopy
  });
  const pipeline = getPipeline(device, WEBGPU_SUM_ALL_WGSL);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer } },
      { binding: 1, resource: { buffer: outputBuffer } }
    ]
  });
  dispatchElementwise(device, pipeline, bindGroup, 1);
  return {
    kind: "webgpu",
    dtype: "float32",
    shape: [],
    size: 1,
    byteLength,
    buffer: outputBuffer
  };
}

export function meanAllTensor(device: GPUDevice, input: WebGPUTensorStorage, pool?: BufferPool): WebGPUTensorStorage {
  if (input.size === 0) throw new Error("meanAllTensor requires at least one element.");
  const summed = sumAllTensor(device, input, pool);
  const divisor = scalarTensor(device, input.size, pool);
  return divTensors(device, summed, divisor, pool);
}

export function logSumExpAllTensor(device: GPUDevice, input: WebGPUTensorStorage, pool?: BufferPool): WebGPUTensorStorage {
  const byteLength = Float32Array.BYTES_PER_ELEMENT;
  const outputBuffer = pool?.acquire(byteLength, bufferUsage().storageCopy) ?? device.createBuffer({
    size: byteLength,
    usage: bufferUsage().storageCopy
  });
  const pipeline = getPipeline(device, WEBGPU_LOG_SUM_EXP_ALL_WGSL);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer } },
      { binding: 1, resource: { buffer: outputBuffer } }
    ]
  });
  dispatchElementwise(device, pipeline, bindGroup, 1);
  return {
    kind: "webgpu",
    dtype: "float32",
    shape: [],
    size: 1,
    byteLength,
    buffer: outputBuffer
  };
}

function unaryElementwise(
  device: GPUDevice,
  input: WebGPUTensorStorage,
  shader: string,
  pool?: BufferPool
): WebGPUTensorStorage {
  const byteLength = input.byteLength;
  const outputBuffer = pool?.acquire(byteLength, bufferUsage().storageCopy) ?? device.createBuffer({
    size: byteLength,
    usage: bufferUsage().storageCopy
  });
  const pipeline = getPipeline(device, shader);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer } },
      { binding: 1, resource: { buffer: outputBuffer } }
    ]
  });
  dispatchElementwise(device, pipeline, bindGroup, input.size);
  return tensorStorageLike(input, outputBuffer);
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

const pipelineCache = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();

function binaryElementwise(
  op: string,
  device: GPUDevice,
  left: WebGPUTensorStorage,
  right: WebGPUTensorStorage,
  shader: string,
  pool?: BufferPool
): WebGPUTensorStorage {
  assertSameTensorShape(op, left, right);
  const byteLength = left.byteLength;
  const outputBuffer = pool?.acquire(byteLength, bufferUsage().storageCopy) ?? device.createBuffer({
    size: byteLength,
    usage: bufferUsage().storageCopy
  });
  const pipeline = getPipeline(device, shader);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: left.buffer } },
      { binding: 1, resource: { buffer: right.buffer } },
      { binding: 2, resource: { buffer: outputBuffer } }
    ]
  });
  dispatchElementwise(device, pipeline, bindGroup, left.size);
  return tensorStorageLike(left, outputBuffer);
}

function dispatchElementwise(device: GPUDevice, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, size: number): void {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(size / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
}

function getPipeline(device: GPUDevice, shader: string): GPUComputePipeline {
  const cache = pipelineCache.get(device) ?? new Map<string, GPUComputePipeline>();
  const cached = cache.get(shader);
  if (cached) return cached;
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: shader }),
      entryPoint: "main"
    }
  });
  cache.set(shader, pipeline);
  pipelineCache.set(device, cache);
  return pipeline;
}

function tensorStorageLike(source: WebGPUTensorStorage, buffer: GPUBuffer): WebGPUTensorStorage {
  return {
    kind: "webgpu",
    dtype: "float32",
    shape: [...source.shape],
    size: source.size,
    byteLength: source.byteLength,
    buffer
  };
}

function assertSameTensorShape(op: string, left: WebGPUTensorStorage, right: WebGPUTensorStorage): void {
  if (left.dtype !== right.dtype) throw new Error(`${op} dtype mismatch: ${left.dtype} vs ${right.dtype}.`);
  if (left.size !== right.size || !sameShape(left.shape, right.shape)) {
    throw new Error(`${op} shape mismatch: [${left.shape.join(", ")}] vs [${right.shape.join(", ")}].`);
  }
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function binaryElementwiseShader(expression: string): string {
  return `
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < arrayLength(&out)) {
    out[i] = ${expression};
  }
}
`;
}

function unaryElementwiseShader(expression: string): string {
  return `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < arrayLength(&out)) {
    out[i] = ${expression};
  }
}
`;
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
