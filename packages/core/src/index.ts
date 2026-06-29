export type DType = "float32";
export type Device = "cpu" | "webgpu";

export type TensorOptions = {
  requiresGrad?: boolean;
  shape?: readonly number[];
  dtype?: DType;
  device?: Device;
};

type BackwardEdge = {
  parent: Tensor;
  backward: (grad: Tensor) => Tensor;
};

const EPS = 1e-7;

export class Tensor {
  readonly data: Float32Array;
  readonly shape: readonly number[];
  readonly dtype: DType;
  readonly device: Device;
  readonly requiresGrad: boolean;
  grad: Tensor | null = null;
  private readonly parents: readonly BackwardEdge[];

  constructor(
    data: Float32Array | number[],
    shape: readonly number[] = [Array.from(data).length],
    options: TensorOptions = {},
    parents: readonly BackwardEdge[] = []
  ) {
    this.data = data instanceof Float32Array ? data : new Float32Array(data);
    this.shape = shape.length === 0 ? [] : [...shape];
    this.dtype = options.dtype ?? "float32";
    this.device = options.device ?? "cpu";
    this.requiresGrad = options.requiresGrad ?? parents.some((edge) => edge.parent.requiresGrad);
    this.parents = parents;
    const expected = sizeOf(this.shape);
    if (expected !== this.data.length) {
      throw new Error(`Tensor data length ${this.data.length} does not match shape [${this.shape.join(", ")}] (${expected}).`);
    }
  }

  get size(): number {
    return this.data.length;
  }

  get ndim(): number {
    return this.shape.length;
  }

  item(): number {
    if (this.data.length !== 1) throw new Error("item() requires a scalar tensor.");
    return this.data[0] ?? 0;
  }

  toArray(): number[] {
    return Array.from(this.data);
  }

  detach(): Tensor {
    return new Tensor(this.data.slice(), this.shape, { dtype: this.dtype, device: this.device });
  }

  zeroGrad(): void {
    this.grad = null;
  }

  backward(gradient?: Tensor): void {
    const seed = gradient ?? ones(this.shape);
    if (!sameShape(seed.shape, this.shape)) {
      throw new Error(`Backward gradient shape [${seed.shape.join(", ")}] does not match tensor shape [${this.shape.join(", ")}].`);
    }
    const topo: Tensor[] = [];
    const visited = new Set<Tensor>();
    const visit = (node: Tensor): void => {
      if (visited.has(node)) return;
      visited.add(node);
      for (const edge of node.parents) visit(edge.parent);
      topo.push(node);
    };
    visit(this);
    this.grad = seed;
    for (const node of topo.reverse()) {
      if (!node.grad) continue;
      for (const edge of node.parents) {
        if (!edge.parent.requiresGrad) continue;
        const contribution = edge.backward(node.grad);
        edge.parent.grad = edge.parent.grad ? addNoGrad(edge.parent.grad, contribution) : contribution;
      }
    }
  }

  add(other: TensorLike): Tensor {
    return add(this, other);
  }

  sub(other: TensorLike): Tensor {
    return sub(this, other);
  }

  mul(other: TensorLike): Tensor {
    return mul(this, other);
  }

  div(other: TensorLike): Tensor {
    return div(this, other);
  }

  matmul(other: Tensor): Tensor {
    return matmul(this, other);
  }

  sum(axis?: number, keepDims = false): Tensor {
    return sum(this, axis, keepDims);
  }

  mean(axis?: number, keepDims = false): Tensor {
    return mean(this, axis, keepDims);
  }

  reshape(shape: readonly number[]): Tensor {
    return reshape(this, shape);
  }

  transpose(): Tensor {
    return transpose(this);
  }

  relu(): Tensor {
    return relu(this);
  }

  sigmoid(): Tensor {
    return sigmoid(this);
  }

  tanh(): Tensor {
    return tanh(this);
  }

  pow(exponent: number): Tensor {
    return pow(this, exponent);
  }

  logSoftmax(axis = this.ndim - 1): Tensor {
    return logSoftmax(this, axis);
  }

  circularConvolve(other: Tensor): Tensor {
    return circularConvolve(this, other);
  }

  circularCorrelate(other: Tensor): Tensor {
    return circularCorrelate(this, other);
  }
}

export type TensorLike = Tensor | number | readonly number[] | Float32Array;

export function tensor(value: TensorLike, options: TensorOptions = {}): Tensor {
  if (value instanceof Tensor) return value;
  if (typeof value === "number") return new Tensor(new Float32Array([value]), [], options);
  const shape = options.shape ?? [value.length];
  return new Tensor(value instanceof Float32Array ? value : Array.from(value), shape, options);
}

export function fromArray(value: readonly number[], shape?: readonly number[], options: TensorOptions = {}): Tensor {
  return tensor(value, { ...options, shape: shape ?? [value.length] });
}

export function zeros(shape: readonly number[], options: TensorOptions = {}): Tensor {
  return new Tensor(new Float32Array(sizeOf(shape)), shape, options);
}

export function ones(shape: readonly number[], options: TensorOptions = {}): Tensor {
  return full(shape, 1, options);
}

export function full(shape: readonly number[], value: number, options: TensorOptions = {}): Tensor {
  const data = new Float32Array(sizeOf(shape));
  data.fill(value);
  return new Tensor(data, shape, options);
}

export function randn(shape: readonly number[], options: TensorOptions = {}): Tensor {
  const out = new Float32Array(sizeOf(shape));
  for (let i = 0; i < out.length; i += 2) {
    const u = Math.max(Math.random(), EPS);
    const v = Math.random();
    const mag = Math.sqrt(-2 * Math.log(u));
    out[i] = mag * Math.cos(2 * Math.PI * v);
    if (i + 1 < out.length) out[i + 1] = mag * Math.sin(2 * Math.PI * v);
  }
  return new Tensor(out, shape, options);
}

export function add(aLike: TensorLike, bLike: TensorLike): Tensor {
  const a = tensor(aLike);
  const b = tensor(bLike);
  return binaryOp(a, b, (x, y) => x + y, (grad) => grad, (grad) => grad);
}

export function sub(aLike: TensorLike, bLike: TensorLike): Tensor {
  const a = tensor(aLike);
  const b = tensor(bLike);
  return binaryOp(a, b, (x, y) => x - y, (grad) => grad, (grad) => neg(grad));
}

export function mul(aLike: TensorLike, bLike: TensorLike): Tensor {
  const a = tensor(aLike);
  const b = tensor(bLike);
  return binaryOp(a, b, (x, y) => x * y, (grad) => mulNoGrad(grad, b), (grad) => mulNoGrad(grad, a));
}

export function div(aLike: TensorLike, bLike: TensorLike): Tensor {
  const a = tensor(aLike);
  const b = tensor(bLike);
  return binaryOp(
    a,
    b,
    (x, y) => x / y,
    (grad) => divNoGrad(grad, b),
    (grad) => neg(divNoGrad(mulNoGrad(grad, a), mulNoGrad(b, b)))
  );
}

export function neg(x: Tensor): Tensor {
  return unaryOp(x, (v) => -v, (grad) => negNoGrad(grad));
}

export function exp(x: Tensor): Tensor {
  const out = unaryOp(x, Math.exp, (grad) => mulNoGrad(grad, out.detach()));
  return out;
}

export function log(x: Tensor): Tensor {
  return unaryOp(x, (v) => Math.log(Math.max(v, EPS)), (grad) => divNoGrad(grad, x));
}

export function abs(x: Tensor): Tensor {
  return unaryOp(
    x,
    Math.abs,
    (grad) => new Tensor(mapData(grad, (g, i) => {
      const value = x.data[i] ?? 0;
      return value > 0 ? g : value < 0 ? -g : 0;
    }), x.shape)
  );
}

export function pow(x: Tensor, exponent: number): Tensor {
  return unaryOp(
    x,
    (v) => v ** exponent,
    (grad) => mulNoGrad(grad, new Tensor(mapData(x, (v) => exponent * v ** (exponent - 1)), x.shape))
  );
}

export function sqrt(x: Tensor): Tensor {
  const out = unaryOp(
    x,
    (v) => Math.sqrt(Math.max(v, EPS)),
    (grad) => divNoGrad(grad, mulNoGrad(out.detach(), tensor(2)))
  );
  return out;
}

export function tanh(x: Tensor): Tensor {
  const out = unaryOp(x, Math.tanh, (grad) => {
    const y = out.detach();
    return mulNoGrad(grad, subNoGrad(ones(y.shape), mulNoGrad(y, y)));
  });
  return out;
}

export function clip(x: Tensor, min: number, max: number): Tensor {
  if (min > max) throw new Error(`clip min (${min}) must be <= max (${max}).`);
  return unaryOp(
    x,
    (v) => Math.min(max, Math.max(min, v)),
    (grad) => new Tensor(mapData(grad, (g, i) => {
      const value = x.data[i] ?? 0;
      return value >= min && value <= max ? g : 0;
    }), x.shape)
  );
}

export function relu(x: Tensor): Tensor {
  return unaryOp(x, (v) => Math.max(0, v), (grad) => new Tensor(mapData(grad, (g, i) => g * ((x.data[i] ?? 0) > 0 ? 1 : 0)), x.shape));
}

export function sigmoid(x: Tensor): Tensor {
  const out = unaryOp(x, (v) => 1 / (1 + Math.exp(-v)), (grad) => {
    const s = out.detach();
    return mulNoGrad(grad, mulNoGrad(s, subNoGrad(ones(s.shape), s)));
  });
  return out;
}

export function sum(x: Tensor, axis?: number, keepDims = false): Tensor {
  if (axis === undefined) {
    const value = x.data.reduce((acc, n) => acc + n, 0);
    return new Tensor(new Float32Array([value]), [], {}, [{ parent: x, backward: (grad) => full(x.shape, grad.item()) }]);
  }
  const normalized = normalizeAxis(axis, x.ndim);
  const outShape = x.shape.filter((_, i) => i !== normalized);
  const finalShape = keepDims ? replaceAt(x.shape, normalized, 1) : outShape;
  const out = new Float32Array(sizeOf(finalShape));
  forEachIndex(x.shape, (idx) => {
    const reduced = keepDims ? replaceAt(idx, normalized, 0) : idx.filter((_, i) => i !== normalized);
    const outIndex = offsetOf(reduced, finalShape);
    out[outIndex] = (out[outIndex] ?? 0) + (x.data[offsetOf(idx, x.shape)] ?? 0);
  });
  return new Tensor(out, finalShape, {}, [{
    parent: x,
    backward: (grad) => {
      const shapedGrad = keepDims ? grad : reshapeForReducedAxis(grad, x.shape, normalized);
      return broadcastTo(shapedGrad, shapedGrad.shape, x.shape);
    }
  }]);
}

export function mean(x: Tensor, axis?: number, keepDims = false): Tensor {
  const denom = axis === undefined ? x.size : x.shape[normalizeAxis(axis, x.ndim)] ?? 1;
  return div(sum(x, axis, keepDims), denom);
}

export function max(x: Tensor): Tensor {
  let best = -Infinity;
  let bestIndex = 0;
  for (let i = 0; i < x.data.length; i++) {
    const value = x.data[i] ?? -Infinity;
    if (value > best) {
      best = value;
      bestIndex = i;
    }
  }
  return new Tensor(new Float32Array([best]), [], {}, [{
    parent: x,
    backward: (grad) => {
      const data = new Float32Array(x.size);
      data[bestIndex] = grad.item();
      return new Tensor(data, x.shape);
    }
  }]);
}

export function matmul(a: Tensor, b: Tensor): Tensor {
  if (a.ndim !== 2 || b.ndim !== 2) throw new Error("matmul currently supports rank-2 tensors.");
  const [m, k] = a.shape as [number, number];
  const [k2, n] = b.shape as [number, number];
  if (k !== k2) throw new Error(`matmul shape mismatch: [${a.shape.join(", ")}] x [${b.shape.join(", ")}].`);
  const out = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let p = 0; p < k; p++) acc += (a.data[i * k + p] ?? 0) * (b.data[p * n + j] ?? 0);
      out[i * n + j] = acc;
    }
  }
  return new Tensor(out, [m, n], {}, [
    { parent: a, backward: (grad) => matmulNoGrad(grad, transposeNoGrad(b)) },
    { parent: b, backward: (grad) => matmulNoGrad(transposeNoGrad(a), grad) }
  ]);
}

export function circularConvolve(a: Tensor, b: Tensor): Tensor {
  assertSameVector("circularConvolve", a, b);
  const out = circularConvolveNoGrad(a, b);
  return new Tensor(out.data, out.shape, {}, [
    { parent: a, backward: (grad) => circularCorrelateNoGrad(grad, b) },
    { parent: b, backward: (grad) => circularCorrelateNoGrad(grad, a) }
  ]);
}

export function circularCorrelate(a: Tensor, b: Tensor): Tensor {
  assertSameVector("circularCorrelate", a, b);
  const out = circularCorrelateNoGrad(a, b);
  return new Tensor(out.data, out.shape, {}, [
    { parent: a, backward: (grad) => circularConvolveNoGrad(grad, b) },
    { parent: b, backward: (grad) => circularCorrelateNoGrad(a, grad) }
  ]);
}

export const bind = circularConvolve;
export const unbind = circularCorrelate;

export function transpose(x: Tensor): Tensor {
  if (x.ndim !== 2) throw new Error("transpose currently supports rank-2 tensors.");
  const out = transposeNoGrad(x);
  return new Tensor(out.data, out.shape, {}, [{ parent: x, backward: (grad) => transposeNoGrad(grad) }]);
}

export function reshape(x: Tensor, shape: readonly number[]): Tensor {
  if (sizeOf(shape) !== x.size) throw new Error(`Cannot reshape ${x.size} values to [${shape.join(", ")}].`);
  return new Tensor(x.data.slice(), shape, {}, [{ parent: x, backward: (grad) => new Tensor(grad.data.slice(), x.shape) }]);
}

export function logsumexp(x: Tensor, axis?: number, keepDims = false): Tensor {
  if (axis === undefined) {
    const m = max(x);
    return add(log(sum(exp(sub(x, m)))), m);
  }
  const normalized = normalizeAxis(axis, x.ndim);
  const m = maxAlongAxis(x, normalized, true);
  const reduced = log(sum(exp(sub(x, m)), normalized, keepDims));
  return add(reduced, keepDims ? m : reshapeWithoutAxis(m, normalized));
}

export function softmax(x: Tensor, axis = x.ndim - 1): Tensor {
  const normalized = normalizeAxis(axis, x.ndim);
  const shifted = sub(x, maxAlongAxis(x, normalized, true));
  const ex = exp(shifted);
  return div(ex, sum(ex, normalized, true));
}

export function logSoftmax(x: Tensor, axis = x.ndim - 1): Tensor {
  return sub(x, logsumexp(x, axis, true));
}

function binaryOp(
  a: Tensor,
  b: Tensor,
  forward: (a: number, b: number) => number,
  backwardA: (grad: Tensor) => Tensor,
  backwardB: (grad: Tensor) => Tensor
): Tensor {
  const shape = broadcastShape(a.shape, b.shape);
  const out = new Float32Array(sizeOf(shape));
  forEachIndex(shape, (idx) => {
    out[offsetOf(idx, shape)] = forward(valueAtBroadcast(a, idx, shape), valueAtBroadcast(b, idx, shape));
  });
  return new Tensor(out, shape, {}, [
    { parent: a, backward: (grad) => reduceBroadcast(backwardA(grad), a.shape) },
    { parent: b, backward: (grad) => reduceBroadcast(backwardB(grad), b.shape) }
  ]);
}

function unaryOp(x: Tensor, forward: (value: number) => number, backward: (grad: Tensor) => Tensor): Tensor {
  return new Tensor(mapData(x, forward), x.shape, {}, [{ parent: x, backward }]);
}

function maxAlongAxis(x: Tensor, axis: number, keepDims: boolean): Tensor {
  const outShape = keepDims ? replaceAt(x.shape, axis, 1) : x.shape.filter((_, i) => i !== axis);
  const out = new Float32Array(sizeOf(outShape));
  out.fill(-Infinity);
  forEachIndex(x.shape, (idx) => {
    const reduced = keepDims ? replaceAt(idx, axis, 0) : idx.filter((_, i) => i !== axis);
    const outIndex = offsetOf(reduced, outShape);
    out[outIndex] = Math.max(out[outIndex] ?? -Infinity, x.data[offsetOf(idx, x.shape)] ?? -Infinity);
  });
  return new Tensor(out, outShape);
}

function reshapeWithoutAxis(x: Tensor, axis: number): Tensor {
  const shape = x.shape.filter((_, i) => i !== axis);
  return new Tensor(x.data.slice(), shape);
}

function reshapeForReducedAxis(grad: Tensor, originalShape: readonly number[], axis: number): Tensor {
  const expectedShape = originalShape.filter((_, i) => i !== axis);
  if (!sameShape(grad.shape, expectedShape)) {
    throw new Error(`Reduced gradient shape [${grad.shape.join(", ")}] does not match expected [${expectedShape.join(", ")}].`);
  }
  const restoredShape = [...expectedShape.slice(0, axis), 1, ...expectedShape.slice(axis)];
  return new Tensor(grad.data.slice(), restoredShape);
}

function broadcastTo(x: Tensor, fromShape: readonly number[], toShape: readonly number[]): Tensor {
  if (!sameShape(fromShape, x.shape)) throw new Error("Internal broadcast source shape mismatch.");
  const out = new Float32Array(sizeOf(toShape));
  forEachIndex(toShape, (idx) => {
    out[offsetOf(idx, toShape)] = valueAtBroadcast(x, idx, toShape);
  });
  return new Tensor(out, toShape);
}

function reduceBroadcast(grad: Tensor, targetShape: readonly number[]): Tensor {
  if (sameShape(grad.shape, targetShape)) return grad.detach();
  const out = new Float32Array(sizeOf(targetShape));
  forEachIndex(grad.shape, (idx) => {
    const targetIdx = projectBroadcastIndex(idx, grad.shape, targetShape);
    const outIndex = offsetOf(targetIdx, targetShape);
    out[outIndex] = (out[outIndex] ?? 0) + (grad.data[offsetOf(idx, grad.shape)] ?? 0);
  });
  return new Tensor(out, targetShape);
}

function addNoGrad(a: Tensor, b: Tensor): Tensor {
  return binaryNoGrad(a, b, (x, y) => x + y);
}

function subNoGrad(a: Tensor, b: Tensor): Tensor {
  return binaryNoGrad(a, b, (x, y) => x - y);
}

function mulNoGrad(a: Tensor, b: Tensor): Tensor {
  return binaryNoGrad(a, b, (x, y) => x * y);
}

function divNoGrad(a: Tensor, b: Tensor): Tensor {
  return binaryNoGrad(a, b, (x, y) => x / y);
}

function negNoGrad(x: Tensor): Tensor {
  return new Tensor(mapData(x, (v) => -v), x.shape);
}

function binaryNoGrad(a: Tensor, b: Tensor, forward: (a: number, b: number) => number): Tensor {
  const shape = broadcastShape(a.shape, b.shape);
  const out = new Float32Array(sizeOf(shape));
  forEachIndex(shape, (idx) => {
    out[offsetOf(idx, shape)] = forward(valueAtBroadcast(a, idx, shape), valueAtBroadcast(b, idx, shape));
  });
  return new Tensor(out, shape);
}

function matmulNoGrad(a: Tensor, b: Tensor): Tensor {
  return matmul(a.detach(), b.detach()).detach();
}

function circularConvolveNoGrad(a: Tensor, b: Tensor): Tensor {
  assertSameVector("circularConvolve", a, b);
  const n = a.size;
  const out = new Float32Array(n);
  for (let t = 0; t < n; t++) {
    let acc = 0;
    for (let i = 0; i < n; i++) {
      acc += (a.data[i] ?? 0) * (b.data[mod(t - i, n)] ?? 0);
    }
    out[t] = acc;
  }
  return new Tensor(out, a.shape);
}

function circularCorrelateNoGrad(a: Tensor, b: Tensor): Tensor {
  assertSameVector("circularCorrelate", a, b);
  const n = a.size;
  const out = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    let acc = 0;
    for (let i = 0; i < n; i++) {
      acc += (a.data[i] ?? 0) * (b.data[mod(i - k, n)] ?? 0);
    }
    out[k] = acc;
  }
  return new Tensor(out, a.shape);
}

function transposeNoGrad(x: Tensor): Tensor {
  const [rows, cols] = x.shape as [number, number];
  const out = new Float32Array(x.size);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) out[j * rows + i] = x.data[i * cols + j] ?? 0;
  }
  return new Tensor(out, [cols, rows]);
}

function mapData(x: Tensor, fn: (value: number, index: number) => number): Float32Array {
  const out = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) out[i] = fn(x.data[i] ?? 0, i);
  return out;
}

export function sizeOf(shape: readonly number[]): number {
  return shape.reduce((acc, n) => acc * n, 1);
}

function stridesOf(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i] ?? 1;
  }
  return strides;
}

function offsetOf(index: readonly number[], shape: readonly number[]): number {
  if (shape.length === 0) return 0;
  const strides = stridesOf(shape);
  return index.reduce((acc, value, i) => acc + value * (strides[i] ?? 1), 0);
}

function forEachIndex(shape: readonly number[], fn: (index: number[]) => void): void {
  if (shape.length === 0) {
    fn([]);
    return;
  }
  const total = sizeOf(shape);
  for (let linear = 0; linear < total; linear++) {
    let rest = linear;
    const idx = new Array<number>(shape.length);
    for (let dim = shape.length - 1; dim >= 0; dim--) {
      const size = shape[dim] ?? 1;
      idx[dim] = rest % size;
      rest = Math.floor(rest / size);
    }
    fn(idx);
  }
}

function broadcastShape(a: readonly number[], b: readonly number[]): number[] {
  const rank = Math.max(a.length, b.length);
  const out = new Array<number>(rank);
  for (let i = 0; i < rank; i++) {
    const ad = a[a.length - 1 - i] ?? 1;
    const bd = b[b.length - 1 - i] ?? 1;
    if (ad !== bd && ad !== 1 && bd !== 1) throw new Error(`Shapes [${a.join(", ")}] and [${b.join(", ")}] are not broadcastable.`);
    out[rank - 1 - i] = Math.max(ad, bd);
  }
  return out;
}

function valueAtBroadcast(x: Tensor, outIndex: readonly number[], outShape: readonly number[]): number {
  const projected = projectBroadcastIndex(outIndex, outShape, x.shape);
  return x.data[offsetOf(projected, x.shape)] ?? 0;
}

function projectBroadcastIndex(index: readonly number[], fromShape: readonly number[], targetShape: readonly number[]): number[] {
  const offset = fromShape.length - targetShape.length;
  return targetShape.map((dim, i) => (dim === 1 ? 0 : index[i + offset] ?? 0));
}

function normalizeAxis(axis: number, rank: number): number {
  const normalized = axis < 0 ? rank + axis : axis;
  if (normalized < 0 || normalized >= rank) throw new Error(`Axis ${axis} is out of bounds for rank ${rank}.`);
  return normalized;
}

function assertSameVector(op: string, a: Tensor, b: Tensor): void {
  if (a.ndim !== 1 || b.ndim !== 1) throw new Error(`${op} currently supports rank-1 tensors.`);
  if (a.size !== b.size) throw new Error(`${op} shape mismatch: [${a.shape.join(", ")}] vs [${b.shape.join(", ")}].`);
}

function mod(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function replaceAt(values: readonly number[], index: number, value: number): number[] {
  return values.map((current, i) => (i === index ? value : current));
}

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((value, i) => value === b[i]);
}
