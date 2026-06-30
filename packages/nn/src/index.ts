import { add, exp, log, logSoftmax, matmul, mean, mul, pow, randn, relu, sigmoid, sqrt, sub, sum, Tensor, tensor, zeros, mul as coreMul, div as coreDiv } from "@symtorch/core";

export class Parameter extends Tensor {
  constructor(data: Tensor | readonly number[], shape?: readonly number[]) {
    const source = data instanceof Tensor ? data : tensor(data, shape ? { shape } : {});
    super(source.data.slice(), source.shape, { requiresGrad: true });
  }
}

export abstract class Module {
  abstract forward(input: Tensor): Tensor;

  parameters(): Parameter[] {
    const found: Parameter[] = [];
    for (const value of Object.values(this)) collectParameters(value, found);
    return found;
  }

  zeroGrad(): void {
    for (const param of this.parameters()) param.zeroGrad();
  }
}

export class Linear extends Module {
  readonly weight: Parameter;
  readonly bias: Parameter | null;

  constructor(readonly inFeatures: number, readonly outFeatures: number, options: { bias?: boolean } = {}) {
    super();
    const scale = Math.sqrt(2 / Math.max(1, inFeatures));
    this.weight = new Parameter(mul(randn([inFeatures, outFeatures]), scale));
    this.bias = options.bias === false ? null : new Parameter(zeros([outFeatures]));
  }

  forward(input: Tensor): Tensor {
    const y = matmul(input, this.weight);
    return this.bias ? add(y, this.bias) : y;
  }
}

export class Sequential extends Module {
  constructor(readonly layers: readonly Module[]) {
    super();
  }

  forward(input: Tensor): Tensor {
    return this.layers.reduce((value, layer) => layer.forward(value), input);
  }

  override parameters(): Parameter[] {
    return this.layers.flatMap((layer) => layer.parameters());
  }
}

export class ReLU extends Module {
  forward(input: Tensor): Tensor {
    return relu(input);
  }
}

export class Sigmoid extends Module {
  forward(input: Tensor): Tensor {
    return sigmoid(input);
  }
}

export class Dropout extends Module {
  private _training = false;

  constructor(readonly p = 0.5) {
    super();
    if (p < 0 || p >= 1) throw new Error(`Dropout probability must be in [0, 1), received ${p}.`);
  }

  set training(value: boolean) {
    this._training = value;
  }

  get training(): boolean {
    return this._training;
  }

  forward(input: Tensor): Tensor {
    if (!this._training || this.p === 0) return input;
    const scale = 1 / (1 - this.p);
    const mask = new Float32Array(input.size);
    for (let i = 0; i < input.size; i++) {
      mask[i] = Math.random() >= this.p ? scale : 0;
    }
    return mul(input, new Tensor(mask, input.shape));
  }
}

export class LayerNorm extends Module {
  readonly weight: Parameter;
  readonly bias: Parameter;

  constructor(readonly normalizedShape: number, readonly eps = 1e-5) {
    super();
    this.weight = new Parameter(Array.from({ length: normalizedShape }, () => 1), [normalizedShape]);
    this.bias = new Parameter(zeros([normalizedShape]));
  }

  forward(input: Tensor): Tensor {
    if (input.shape[input.shape.length - 1] !== this.normalizedShape) {
      throw new Error(`LayerNorm expected last dimension ${this.normalizedShape}, got [${input.shape.join(", ")}].`);
    }
    const axis = input.ndim - 1;
    const mu = mean(input, axis, true);
    const centered = sub(input, mu);
    const variance = mean(pow(centered, 2), axis, true);
    const normalized = centered.div(sqrt(add(variance, this.eps)));
    return add(mul(normalized, this.weight), this.bias);
  }
}

export function mseLoss(prediction: Tensor, target: Tensor): Tensor {
  const error = sub(prediction, target);
  return mean(mul(error, error));
}

export function binaryCrossEntropy(prediction: Tensor, target: Tensor): Tensor {
  const eps = tensor(1e-7);
  const one = tensor(1);
  const clipped = add(mul(prediction, 1 - 2e-7), eps);
  const loss = sub(tensor(0), add(mul(target, log(clipped)), mul(sub(one, target), log(sub(one, clipped)))));
  return mean(loss);
}

export function binaryCrossEntropyWithLogits(logits: Tensor, target: Tensor): Tensor {
  const zero = tensor(0);
  const maxPart = relu(logits);
  const absLogits = add(relu(logits), relu(sub(zero, logits)));
  const loss = add(sub(maxPart, mul(logits, target)), log(add(tensor(1), exp(sub(zero, absLogits)))));
  return mean(loss);
}

export function crossEntropyLoss(logits: Tensor, targetClassIndices: readonly number[]): Tensor {
  if (logits.ndim !== 2) throw new Error("crossEntropyLoss expects logits with shape [batch, classes].");
  const [batch, classes] = logits.shape as [number, number];
  if (targetClassIndices.length !== batch) {
    throw new Error(`Expected ${batch} target labels, got ${targetClassIndices.length}.`);
  }
  const labels = new Float32Array(batch * classes);
  for (let row = 0; row < batch; row++) {
    const label = targetClassIndices[row];
    if (label === undefined || label < 0 || label >= classes || !Number.isInteger(label)) {
      throw new Error(`Invalid class index ${String(label)} for ${classes} classes.`);
    }
    labels[row * classes + label] = 1;
  }
  const oneHot = new Tensor(labels, [batch, classes]);
  return mean(sub(tensor(0), sum(mul(oneHot, logSoftmax(logits, 1)), 1)));
}

export abstract class Optimizer {
  constructor(readonly params: readonly Parameter[]) {}
  abstract step(): void;

  zeroGrad(): void {
    for (const param of this.params) param.zeroGrad();
  }
}

export class SGD extends Optimizer {
  constructor(params: readonly Parameter[], readonly lr = 1e-2) {
    super(params);
  }

  step(): void {
    for (const param of this.params) {
      if (!param.grad) continue;
      for (let i = 0; i < param.data.length; i++) param.data[i] = (param.data[i] ?? 0) - this.lr * (param.grad.data[i] ?? 0);
    }
  }
}

export class Adam extends Optimizer {
  private readonly m: Float32Array[];
  private readonly v: Float32Array[];
  private t = 0;

  constructor(
    params: readonly Parameter[],
    readonly lr = 1e-3,
    readonly beta1 = 0.9,
    readonly beta2 = 0.999,
    readonly eps = 1e-8
  ) {
    super(params);
    this.m = params.map((p) => new Float32Array(p.size));
    this.v = params.map((p) => new Float32Array(p.size));
  }

  step(): void {
    this.t += 1;
    for (let p = 0; p < this.params.length; p++) {
      const param = this.params[p];
      if (!param?.grad) continue;
      const m = this.m[p];
      const v = this.v[p];
      if (!m || !v) continue;
      for (let i = 0; i < param.data.length; i++) {
        const g = param.grad.data[i] ?? 0;
        m[i] = this.beta1 * (m[i] ?? 0) + (1 - this.beta1) * g;
        v[i] = this.beta2 * (v[i] ?? 0) + (1 - this.beta2) * g * g;
        const mHat = (m[i] ?? 0) / (1 - this.beta1 ** this.t);
        const vHat = (v[i] ?? 0) / (1 - this.beta2 ** this.t);
        param.data[i] = (param.data[i] ?? 0) - this.lr * mHat / (Math.sqrt(vHat) + this.eps);
      }
    }
  }
}

function collectParameters(value: unknown, out: Parameter[]): void {
  if (value instanceof Parameter) {
    out.push(value);
    return;
  }
  if (value instanceof Module) {
    out.push(...value.parameters());
    return;
  }
  if (Array.isArray(value)) {
    for (const item of value) collectParameters(item, out);
  }
}
