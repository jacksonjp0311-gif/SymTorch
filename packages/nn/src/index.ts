import { add, log, matmul, mean, mul, randn, relu, sigmoid, sub, Tensor, tensor, zeros } from "@symtorch/core";

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
