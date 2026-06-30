import { describe, expect, it } from "vitest";
import {
  add,
  bind,
  circularConvolve,
  circularCorrelate,
  clip,
  getBackend,
  getDefaultDevice,
  listBackends,
  logsumexp,
  matmul,
  mean,
  mul,
  pow,
  setDefaultDevice,
  sigmoid,
  softmax,
  sqrt,
  tanh,
  Tensor,
  tensor,
  unbind,
  withDefaultDevice
} from "@symtorch/core";

describe("@symtorch/core", () => {
  it("registers backend descriptors and routes tensor device defaults", () => {
    const backends = listBackends();

    expect(backends.map((backend) => backend.id)).toEqual(["cpu", "webgpu"]);
    expect(getBackend("cpu")).toMatchObject({ id: "cpu", status: "available" });
    expect(getBackend("webgpu")).toMatchObject({ id: "webgpu", status: "placeholder" });
    expect(getDefaultDevice()).toBe("cpu");

    const scoped = withDefaultDevice("webgpu", () => tensor([1, 2, 3]));
    expect(scoped.device).toBe("webgpu");
    expect(scoped.storage.kind).toBe("webgpu");
    expect(scoped.size).toBe(3);
    expect(getDefaultDevice()).toBe("cpu");

    setDefaultDevice("cpu");
    expect(tensor(1).device).toBe("cpu");
  });

  it("keeps readback explicit for non-CPU tensor storage", async () => {
    const cpu = tensor([1, 2], { shape: [2] });
    const gpu = tensor([1, 2], { shape: [2], device: "webgpu" });

    await expect(cpu.read()).resolves.toEqual(new Float32Array([1, 2]));
    await expect(cpu.toCPU()).resolves.toMatchObject({ device: "cpu" });
    expect(() => gpu.toArray()).toThrow("WebGPU storage is a placeholder");
    await expect(gpu.read()).rejects.toThrow("GPU readback is not implemented");
    await expect(gpu.toCPU()).rejects.toThrow("GPU readback is not implemented");
  });

  it("adds tensors with broadcasting", () => {
    const a = tensor([1, 2, 3, 4], { shape: [2, 2] });
    const b = tensor([10, 20], { shape: [2] });
    expect(add(a, b).toArray()).toEqual([11, 22, 13, 24]);
  });

  it("computes scalar gradients through elementwise ops", () => {
    const x = tensor([2, 3], { requiresGrad: true, shape: [2] });
    const y = mean(mul(x, x));
    y.backward();
    expect(x.grad?.toArray().map((value) => Number(value.toFixed(4)))).toEqual([2, 3]);
  });

  it("computes matmul gradients", () => {
    const x = tensor([1, 2], { requiresGrad: true, shape: [1, 2] });
    const w = tensor([3, 4], { requiresGrad: true, shape: [2, 1] });
    const y = matmul(x, w).sum();
    y.backward();
    expect(x.grad?.toArray()).toEqual([3, 4]);
    expect(w.grad?.toArray()).toEqual([1, 2]);
  });

  it("normalizes softmax rows", () => {
    const s = softmax(tensor([1, 2, 3], { shape: [1, 3] }));
    const total = s.toArray().reduce((acc, value) => acc + value, 0);
    expect(total).toBeCloseTo(1, 5);
  });

  it("computes axis-aware logsumexp", () => {
    const x = tensor([1, 2, 3, 4], { shape: [2, 2] });
    const y = logsumexp(x, 1);
    expect(y.shape).toEqual([2]);
    expect(y.toArray()[0]).toBeCloseTo(Math.log(Math.exp(1) + Math.exp(2)), 5);
    expect(y.toArray()[1]).toBeCloseTo(Math.log(Math.exp(3) + Math.exp(4)), 5);
  });

  it("matches finite-difference gradients for unary chains", () => {
    expectGradientClose(
      [0.2, -0.4, 0.7],
      [3],
      (x) => mean(add(pow(tanh(x), 2), sigmoid(x)))
    );
  });

  it("supports sqrt and clip gradients", () => {
    expectGradientClose([0.25, 1, 4], [3], (x) => mean(sqrt(x)));
    const x = tensor([-2, -0.5, 0.5, 2], { shape: [4], requiresGrad: true });
    clip(x, -1, 1).sum().backward();
    expect(x.grad?.toArray()).toEqual([0, 1, 1, 0]);
  });

  it("matches finite-difference gradients for broadcasted binary chains", () => {
    expectGradientClose(
      [1.2, -0.7, 0.3, 2.1],
      [2, 2],
      (x) => mean(mul(add(x, tensor([0.5, -1], { shape: [2] })), x))
    );
  });

  it("matches finite-difference gradients for matmul", () => {
    expectGradientClose(
      [1, -2, 0.5, 3],
      [2, 2],
      (x) => mean(matmul(x, tensor([0.2, -0.4, 1.1, 0.7], { shape: [2, 2] })))
    );
  });

  it("matches finite-difference gradients for axis reductions", () => {
    const values = [0.2, -0.4, 0.7, 1.1, -1.3, 0.5];
    const shape = [2, 3];
    expectGradientClose(values, shape, (x) => sumSquares(x.sum(0)));
    expectGradientClose(values, shape, (x) => sumSquares(x.sum(1)));
    expectGradientClose(values, shape, (x) => sumSquares(x.mean(0)));
    expectGradientClose(values, shape, (x) => sumSquares(x.mean(1)));
    expectGradientClose(values, shape, (x) => sumSquares(logsumexp(x, 0)));
    expectGradientClose(values, shape, (x) => sumSquares(logsumexp(x, 1)));
  });

  it("computes circular convolution and correlation for vector-symbolic binding", () => {
    const a = tensor([1, 2, 3], { shape: [3] });
    const b = tensor([4, 5, 6], { shape: [3] });

    expect(circularConvolve(a, b).toArray()).toEqual([31, 31, 28]);
    expect(circularCorrelate(circularConvolve(a, b), b).shape).toEqual([3]);
    expect(bind(a, b).toArray()).toEqual(circularConvolve(a, b).toArray());
    expect(unbind(bind(a, b), b).shape).toEqual([3]);
  });

  it("matches finite-difference gradients for circular binding ops", () => {
    expectGradientClose(
      [0.2, -0.4, 0.7, 1.1],
      [4],
      (x) => sumSquares(circularConvolve(x, tensor([0.5, -0.25, 0.75, 0.1], { shape: [4] })))
    );
    expectGradientClose(
      [0.2, -0.4, 0.7, 1.1],
      [4],
      (x) => sumSquares(circularCorrelate(x, tensor([0.5, -0.25, 0.75, 0.1], { shape: [4] })))
    );
  });

  it("computes batched matmul for rank-3 tensors", () => {
    const a = tensor([1, 2, 3, 4, 5, 6, 7, 8], { shape: [2, 2, 2] });
    const b = tensor([1, 0, 0, 1], { shape: [2, 2] });
    const result = matmul(a, b);
    expect(result.shape).toEqual([2, 2, 2]);
    expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it("computes batched matmul between two rank-3 tensors", () => {
    const a = tensor([1, 2, 3, 4], { shape: [2, 1, 2] });
    const b = tensor([1, 0, 0, 1, 2, 0, 0, 2], { shape: [2, 2, 2] });
    const result = matmul(a, b);
    expect(result.shape).toEqual([2, 1, 2]);
    expect(result.toArray()[0]).toBeCloseTo(1, 5);
    expect(result.toArray()[1]).toBeCloseTo(2, 5);
    expect(result.toArray()[2]).toBeCloseTo(3, 5);
    expect(result.toArray()[3]).toBeCloseTo(4, 5);
  });

  it("computes batched matmul between rank-2 weight and rank-3 input", () => {
    const input = tensor([1, 2, 3, 4], { shape: [2, 1, 2] });
    const weight = tensor([1, 1, 1, -1], { shape: [2, 2] });
    const result = matmul(input, weight);
    expect(result.shape).toEqual([2, 1, 2]);
    expect(result.toArray()[0]).toBeCloseTo(3, 5);
    expect(result.toArray()[1]).toBeCloseTo(-1, 5);
    expect(result.toArray()[2]).toBeCloseTo(7, 5);
    expect(result.toArray()[3]).toBeCloseTo(-1, 5);
  });

  it("backpropagates through batched matmul", () => {
    const a = tensor([1, 2, 3, 4], { requiresGrad: true, shape: [2, 1, 2] });
    const b = tensor([1, 0, 0, 1], { requiresGrad: true, shape: [2, 2] });
    const result = matmul(a, b);
    result.sum().backward();
    expect(a.grad?.shape).toEqual([2, 1, 2]);
    expect(b.grad?.shape).toEqual([2, 2]);
    expect(a.grad?.toArray().reduce((acc, v) => acc + v, 0)).not.toBe(0);
    expect(b.grad?.toArray().reduce((acc, v) => acc + v, 0)).not.toBe(0);
  });

  it("matches finite-difference gradients for batched matmul", () => {
    const w = tensor([0.2, -0.4, 1.1, 0.7], { shape: [2, 2] });
    expectGradientClose(
      [1, -2, 0.5, 3],
      [2, 1, 2],
      (x) => mean(matmul(x, w))
    );

    expectGradientClose(
      [0.3, -0.1, 0.8, -0.5, 0.2, 0.9, -0.3, 0.6],
      [2, 2, 2],
      (x) => mean(matmul(x, tensor([0.5, -0.2, 0.1, 0.8], { shape: [2, 2] })))
    );
  });

  it("throws for rank-1 matmul inputs", () => {
    const a = tensor([1, 2], { shape: [2] });
    const b = tensor([3, 4], { shape: [2] });
    expect(() => matmul(a, b)).toThrow("at least rank-2");
  });
});

function expectGradientClose(values: readonly number[], shape: readonly number[], fn: (x: Tensor) => Tensor): void {
  const x = tensor(values, { shape, requiresGrad: true });
  const y = fn(x);
  y.backward();
  const analytic = x.grad?.toArray();
  if (!analytic) throw new Error("Expected analytic gradient.");
  const numeric = finiteDifference(values, shape, fn);
  expect(analytic.length).toBe(numeric.length);
  for (let i = 0; i < analytic.length; i++) {
    expect(analytic[i]).toBeCloseTo(numeric[i] ?? 0, 2);
  }
}

function finiteDifference(values: readonly number[], shape: readonly number[], fn: (x: Tensor) => Tensor): number[] {
  const eps = 1e-2;
  return values.map((_, i) => {
    const plus = [...values];
    const minus = [...values];
    plus[i] = (plus[i] ?? 0) + eps;
    minus[i] = (minus[i] ?? 0) - eps;
    const hi = fn(tensor(plus, { shape })).item();
    const lo = fn(tensor(minus, { shape })).item();
    return (hi - lo) / (2 * eps);
  });
}

function sumSquares(x: Tensor): Tensor {
  return mean(mul(x, x));
}
