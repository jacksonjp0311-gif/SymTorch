import { describe, expect, it } from "vitest";
import { add, clip, logsumexp, matmul, mean, mul, pow, sigmoid, softmax, sqrt, tanh, Tensor, tensor } from "@symtorch/core";

describe("@symtorch/core", () => {
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
