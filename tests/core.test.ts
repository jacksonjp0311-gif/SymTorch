import { describe, expect, it } from "vitest";
import { add, matmul, mean, mul, softmax, tensor } from "@symtorch/core";

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
});

