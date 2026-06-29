import { describe, expect, it } from "vitest";
import { logSoftmax, mean, Tensor, tensor } from "@symtorch/core";
import { binaryCrossEntropyWithLogits, crossEntropyLoss, LayerNorm } from "@symtorch/nn";

describe("@symtorch/nn", () => {
  it("computes cross entropy from logits using stable log-softmax", () => {
    const logits = tensor([2, 1, 0, 0, 1, 2], { shape: [2, 3] });
    const loss = crossEntropyLoss(logits, [0, 2]);
    const logProbs = logSoftmax(logits, 1).toArray();
    const expected = -(logProbs[0]! + logProbs[5]!) / 2;
    expect(loss.item()).toBeCloseTo(expected, 5);
  });

  it("backpropagates through cross entropy logits", () => {
    const logits = tensor([2, 1, 0, 0, 1, 2], { shape: [2, 3], requiresGrad: true });
    crossEntropyLoss(logits, [0, 2]).backward();
    expect(logits.grad?.shape).toEqual([2, 3]);
    expect(logits.grad?.toArray().reduce((acc, value) => acc + value, 0)).toBeCloseTo(0, 5);
  });

  it("computes stable BCE with logits for extreme scores", () => {
    const logits = tensor([50, -50, 0], { shape: [3], requiresGrad: true });
    const target = tensor([1, 0, 1], { shape: [3] });
    const loss = binaryCrossEntropyWithLogits(logits, target);
    expect(Number.isFinite(loss.item())).toBe(true);
    loss.backward();
    expect(logits.grad?.toArray().every(Number.isFinite)).toBe(true);
  });

  it("normalizes the final dimension with LayerNorm", () => {
    const layer = new LayerNorm(3);
    const input = tensor([1, 2, 3, 2, 4, 6], { shape: [2, 3], requiresGrad: true });
    const output = layer.forward(input);
    expect(output.shape).toEqual([2, 3]);
    const rowMeans = [mean(new Tensor(output.data.slice(0, 3), [3])).item(), mean(new Tensor(output.data.slice(3, 6), [3])).item()];
    expect(rowMeans[0]).toBeCloseTo(0, 5);
    expect(rowMeans[1]).toBeCloseTo(0, 5);
    output.sum().backward();
    expect(input.grad?.shape).toEqual([2, 3]);
    expect(layer.weight.grad?.shape).toEqual([3]);
    expect(layer.bias.grad?.shape).toEqual([3]);
  });
});
