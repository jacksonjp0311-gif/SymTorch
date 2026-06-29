import { add, logsumexp, mean, mul, Tensor, tensor } from "@symtorch/core";
import { crossEntropyLoss, LayerNorm } from "@symtorch/nn";

type GradientCase = {
  name: string;
  values: readonly number[];
  shape: readonly number[];
  forward: (input: Tensor) => Tensor;
};

const EPSILON = 1e-3;
const TOLERANCE = 2e-2;

const cases: GradientCase[] = [
  {
    name: "axis reduction gradient check",
    values: [0.2, -0.4, 0.7, 1.1, -1.3, 0.5],
    shape: [2, 3],
    forward: (input) => add(
      add(squareMean(input.sum(0)), squareMean(input.sum(1))),
      add(squareMean(input.mean(0)), add(squareMean(input.mean(1)), add(squareMean(logsumexp(input, 0)), squareMean(logsumexp(input, 1)))))
    )
  },
  {
    name: "cross entropy gradient check",
    values: [1.5, -0.2, 0.3, -0.7, 0.8, 1.2],
    shape: [2, 3],
    forward: (input) => crossEntropyLoss(input, [0, 2])
  },
  {
    name: "LayerNorm gradient check",
    values: [1, 2, 3, 2, 4, 7],
    shape: [2, 3],
    forward: (input) => {
      const layer = new LayerNorm(3);
      return mean(mul(layer.forward(input), tensor([0.2, -0.4, 0.6], { shape: [3] })));
    }
  }
];

console.log("SymTorch Gradient Demo");
for (const item of cases) {
  assertGradientClose(item);
  console.log(`${item.name}: PASS`);
}

function assertGradientClose(item: GradientCase): void {
  const input = tensor(item.values, { shape: item.shape, requiresGrad: true });
  const loss = item.forward(input);
  loss.backward();
  const actual = Array.from(input.grad?.data ?? []);
  const expected = finiteDifference(item.values, item.shape, item.forward);
  if (actual.length !== expected.length) {
    throw new Error(`${item.name}: gradient length mismatch ${actual.length} !== ${expected.length}`);
  }
  for (let i = 0; i < actual.length; i++) {
    const delta = Math.abs((actual[i] ?? 0) - (expected[i] ?? 0));
    if (delta > TOLERANCE) {
      throw new Error(`${item.name}: gradient mismatch at ${i}: actual=${actual[i]} expected=${expected[i]} delta=${delta}`);
    }
  }
}

function finiteDifference(values: readonly number[], shape: readonly number[], forward: (input: Tensor) => Tensor): number[] {
  return values.map((_, index) => {
    const plus = [...values];
    const minus = [...values];
    plus[index]! += EPSILON;
    minus[index]! -= EPSILON;
    const plusLoss = forward(tensor(plus, { shape })).item();
    const minusLoss = forward(tensor(minus, { shape })).item();
    return (plusLoss - minusLoss) / (2 * EPSILON);
  });
}

function squareMean(input: Tensor): Tensor {
  return mean(mul(input, input));
}
