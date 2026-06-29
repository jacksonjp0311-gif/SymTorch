import { tensor } from "@symtorch/core";
import { Linear, mseLoss, SGD } from "@symtorch/nn";

const x = tensor([0, 1, 2, 3], { shape: [4, 1] });
const y = tensor([1, 3, 5, 7], { shape: [4, 1] });

const model = new Linear(1, 1);
const optim = new SGD(model.parameters(), 0.05);

for (let step = 0; step < 120; step++) {
  optim.zeroGrad();
  const prediction = model.forward(x);
  const loss = mseLoss(prediction, y);
  loss.backward();
  optim.step();
  if (step % 30 === 0) console.log(`step=${step} loss=${loss.item().toFixed(6)}`);
}

console.log("prediction", model.forward(x).toArray().map((value) => Number(value.toFixed(3))));

