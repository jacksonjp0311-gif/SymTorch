import { HolographicMemory, vectorSymbol } from "@symtorch/agent";

const memory = new HolographicMemory(5);
const riskRole = vectorSymbol([1, 0, 0, 0, 0]);
const approvedRole = vectorSymbol([0, 1, 0, 0, 0]);
const highRisk = vectorSymbol([0.8, 0.1, 0.1, 0, 0]);
const approved = vectorSymbol([0.1, 0.9, 0, 0, 0]);

memory.bind(riskRole, highRisk);
memory.bind(approvedRole, approved);

const recalledRisk = memory.recall(riskRole);
const recalledApproved = memory.recall(approvedRole);
const riskHit = memory.similarity(recalledRisk, highRisk);
const riskMiss = memory.similarity(recalledRisk, approved);
const approvedHit = memory.similarity(recalledApproved, approved);
const approvedMiss = memory.similarity(recalledApproved, highRisk);

console.log("SymTorch Holographic Memory Demo");
console.log("snapshot:");
console.log(JSON.stringify(memory.snapshot(), null, 2));
console.log("recall scores:");
console.log(JSON.stringify({
  riskHit: Number(riskHit.toFixed(4)),
  riskMiss: Number(riskMiss.toFixed(4)),
  approvedHit: Number(approvedHit.toFixed(4)),
  approvedMiss: Number(approvedMiss.toFixed(4))
}, null, 2));

if (riskHit <= riskMiss) throw new Error("Expected risk role to recall the risk value more strongly.");
if (approvedHit <= approvedMiss) throw new Error("Expected approved role to recall the approved value more strongly.");
