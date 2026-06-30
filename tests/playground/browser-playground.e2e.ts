import { expect, test } from "@playwright/test";

test("browser playground trains, exports, imports, and records decisions", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Case Escalation" })).toBeVisible();
  await expect(page.locator("#diagnostics")).toContainText("Rule validation: PASS");
  await expect(page.locator("#decisionList")).toContainText("case-hot");
  await expect(page.locator("#policyHealth")).toContainText("symtorch.policyBundle.v1");
  await expect(page.locator("#policyHealth")).toContainText("PASS");

  await page.locator("#scenarioSelect").selectOption("fraud-review");
  await expect(page.getByRole("heading", { name: "Fraud Review" })).toBeVisible();
  await expect(page.locator("#decisionList")).toContainText("txn-hot");

  await page.getByRole("button", { name: "Train High Risk" }).click();
  await expect(page.locator("#trainingStats")).toContainText("threshold:");
  await expect(page.locator("#trainingStats")).toContainText("epochs: 100");
  await expect(page.locator("#trainingHistory span")).toHaveCount(12);
  await expect(page.locator("#traceOutput")).toContainText("high_risk");

  await page.getByRole("button", { name: "Export", exact: true }).click();
  const exported = await page.locator("#stateBuffer").inputValue();
  expect(exported).toContain("symtorch.playground.v1");
  expect(exported).toContain("symtorch.trainingRun.v1");
  expect(exported).toContain("fraud-review");
  expect(exported).toContain("trainingExamples");

  await page.getByRole("button", { name: "Export Scenario" }).click();
  const exportedScenario = await page.locator("#stateBuffer").inputValue();
  expect(exportedScenario).toContain("symtorch.scenario.v1");
  expect(exportedScenario).toContain("Fraud Review");

  await page.getByRole("button", { name: "Export Bundle" }).click();
  const exportedBundle = await page.locator("#stateBuffer").inputValue();
  expect(exportedBundle).toContain("symtorch.policyBundle.v1");
  expect(exportedBundle).toContain("\"hash\"");

  await page.locator("#ruleSource").fill("escalate(X) :- missing_predicate(X).");
  await page.getByRole("button", { name: "Evaluate" }).click();
  await expect(page.locator("#diagnostics")).toContainText("missing_predicate");

  await page.locator("#stateBuffer").fill(exportedBundle);
  await page.getByRole("button", { name: "Import" }).click();
  await expect(page.locator("#stateStatus")).toContainText("Imported policy bundle.");
  await expect(page.locator("#diagnostics")).toContainText("Rule validation: PASS");
  await expect(page.locator("#policyHealth")).toContainText("PASS");

  await page.locator("#stateBuffer").fill(exportedScenario);
  await page.getByRole("button", { name: "Import" }).click();
  await expect(page.locator("#stateStatus")).toContainText("Imported scenario contract.");
  await expect(page.locator("#diagnostics")).toContainText("Rule validation: PASS");

  await page.locator("#stateBuffer").fill(exported);
  await page.getByRole("button", { name: "Import" }).click();
  await expect(page.locator("#stateStatus")).toContainText("Imported playground state.");
  await expect(page.locator("#trainingStats")).toContainText("epochs: 100");

  await page.getByRole("button", { name: "Record Top 2" }).click();
  await expect(page.locator("#traceOutput")).toContainText("decision-1");
  await expect(page.locator("#traceOutput")).toContainText("txn-hot");
  await expect(page.locator("#traceOutput")).toContainText("\"ok\": true");
  await expect(page.locator("#policyHealth")).toContainText("Replay");
});
