// Capture one section of a dashboard page, scrolled into view.
// Usage: node screenshots_sections.js <outDir> [baseUrl]
const { chromium } = require("playwright");

const outDir = process.argv[2];
const base = process.argv[3] || "http://localhost:8501";
const shots = [
  // [page, heading to scroll to, output name]
  ["Model Evaluation", "Robustness: the saved models on three extra unseen wards", "dash_model_robustness"],
  ["Experiments", "E3 Zero-day: recall on an attack family never seen in training", "dash_zero_day"],
];

(async () => {
  // PW_CHROMIUM points at a specific Chromium build; otherwise Playwright's own browser is used
  const browser = await chromium.launch({ executablePath: process.env.PW_CHROMIUM || undefined });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1.5 });
  await page.goto(base, { waitUntil: "networkidle", timeout: 120000 });
  await page.waitForTimeout(6000);
  for (const [name, heading, out] of shots) {
    await page.getByText(name, { exact: true }).first().click();
    await page.waitForTimeout(9000);
    const h = page.getByText(heading, { exact: true }).first();
    await h.scrollIntoViewIfNeeded();
    await h.evaluate((el) => el.scrollIntoView({ block: "start" }));
    await page.waitForTimeout(2500);
    await page.screenshot({ path: `${outDir}/${out}.png` });
    console.log("saved", `${outDir}/${out}.png`);
  }
  await browser.close();
})();
