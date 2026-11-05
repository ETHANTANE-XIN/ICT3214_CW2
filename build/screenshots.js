// Capture dashboard pages for the report and slides.
// Usage: node screenshots.js <outDir> [baseUrl] [comma-separated page names]
const { chromium } = require("playwright");

const outDir = process.argv[2];
const base = process.argv[3] || "http://localhost:8501";
const pages = process.argv[4] ? process.argv[4].split(",")
  : ["Live Ward Monitor", "Model Evaluation", "Experiments", "Dataset Explorer", "Explainability"];

(async () => {
  // PW_CHROMIUM points at a specific Chromium build; otherwise Playwright's own browser is used
  const browser = await chromium.launch({ executablePath: process.env.PW_CHROMIUM || undefined });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1.5 });
  await page.goto(base, { waitUntil: "networkidle", timeout: 120000 });
  await page.waitForTimeout(6000);
  for (const name of pages) {
    await page.getByText(name, { exact: true }).first().click();
    await page.waitForTimeout(name === "Live Ward Monitor" ? 7000 : 9000);
    const file = `${outDir}/dash_${name.toLowerCase().replace(/ /g, "_")}.png`;
    await page.screenshot({ path: file, fullPage: name !== "Live Ward Monitor" });
    // the live page also gets a viewport-sized capture for the report
    console.log("saved", file);
  }
  await browser.close();
})();
