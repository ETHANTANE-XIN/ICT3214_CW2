// Screenshot terminal HTML pages: node terminal_shots.js <htmlDir> <outDir> name1 name2 ...
const path = require("path");
const { pathToFileURL } = require("url");
const { chromium } = require("playwright");
const [htmlDir, outDir, ...names] = process.argv.slice(2);

(async () => {
  // PW_CHROMIUM points at a specific Chromium build; otherwise Playwright's own browser is used
  const browser = await chromium.launch({ executablePath: process.env.PW_CHROMIUM || undefined });
  const page = await browser.newPage({ viewport: { width: 1100, height: 800 }, deviceScaleFactor: 2 });
  for (const name of names) {
    await page.goto(pathToFileURL(path.join(htmlDir, `${name}.html`)).href);
    await page.locator(".term").screenshot({ path: `${outDir}/${name}.png` });
    console.log("saved", `${outDir}/${name}.png`);
  }
  await browser.close();
})();
