// Screenshot terminal HTML pages: node terminal_shots.js <htmlDir> <outDir> name1 name2 ...
const { chromium } = require("playwright");
const [htmlDir, outDir, ...names] = process.argv.slice(2);

(async () => {
  const browser = await chromium.launch({ executablePath: "/opt/pw-browsers/chromium" });
  const page = await browser.newPage({ viewport: { width: 1100, height: 800 }, deviceScaleFactor: 2 });
  for (const name of names) {
    await page.goto(`file://${htmlDir}/${name}.html`);
    await page.locator(".term").screenshot({ path: `${outDir}/${name}.png` });
    console.log("saved", `${outDir}/${name}.png`);
  }
  await browser.close();
})();
