// npm install --no-save playwright; npx playwright install chromium
// node scripts/screenshot_revision_casebook.cjs
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');
const { pathToFileURL } = require('url');
(async () => {
 const root=path.resolve(__dirname,'..');
 const pages=path.join(root,'examples/revision_casebook/pages');
 const output=path.join(root,'examples/revision_casebook/screenshots');
 fs.mkdirSync(output,{recursive:true});
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1500,height:1000},deviceScaleFactor:1});
 for(const name of fs.readdirSync(pages).filter(s=>s.endsWith('.html')).sort()){
  await page.goto(pathToFileURL(path.join(pages,name)).href);
  await page.screenshot({path:path.join(output,name.replace('.html','.png')),fullPage:true});
  console.log(name);
 }
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
