import puppeteer from 'puppeteer-core';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import net from 'node:net';

// Isolated browser state and intercepted API calls keep demonstration data out of real projects.
const chrome = process.env.CHROME_PATH || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser', 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find(existsSync);
if (!chrome) throw new Error('No se encontró Chrome/Chromium. Define CHROME_PATH con la ruta del ejecutable.');
const output = resolve(process.env.CAPTURE_DIR || `capturas/${new Date().toISOString().replace(/[:.]/g, '-')}`);
mkdirSync(output, { recursive: true });
const port = await new Promise((ok, fail) => { const s = net.createServer(); s.on('error', fail); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => ok(p)); }); });
const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], { stdio: ['ignore', 'pipe', 'pipe'] });
let serverLog = ''; server.stdout.on('data', x => serverLog += x); server.stderr.on('data', x => serverLog += x);
let browser;
try {
  const url = `http://127.0.0.1:${port}`;
  let ready = false;
  for (let i = 0; i < 100; i++) { if (server.exitCode !== null) break; try { if ((await fetch(url)).ok) { ready = true; break; } } catch {} await new Promise(r => setTimeout(r, 200)); }
  if (!ready) throw new Error(`No se pudo iniciar Vite. ${serverLog}`);
  const viewport = {
    width: Number(process.env.CAPTURE_WIDTH || 1600),
    height: Number(process.env.CAPTURE_HEIGHT || 1000),
    deviceScaleFactor: 1,
  };
  browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'], defaultViewport: viewport });
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('dialog', dialog => dialog.accept());
  await page.setRequestInterception(true);
  const appliedSplits=[];
  page.on('request', request => {
    if (request.url().includes('/api/backend')) {
      const action = JSON.parse(request.postData() || '{}').action;
      const report = {layout:'split-folders', samples:20, classes:{cats:10,dogs:10}, originalSplits:['test','train'], labelFiles:[], splitPolicy:'merge-and-resplit'};
      const body=JSON.parse(request.postData() || '{}');
      if(action==='data.pipeline.apply')appliedSplits.push(body.dataset.splits);
      const result = action === 'data.pipeline.apply' ? {...body.dataset,revision:(body.dataset.revision||1)+1,pipeline:body.pipeline} : action === 'data.kaggle.inspect' ? report : action === 'data.exists' ? {exists:true} : action === 'data.kaggle.import' ? {id:'kaggle-qa',source:'kaggle',kaggleHandle:'owner/images',importReport:report,samples:20,inputShape:[3,16,16],outputShape:[2],classes:['cats','dogs'],description:'Kaggle QA',splits:{train:70,validation:15,test:15},splitCounts:{train:14,validation:3,test:3},splitSource:'generated',options:{channels:3,resolution:16}} : action === 'project.list' ? [] : action === 'system.gpu' ? { cuda: false, name: 'CPU', total_vram: 0, allocated_vram: 0 } : null;
      request.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(result === null ? { ok: false, error: { message: 'No disponible en capturas' } } : { ok: true, result }) });
    } else if (request.url().includes('/api/training-events')) request.abort();
    else request.continue();
  });
  const project = { id: 'captura-demo', name: 'Proyecto de demostración', description: 'Datos de ejemplo para capturas', path: 'demo', createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), architecture: 'mlp', taskId: 'tabular.classification' };
  await page.evaluateOnNewDocument(project => { localStorage.setItem('mb-language', 'es'); localStorage.setItem('mb-theme', 'dark'); if (!localStorage.getItem('mb-projects')) localStorage.setItem('mb-projects', JSON.stringify([project])); }, project);
  await page.goto(url, { waitUntil: 'networkidle2' });
  const shot = async name => { await page.evaluate(() => document.fonts.ready); await new Promise(r => setTimeout(r, 350)); await page.screenshot({ path: resolve(output, `${name}.png`) }); console.log(`${name}.png`); };
  await page.waitForSelector('.project-row'); await shot('01-proyectos');
  await page.click('.project-row'); await page.waitForSelector('.model-catalog-page'); await shot('02-modelos');
  // Exercise the CNN classification data flow in an isolated browser.
  // The current catalog lists CNN 2D at index 5; assert its name below.
  for (let i = 5; i < 6; i++) {
    await page.waitForSelector('.model-catalog-page');
    await page.evaluate(index => document.querySelectorAll('.catalog-architecture-list > button')[index].click(), i);
    const name = await page.$eval('.architecture-detail-panel h2', e => e.textContent.trim());
    if(name !== 'CNN 2D') throw new Error('CNN 2D catalog location changed');
    const slug = `${String(i + 1).padStart(2, '0')}-${name.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')}`;
    await shot(`modelos/${slug}-catalogo`.replace('/', '-'));
    await page.$eval('.available-tasks button', e => e.click());
    await new Promise(resolve=>setTimeout(resolve,100));
    if(await page.$('.app-dialog .primary')) await page.click('.app-dialog .primary');
    await page.evaluate(() => document.querySelectorAll('.sidebar nav button')[2].click());
    await page.waitForSelector('.react-flow__node');
    await shot(`modelo-${slug}-constructor`);
    if (i === 5) {
      await page.evaluate(() => document.querySelectorAll('.sidebar nav button')[1].click());
      await page.waitForSelector('.source-choice-card');
      await page.evaluate(()=>[...document.querySelectorAll('.source-choice-card')].find(button=>button.textContent.includes('Kaggle')).click());
      await page.waitForSelector('.kaggle-source');
      await page.type('.kaggle-source input', 'owner/images');
      await shot('kaggle-01-source');
      await page.click('.kaggle-source > button.secondary');
      await page.waitForSelector('.kaggle-report');
      await shot('kaggle-02-inspection');
      const text = await page.$eval('.kaggle-report', e => e.textContent);
      if (!text.includes('cats: 10') || !text.includes('test, train')) throw new Error('Missing class/split report');
      await page.click('.kaggle-source > button.primary');
      await page.waitForSelector('.data-sidebar .kaggle-report');
      await shot('kaggle-03-imported');
      await page.click('.pipeline-actions button.primary');
      await page.waitForFunction(() => document.querySelector('.pipeline-actions button.primary')?.disabled && !document.querySelector('.pipeline-actions .dirty'));
      await page.focus('.split-handle');
      await page.keyboard.press('ArrowLeft');

      await page.waitForFunction(() => !document.querySelector('.pipeline-actions button.primary')?.disabled);
      await page.click('.pipeline-actions button.primary');
      await page.waitForFunction(() => document.querySelector('.pipeline-actions button.primary')?.disabled && !document.querySelector('.pipeline-actions .dirty'));
      if(appliedSplits.length!==2 || appliedSplits[1].train!==69)throw new Error('Split change was not reapplied');
      console.log('Kaggle UI: import and subsequent split changes passed');

    }

    await page.evaluate(() => document.querySelectorAll('.sidebar nav button')[0].click());
  }
  console.log(`Kaggle UI screenshots: ${output}`);
  if (pageErrors.length) throw new Error(`Errores del navegador: ${pageErrors.join('\n')}`);
} finally {
  if (browser) await browser.close();
  server.kill();
}
