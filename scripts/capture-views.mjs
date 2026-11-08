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
  page.on('request', request => {
    if (request.url().includes('/api/backend')) {
      const action = JSON.parse(request.postData() || '{}').action;
      const result = action === 'data.exists' ? { exists: true } : action === 'data.catalog' ? [] : action === 'project.list' ? [] : action === 'system.gpu' ? { cuda: false, name: 'CPU', total_vram: 0, allocated_vram: 0 } : null;
      request.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(result === null ? { ok: false, error: { message: 'No disponible en capturas' } } : { ok: true, result }) });
    } else if (request.url().includes('/api/training-events')) request.abort();
    else request.continue();
  });
  const project = { id: 'captura-demo', name: 'Proyecto de demostración', description: 'Datos de ejemplo para capturas', path: 'demo', createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), architecture: 'mlp', taskId: 'tabular.classification' };
  await page.evaluateOnNewDocument(project => { localStorage.setItem('mb-language', 'es'); localStorage.setItem('mb-theme', 'dark'); localStorage.setItem('mb-accent', 'orange'); if (!localStorage.getItem('mb-projects')) localStorage.setItem('mb-projects', JSON.stringify([project])); }, project);
  await page.goto(url, { waitUntil: 'networkidle2' });
  const shot = async name => { await page.evaluate(() => document.fonts.ready); await new Promise(r => setTimeout(r, 350)); await page.screenshot({ path: resolve(output, `${name}.png`) }); console.log(`${name}.png`); };
  await page.waitForSelector('.project-row'); await shot('01-proyectos');
  await page.click('.project-row'); await page.waitForSelector('.model-catalog-page'); await shot('02-modelos');
  // Capture the actual catalog and generated graph for every architecture, in catalog order.
  const count = await page.$$eval('.catalog-architecture-list > button', buttons => buttons.length);
  for (let i = 0; i < count; i++) {
    await page.waitForSelector('.model-catalog-page');
    await page.evaluate(index => document.querySelectorAll('.catalog-architecture-list > button')[index].click(), i);
    const name = await page.$eval('.architecture-detail-panel h2', e => e.textContent.trim());
    const slug = `${String(i + 1).padStart(2, '0')}-${name.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')}`;
    await shot(`modelos/${slug}-catalogo`.replace('/', '-'));
    await page.$eval('.available-tasks button', e => e.click());
    await new Promise(resolve=>setTimeout(resolve,100));
    if(await page.$('.app-dialog .primary')) await page.click('.app-dialog .primary');
    await page.evaluate(() => document.querySelectorAll('.sidebar nav button')[2].click());
    await page.waitForSelector('.react-flow__node');
    await shot(`modelo-${slug}-constructor`);
    await page.evaluate(() => document.querySelectorAll('.sidebar nav button')[0].click());
  }
  // Add a small, explicit demonstration dataset so the remaining screens are populated.
  await page.evaluate(() => {
    const key = 'mb-state-captura-demo'; const state = JSON.parse(localStorage.getItem(key));
    state.dataset = { id: 'demo-data', source: 'synthetic', samples: 450, inputShape: [4], outputShape: [3], classes: ['A', 'B', 'C'], description: 'Datos de demostración', splits: { train: 70, validation: 15, test: 15 }, preview: { type: 'tabular', columns: ['var_1', 'var_2', 'var_3', 'var_4'], items: Array.from({ length: 6 }, (_, i) => ({ features: [i, i + 0.2, i + 0.4, i + 0.6], target: ['A', 'B', 'C'][i % 3] })) } };
    const run = state.trainingRuns.find(item => item.id === state.activeRunId);
    run.status = 'completed'; run.result = { status: 'completed' };
    run.history = Array.from({ length: 15 }, (_, i) => ({ epoch: i + 1, trainLoss: 1.2 * Math.exp(-i / 5) + 0.1, valLoss: 1.3 * Math.exp(-i / 5.5) + 0.15, metric: 0.7 * Math.exp(-i / 6) + 0.1 }));
    state.step = 'data'; localStorage.setItem(key, JSON.stringify(state));
  });
  await page.reload({ waitUntil: 'networkidle2' }); await page.waitForSelector('.project-row'); await page.click('.project-row');
  await page.waitForSelector('.sidebar'); await shot('03-datos');
  await page.evaluate(() => document.querySelectorAll('.sidebar nav button')[2].click()); await page.waitForSelector('.react-flow__node'); await shot('04-constructor');
  await page.evaluate(() => document.querySelectorAll('.sidebar nav button')[3].click()); await page.waitForSelector('.sidebar nav button.active'); await shot('05-entrenamiento');
  await page.evaluate(() => document.querySelectorAll('.sidebar nav button')[4].click()); await page.waitForSelector('.inference-work-area'); await shot('06-inferencia');
  console.log(`\nCapturas guardadas en: ${output}`);
  if (pageErrors.length) throw new Error(`Errores del navegador: ${pageErrors.join('\n')}`);
} finally {
  if (browser) await browser.close();
  server.kill();
}
