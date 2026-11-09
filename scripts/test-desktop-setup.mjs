import puppeteer from 'puppeteer-core';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import net from 'node:net';
import assert from 'node:assert/strict';

const chrome = process.env.CHROME_PATH || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser', 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'].find(existsSync);
if (!chrome) throw new Error('Chrome/Chromium no encontrado; configura CHROME_PATH.');
const port = await new Promise((resolve, reject) => { const s = net.createServer(); s.on('error', reject); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => resolve(p)); }); });
const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], { stdio: 'ignore' });
let browser;
try {
  const url = `http://127.0.0.1:${port}`;
  let ready = false;
  for (let i = 0; i < 100; i++) { try { if ((await fetch(url)).ok) { ready = true; break; } } catch {} await new Promise(r => setTimeout(r, 100)); }
  assert(ready, 'No se pudo iniciar Vite');
  browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  await page.evaluateOnNewDocument(() => {
    let callbacks = {}, next = 1;
    window.__attempts = [];
    window.__TAURI_INTERNALS__ = {
      transformCallback: fn => { const id = next++; callbacks[id] = fn; return id; },
      unregisterCallback: () => {},
      invoke: async (cmd, args) => {
        if (cmd === 'runtime_status') return { ready: false };
        if (cmd === 'plugin:event|listen') { window.__progress = callbacks[args.handler]; return 1; }
        if (cmd === 'prepare_runtime') {
          window.__attempts.push(args.backend);
          if (args.backend === 'cpu') throw new Error('Fallo de prueba');
          window.__progress({ payload: { stage: 'install', message: 'Instalando', completed: 23, total: 81, current: 'torch' } });
          return new Promise(() => {});
        }
        return null;
      },
    };
  });
  await page.goto(url, { waitUntil: 'networkidle0' });
  await page.waitForSelector('[role=radio]');
  assert.equal(await page.$eval('.setup-primary', button => button.disabled), true);
  const options = await page.$$('[role=radio]');
  await options[1].click();
  assert.equal(await options[1].evaluate(button => button.getAttribute('aria-checked')), 'true');
  await page.click('.setup-primary');
  await page.waitForSelector('.setup-error');
  await page.click('.setup-secondary');
  await page.waitForSelector('[role=radio]');
  await page.$eval('[role=radio]', button => button.click());
  await page.click('.setup-primary');
  await page.waitForFunction(() => document.body.innerText.includes('23 de 81'));
  assert.deepEqual(await page.evaluate(() => window.__attempts), ['cpu', 'nvidia']);
  assert.equal(await page.$eval('[role=progressbar]', bar => bar.getAttribute('aria-valuenow')), '28');
  assert.match(await page.$eval('.setup-current', node => node.textContent), /torch/);
  console.log('Desktop setup: selection, error recovery and component progress passed');
} finally {
  if (browser) await browser.close();
  server.kill();
}
