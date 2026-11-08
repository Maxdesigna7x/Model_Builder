// Real backend, isolated projects, no model downloads.
import puppeteer from 'puppeteer-core';
import {spawn,execFileSync} from 'node:child_process';
import {existsSync,mkdtempSync,mkdirSync,readFileSync,readdirSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import net from 'node:net';
const python=process.env.MODELBUILDER_PYTHON||'python3';
const chrome=process.env.CHROME_PATH||['/usr/bin/google-chrome','/usr/bin/chromium'].find(existsSync);
if(!chrome)throw Error('Set CHROME_PATH.');
const fixture=mkdtempSync(join(tmpdir(),'modelbuilder-dataset-ui-'));
execFileSync(python,['-c',"from pathlib import Path;from PIL import Image;import sys;r=Path(sys.argv[1]);[(p.parent.mkdir(parents=True,exist_ok=True),Image.new('RGB',(180,120),(i*20,40,80)).save(p)) for c in ['azul','rojo'] for i in range(4) for p in [r/'images'/c/f'{i}.png']]",fixture]);
const workspace=join(fixture,'workspace'),output=resolve(process.env.CAPTURE_DIR||'capturas/dataset-ui');mkdirSync(output,{recursive:true});
const port=await new Promise(ok=>{const s=net.createServer();s.listen(0,'127.0.0.1',()=>{const p=s.address().port;s.close(()=>ok(p));});});
const server=spawn(process.execPath,['node_modules/vite/bin/vite.js','--host','127.0.0.1','--port',String(port),'--strictPort'],{env:{...process.env,MODELBUILDER_PYTHON:python,MODELBUILDER_WORKSPACE:workspace},stdio:['ignore','pipe','pipe']});
let log='',browser;server.stdout.on('data',x=>log+=x);server.stderr.on('data',x=>log+=x);
const wait=async(fn,label)=>{for(let i=0;i<300;i++){if(await fn())return;await new Promise(r=>setTimeout(r,300));}throw Error(`Timed out: ${label}`);};
try{
 const url=`http://127.0.0.1:${port}`;await wait(async()=>{try{return(await fetch(url)).ok}catch{return false}},'Vite');
 browser=await puppeteer.launch({executablePath:chrome,headless:true,args:['--no-sandbox','--disable-dev-shm-usage'],defaultViewport:{width:1440,height:1000}});
 const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());
 await page.evaluateOnNewDocument(()=>{localStorage.setItem('mb-language','es');localStorage.setItem('mb-theme','dark')});await page.goto(url,{waitUntil:'domcontentloaded'});
 const click=async(selector,text)=>{await page.waitForFunction((s,t)=>[...document.querySelectorAll(s)].some(e=>e.textContent.trim()===t&&!e.disabled),{},selector,text);await page.evaluate((s,t)=>[...document.querySelectorAll(s)].find(e=>e.textContent.trim()===t&&!e.disabled).click(),selector,text)};
 const fill=async(selector,value)=>{const e=await page.waitForSelector(selector);await e.click({clickCount:3});await page.keyboard.press('Backspace');await e.type(String(value))};
 await page.click('[data-create-type=dataset]');await page.waitForSelector('.modal input');await page.type('.modal input','QA dataset');await page.click('.modal .modal-actions .primary');await page.waitForSelector('.dataset-shell');
 const root=()=>join(workspace,'projects',readdirSync(join(workspace,'projects'))[0]);
 const state=()=>{try{return JSON.parse(readFileSync(join(root(),'dataset-project.json'),'utf8'))}catch{return null}};
 await fill('.dataset-form input:not([type])',join(fixture,'images'));const checks=await page.$$('.dataset-form input[type=checkbox]');await checks[1].click();await click('.dataset-form button','Importar imágenes');
 await wait(()=>state()?.jobs?.some(j=>j.operation==='images'&&j.state==='completed'),'images');if(state().samples.length!==8||state().classes.length!==2)throw Error('Incorrect valid image/class counts');await page.waitForFunction(()=>document.querySelectorAll('.dataset-card').length===8);
 await click('.dataset-nav button','Anotar y extraer');await page.click('.dataset-card button');await page.waitForFunction(()=>{const im=document.querySelector('.dataset-viewer img');return im?.complete&&im.naturalWidth>0});await page.$eval('.dataset-viewer',e=>e.scrollIntoView({block:'center'}));const area=await page.$eval('.dataset-viewer',e=>{const r=e.getBoundingClientRect();return{x:r.x,y:r.y,w:r.width,h:r.height}});
 await page.mouse.move(area.x+area.w*.2,area.y+area.h*.2);await page.mouse.down();await page.mouse.move(area.x+area.w*.7,area.y+area.h*.7,{steps:8});await page.mouse.up();await click('.dataset-panel .dataset-row button','azul');await click('.dataset-panel button','Guardar cajas');
 await wait(()=>state()?.samples.some(s=>s.boxes.length===1),'box save');const box=state().samples.find(s=>s.boxes.length===1).boxes[0];if(box.xywh.some((v,i)=>Math.abs(v-[.45,.45,.5,.5][i])>.025))throw Error(`Box geometry: ${box.xywh}`);await page.screenshot({path:join(output,'anotador.png'),fullPage:true});
 await click('.dataset-nav button','Exportar');await click('.dataset-form button','Exportar');await wait(()=>state()?.exports?.length===1,'export');if(!existsSync(join(state().exports[0].path,'manifest.json')))throw Error('Missing export manifest');await page.screenshot({path:join(output,'exportar.png'),fullPage:true});
 await click('.dataset-nav button','← Inicio');await page.waitForSelector('.project-row');await page.click('.project-row');await page.waitForSelector('.dataset-shell');if(state().samples.length!==8||!state().samples.some(s=>s.boxes.length))throw Error('Reopen lost state');if(existsSync(join(root(),'state.json')))throw Error('Training autosave wrote dataset state');if(errors.length)throw Error(errors.join('\n'));
 console.log(`Dataset UI passed: real import, boxes, export, reopen. Fixtures: ${fixture}; screenshots: ${output}`);
}catch(e){if(browser){const pages=await browser.pages();const last=pages.at(-1);if(last){console.error(await last.$eval('body',e=>e.innerText));await last.screenshot({path:join(output,'failure.png'),fullPage:true});}}console.error(log);throw e}finally{if(browser)await browser.close();server.kill()}
