import assert from 'node:assert/strict';
import puppeteer from 'puppeteer-core';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import net from 'node:net';

const chrome = process.env.CHROME_PATH || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser', 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find(existsSync);
if (!chrome) throw new Error('No se encontró Chrome/Chromium. Define CHROME_PATH con la ruta del ejecutable.');
const output = resolve(process.env.CAPTURE_DIR || 'capturas/groups-minimap');
mkdirSync(output, { recursive: true });
const port = await new Promise((ok, fail) => { const server = net.createServer(); server.on('error', fail); server.listen(0, '127.0.0.1', () => { const value = server.address().port; server.close(() => ok(value)); }); });
const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], { stdio: ['ignore', 'pipe', 'pipe'] });
let serverLog = '';
server.stdout.on('data', chunk => serverLog += chunk);
server.stderr.on('data', chunk => serverLog += chunk);
let browser;

const input = (id, blockType, label, properties = {}, position = { x: 0, y: 0 }) => ({ id, type: 'modelNode', position, data: { blockType, label, category: blockType === 'conv2d' ? 'Convolution' : 'Model', properties } });
const graphNodes = [
  input('input', 'input', 'Input', {}, { x: 40, y: 170 }),
  input('conv1', 'conv2d', 'Conv A', { in_channels: 3, out_channels: 8, kernel_size: 3, stride: 1, padding: 1 }, { x: 300, y: 170 }),
  input('relu1', 'relu', 'ReLU A', {}, { x: 540, y: 170 }),
  input('conv2', 'conv2d', 'Conv B', { in_channels: 8, out_channels: 8, kernel_size: 3, stride: 1, padding: 1 }, { x: 780, y: 170 }),
  input('pool', 'adaptiveavgpool2d', 'Pool', { output_size: 1 }, { x: 1020, y: 170 }),
  input('flatten', 'flatten', 'Flatten', {}, { x: 1260, y: 170 }),
  input('head', 'linear', 'Classifier', { in_features: 8, out_features: 2 }, { x: 1500, y: 170 }),
  input('output', 'output', 'Output', { output_name: 'classes' }, { x: 1740, y: 170 }),
];
const graphEdges = graphNodes.slice(0, -1).map((node, index) => ({ id: `${node.id}-${graphNodes[index + 1].id}`, source: node.id, target: graphNodes[index + 1].id, type: 'smoothstep' }));
const project = { id: 'editor-training-qa', name: 'Editor QA', description: 'Isolated UI fixtures', path: 'editor-training-qa', createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), architecture: 'cnn', taskId: 'image.classification' };
const model = (id, name) => ({ id, name, architecture: 'cnn', nodes: graphNodes.map(node => ({ ...node, position: { ...node.position } })), edges: graphEdges.map(edge => ({ ...edge })), graphValid: false });
const history = [
  { epoch: 1, trainLoss: 0.92, valLoss: 1.28, metric: 0.42, trainMetric: 0.64, valMetric: 0.42, learningRate: 0.001 },
  { epoch: 2, trainLoss: 0.51, valLoss: 0.94, metric: 0.61, trainMetric: 0.83, valMetric: 0.61, learningRate: 0.001 },
  { epoch: 3, trainLoss: 0.30, valLoss: 0.73, metric: 0.74, trainMetric: 0.91, valMetric: 0.74, learningRate: 0.001 },
];
const legacyHistory = [{ epoch: 1, trainLoss: 0.8, valLoss: 1.1, metric: 0.52 }, { epoch: 2, trainLoss: 0.5, valLoss: 0.9, metric: 0.68 }];
const runs = [
  { id: 'run-live', name: 'Run live', status: 'draft', config: { epochs: 2 }, history: [], result: null, createdAt: new Date().toISOString(), modelId: 'model-a', modelTask: 'image.classification', modelArchitecture: 'cnn', modelInputShape: [3, 16, 16], modelGraph: { nodes: graphNodes, edges: graphEdges } },
  { id: 'run-history', name: 'Run with train series', status: 'completed', config: {}, history, result: { runId: 'history-run', status: 'completed', metricName: 'accuracy', history }, createdAt: new Date().toISOString(), modelId: 'model-b', modelGraph: { nodes: graphNodes, edges: graphEdges } },
  { id: 'run-legacy', name: 'Legacy run', status: 'completed', config: {}, history: legacyHistory, result: { runId: 'legacy-run', status: 'completed', metricName: 'accuracy', history: legacyHistory }, createdAt: new Date().toISOString(), modelId: 'model-b', modelGraph: { nodes: graphNodes, edges: graphEdges } },
  { id: 'run-extra', name: 'Extra draft', status: 'draft', config: {}, history: [], result: null, createdAt: new Date().toISOString() },
];
const state = {
  step: 'builder', task: 'image.classification', architecture: 'cnn',
  dataset: { id: 'qa-data', source: 'synthetic', samples: 60, inputShape: [3, 16, 16], outputShape: [2], classes: ['cats', 'dogs'], splits: { train: 70, validation: 15, test: 15 }, options: { channels: 3, resolution: 16 } },
  nodes: graphNodes, edges: graphEdges, models: [model('model-a', 'Model A'), model('model-b', 'Model B'), model('model-c', 'Model C'), model('model-d', 'Model D')], activeModelId: 'model-a', graphValid: false,
  trainingRuns: runs, activeRunId: 'run-live',
};

const trainingStarts=[];
state.dataset.imageResolution={count:60,min:{width:60,height:71},mean:{width:402.81,height:361.82},max:{width:500,height:500},nonSquareCount:55,unavailable:0};
state.dataset.preview={type:'image',items:Array.from({length:4},(_,i)=>({url:'data:image/svg+xml,'+encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120"><rect width="120" height="120" fill="#34495e"/><circle cx="60" cy="55" r="30" fill="#fb923c"/><text x="60" y="105" text-anchor="middle" fill="white">Sample ${i+1}</text></svg>`),label:'cats'}))};
try {
 const url=`http://127.0.0.1:${port}`;
 for(let i=0;i<100;i++){try{if((await fetch(url)).ok)break}catch{}await new Promise(r=>setTimeout(r,200));}
 browser=await puppeteer.launch({executablePath:chrome,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 const page=await browser.newPage();const errors=[];page.on('pageerror',e=>{errors.push(e.message);console.log('PAGEERROR',e.message)});
 await page.evaluateOnNewDocument((project,state)=>{localStorage.setItem('mb-language','es');localStorage.setItem('mb-accent','orange');localStorage.setItem('mb-theme','dark');if(!localStorage.getItem('mb-projects'))localStorage.setItem('mb-projects',JSON.stringify([project]));if(!localStorage.getItem(`mb-state-${project.id}`))localStorage.setItem(`mb-state-${project.id}`,JSON.stringify(state));class Stream{constructor(){window.fixtureStream=this}close(){}}window.EventSource=Stream},project,state);
 await page.setRequestInterception(true);page.on('request',r=>{if(r.url().includes('/api/backend')){const b=JSON.parse(r.postData()||'{}');let result=null;if(b.action==='project.list')result=[];if(b.action==='system.gpu')result={cuda:false,name:'CPU',total_vram:0,allocated_vram:0};if(b.action==='catalog.get')result={tasks:{},architectures:{},block_types:[]};if(b.action==='graph.validate')result={valid:true,parameters:1200,trainable_parameters:1200,node_shapes:{input:[3,16,16],conv1:[8,16,16],relu1:[8,16,16],conv2:[8,16,16],pool:[8,1,1],flatten:[8],head:[2],output:[2]},backward:true};if(b.action==='data.exists')result={exists:true};if(b.action==='data.preview')result={...state.dataset.preview,sampleIndices:[12,17,23,41]};if(b.action==='model.revision.save')result={revision:1};if(b.action==='model.catalog.guidance')result={reason:''};if(b.action==='model.catalog')result=[{id:'resnet18',name:'ResNet-18',repoId:'microsoft/resnet-18',size:'small',license:'apache-2.0',revision:'abc12345',library:'transformers',processor:'AutoImageProcessor',description:'Residual backbone',inputContract:'RGB 224×224'}];r.respond({status:200,contentType:'application/json',body:JSON.stringify({ok:true,result})})}else if(r.url().includes('/api/start-training')){trainingStarts.push(JSON.parse(r.postData()||'{}'));r.respond({status:200,contentType:'application/json',body:'{"started":true}'})}else if(r.url().includes('/api/training-events'))r.respond({status:200,contentType:'text/event-stream',body:''});else r.continue()});
 const shot=async name=>{await new Promise(r=>setTimeout(r,600));await page.screenshot({path:resolve(output,name+'.png')});const audit=await page.evaluate(()=>[...document.querySelectorAll('body *')].filter(e=>e.children.length===0&&e.textContent.trim()&&e.getBoundingClientRect().width>0&&parseFloat(getComputedStyle(e).fontSize)<12).map(e=>({text:e.textContent.slice(0,60),size:getComputedStyle(e).fontSize})));writeFileSync(resolve(output,name+'-font-audit.json'),JSON.stringify(audit,null,2));if(/vertical|horizontal/.test(name)){const ports=await page.evaluate(()=>[...document.querySelectorAll('.model-node')].flatMap(n=>{const r=n.getBoundingClientRect();return [...n.querySelectorAll('.model-node-port')].map(h=>{const b=h.getBoundingClientRect(),x=b.x+b.width/2,y=b.y+b.height/2;const edge=h.dataset.handlepos;return {edge,error:Math.abs(edge==='left'?x-r.left:edge==='right'?x-r.right:edge==='top'?y-r.top:y-r.bottom)}})}));writeFileSync(resolve(output,name+'-port-audit.json'),JSON.stringify(ports,null,2));if(ports.some(p=>p.error>1))throw Error('Port is inset: '+JSON.stringify(ports))}};
 await page.setViewport({width:1600,height:1000,deviceScaleFactor:1});await page.goto(url,{waitUntil:'networkidle2'});await shot('home');await page.click('.project-row');await page.waitForSelector('.builder-page');


 const settle=()=>new Promise(r=>setTimeout(r,650));
 const createGroup=async name=>{await page.click('[data-action="group"]');await page.waitForSelector('.app-dialog input');await new Promise(r=>setTimeout(r,100));await page.click('.app-dialog input');await page.keyboard.down('Control');await page.keyboard.press('a');await page.keyboard.up('Control');await page.keyboard.type(name);await page.click('.app-dialog .primary');await settle();};
 const selectPair=async(a,b)=>{await page.click('.react-flow__pane',{offset:{x:35,y:180}});await page.click(`.react-flow__node[data-id="${a}"]`);await page.keyboard.down('Shift');await page.click(`.react-flow__node[data-id="${b}"]`);await page.keyboard.up('Shift');await settle();};
 await settle();
 await selectPair('conv1','conv2');assert.equal(await page.$('[data-action="group"]'),null,'skipped middle layer disables grouping');
 await selectPair('conv1','relu1');assert.ok(await page.$('[data-action="group"]'));await createGroup('Inner');
 assert.equal(await page.$('[data-action="group"]'),null,'a selected group cannot group itself again');
 let inner=await page.$eval('.visual-group',e=>e.dataset.groupId);
 await page.click(`[data-group-id="${inner}"] .visual-group-title`);await page.keyboard.down('Shift');await page.click('.react-flow__node[data-id="conv2"]');await page.keyboard.up('Shift');await settle();
 assert.ok(await page.$('[data-action="group"]'));await createGroup('Outer');
 assert.equal(await page.$$('.visual-group').then(a=>a.length),2);
 const nested=await page.evaluate(()=>{const outer=document.querySelector('[data-depth="0"]').getBoundingClientRect(),inner=document.querySelector('[data-depth="1"]').getBoundingClientRect();return outer.x<inner.x&&outer.y<inner.y&&outer.right>inner.right&&outer.bottom>inner.bottom;});assert.ok(nested);
 assert.equal(await page.$('[data-action="group"]'),null);
 await selectPair('pool','flatten');await createGroup('Second');
 const outerId=await page.evaluate(()=>[...document.querySelectorAll('.visual-group')].find(g=>g.querySelector('button').textContent==='Outer').dataset.groupId);
 const secondId=await page.evaluate(()=>[...document.querySelectorAll('.visual-group')].find(g=>g.querySelector('button').textContent==='Second').dataset.groupId);
 await page.click(`[data-group-id="${outerId}"] button`);await page.keyboard.down('Shift');await page.click(`[data-group-id="${secondId}"] button`);await page.keyboard.up('Shift');await settle();
 await createGroup('Grouped groups');assert.equal(await page.$$('.visual-group').then(a=>a.length),4);assert.equal(await page.$('[data-action="group"]'),null);
 await page.click('[data-action="ungroup"]');await settle();assert.equal(await page.$$('.visual-group').then(a=>a.length),3);
 await shot('nested-groups');
 const map='.model-minimap>svg';const before=await page.$eval(map,e=>e.getAttribute('viewBox'));
 const allFit=await page.evaluate(()=>{const bounds=document.querySelector('.model-minimap>svg').getBoundingClientRect();return [...document.querySelectorAll('.minimap-node,.minimap-group')].every(e=>{const r=e.getBoundingClientRect();return r.left>=bounds.left-1&&r.right<=bounds.right+1&&r.top>=bounds.top-1&&r.bottom<=bounds.bottom+1});});assert.ok(allFit,'all nodes and group frames fit in minimap');
 assert.equal(await page.$$('.minimap-node').then(a=>a.length),graphNodes.length);
 await page.click('.react-flow__controls-zoomout');await page.click('.react-flow__controls-zoomout');await settle();assert.equal(await page.$eval(map,e=>e.getAttribute('viewBox')),before,'camera zoom never changes minimap bounds');
 const svg=await page.$(map),rect=await svg.boundingBox();await page.mouse.click(rect.x+rect.width/2,rect.y+rect.height*.8);await settle();assert.equal(await page.$eval(map,e=>e.getAttribute('viewBox')),before,'camera pan never changes minimap bounds');
 await page.click('.minimap-fit');await settle();await shot('full-model-minimap');
 const favicon=await page.evaluate(async()=>{const icon=document.querySelector('link[rel="icon"]');const im=new Image();im.src=icon.href;await im.decode();const canvas=document.createElement('canvas');canvas.width=32;canvas.height=32;const context=canvas.getContext('2d');context.drawImage(im,0,0,32,32);const data=context.getImageData(0,0,32,32).data;let colored=0;for(let i=0;i<data.length;i+=4)if(data[i+3]>100&&data[i]>data[i+2])colored++;return colored;});assert.ok(favicon>20,'favicon renders the original logo in orange');
 assert.equal(errors.length,0,errors.join('\n'));console.log('UI passed: consecutive grouping, nested groups, grouped groups, ungrouping, minimap full bounds independent of camera, and app favicon.');
}finally{if(browser)await browser.close();server.kill();}
