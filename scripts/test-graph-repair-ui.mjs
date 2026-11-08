import assert from 'node:assert/strict';
import puppeteer from 'puppeteer-core';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import net from 'node:net';

const chrome = process.env.CHROME_PATH || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser', 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find(existsSync);
if (!chrome) throw new Error('No se encontró Chrome/Chromium. Define CHROME_PATH con la ruta del ejecutable.');
const output = resolve(process.env.CAPTURE_DIR || 'capturas/repairs');
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

graphNodes.find(n=>n.id==='conv1').data.properties.groups=2;
for(const m of state.models)m.nodes.find(n=>n.id==='conv1').data.properties.groups=2;
const trainingStarts=[];
state.dataset.imageResolution={count:60,min:{width:60,height:71},mean:{width:402.81,height:361.82},max:{width:500,height:500},nonSquareCount:55,unavailable:0};
state.dataset.preview={type:'image',items:Array.from({length:4},(_,i)=>({url:'data:image/svg+xml,'+encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120"><rect width="120" height="120" fill="#34495e"/><circle cx="60" cy="55" r="30" fill="#fb923c"/><text x="60" y="105" text-anchor="middle" fill="white">Sample ${i+1}</text></svg>`),label:'cats'}))};
try {
 const url=`http://127.0.0.1:${port}`;
 for(let i=0;i<100;i++){try{if((await fetch(url)).ok)break}catch{}await new Promise(r=>setTimeout(r,200));}
 browser=await puppeteer.launch({executablePath:chrome,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 const page=await browser.newPage();const errors=[];page.on('pageerror',e=>{errors.push(e.message);console.log('PAGEERROR',e.message)});
 await page.evaluateOnNewDocument((project,state)=>{localStorage.setItem('mb-language','es');localStorage.setItem('mb-accent','orange');localStorage.setItem('mb-theme','dark');if(!localStorage.getItem('mb-projects'))localStorage.setItem('mb-projects',JSON.stringify([project]));if(!localStorage.getItem(`mb-state-${project.id}`))localStorage.setItem(`mb-state-${project.id}`,JSON.stringify(state));class Stream{constructor(){window.fixtureStream=this}close(){}}window.EventSource=Stream},project,state);
 await page.setRequestInterception(true);page.on('request',r=>{if(r.url().includes('/api/backend')){const b=JSON.parse(r.postData()||'{}');if(['graph.validate','graph.solutions'].includes(b.action)){r.continue();return;}let result=null;if(b.action==='project.list')result=[];if(b.action==='system.gpu')result={cuda:false,name:'CPU',total_vram:0,allocated_vram:0};if(b.action==='catalog.get')result={tasks:{},architectures:{},block_types:[]};if(b.action==='graph.validate')result={valid:true,parameters:1200,trainable_parameters:1200,node_shapes:{input:[3,16,16],conv1:[8,16,16],relu1:[8,16,16],conv2:[8,16,16],pool:[8,1,1],flatten:[8],head:[2],output:[2]},backward:true};if(b.action==='data.exists')result={exists:true};if(b.action==='data.preview')result={...state.dataset.preview,sampleIndices:[12,17,23,41]};if(b.action==='model.revision.save')result={revision:1};if(b.action==='model.catalog.guidance')result={reason:''};if(b.action==='model.catalog')result=[{id:'resnet18',name:'ResNet-18',repoId:'microsoft/resnet-18',size:'small',license:'apache-2.0',revision:'abc12345',library:'transformers',processor:'AutoImageProcessor',description:'Residual backbone',inputContract:'RGB 224×224'}];r.respond({status:200,contentType:'application/json',body:JSON.stringify({ok:true,result})})}else if(r.url().includes('/api/start-training')){trainingStarts.push(JSON.parse(r.postData()||'{}'));r.respond({status:200,contentType:'application/json',body:'{"started":true}'})}else if(r.url().includes('/api/training-events'))r.respond({status:200,contentType:'text/event-stream',body:''});else r.continue()});
 const shot=async name=>{await new Promise(r=>setTimeout(r,600));await page.screenshot({path:resolve(output,name+'.png')});const audit=await page.evaluate(()=>[...document.querySelectorAll('body *')].filter(e=>e.children.length===0&&e.textContent.trim()&&e.getBoundingClientRect().width>0&&parseFloat(getComputedStyle(e).fontSize)<12).map(e=>({text:e.textContent.slice(0,60),size:getComputedStyle(e).fontSize})));writeFileSync(resolve(output,name+'-font-audit.json'),JSON.stringify(audit,null,2));if(/vertical|horizontal/.test(name)){const ports=await page.evaluate(()=>[...document.querySelectorAll('.model-node')].flatMap(n=>{const r=n.getBoundingClientRect();return [...n.querySelectorAll('.model-node-port')].map(h=>{const b=h.getBoundingClientRect(),x=b.x+b.width/2,y=b.y+b.height/2;const edge=h.dataset.handlepos;return {edge,error:Math.abs(edge==='left'?x-r.left:edge==='right'?x-r.right:edge==='top'?y-r.top:y-r.bottom)}})}));writeFileSync(resolve(output,name+'-port-audit.json'),JSON.stringify(ports,null,2));if(ports.some(p=>p.error>1))throw Error('Port is inset: '+JSON.stringify(ports))}};
 await page.setViewport({width:1600,height:1000,deviceScaleFactor:1});await page.goto(url,{waitUntil:'networkidle2'});await shot('home');await page.click('.project-row');await page.waitForSelector('.builder-page');


 const pause=()=>new Promise(r=>setTimeout(r,900));
 const stored=()=>page.evaluate(()=>{const s=JSON.parse(localStorage.getItem('mb-state-editor-training-qa'));return s.models.find(m=>m.id===s.activeModelId)});
 await page.waitForSelector('.react-flow__node[data-id="conv1"] .node-repair',{timeout:30000});
 await page.click('.react-flow__node[data-id="conv1"] .node-repair');await page.waitForSelector('.graph-repair-panel');
 await page.waitForSelector('.repair-apply',{timeout:60000});await pause();
 assert.equal((await stored()).nodes.find(n=>n.id==='conv1').data.properties.groups,2,'inspecting never mutates the model');
 await shot('verified-solution');
 await page.click('.repair-apply');await pause();assert.equal((await stored()).nodes.find(n=>n.id==='conv1').data.properties.groups,1);
 await page.waitForFunction(()=>!document.querySelector('.react-flow__node[data-id="conv1"] .invalid'),{timeout:30000});
 const undo=await page.evaluate(()=>[...document.querySelectorAll('.builder-toolbar button')].some(b=>b.textContent.includes('Deshacer reparación')));assert.ok(undo);
 await page.evaluate(()=>[...document.querySelectorAll('.builder-toolbar button')].find(b=>b.textContent.includes('Deshacer reparación')).click());await pause();assert.equal((await stored()).nodes.find(n=>n.id==='conv1').data.properties.groups,2);
 await page.waitForSelector('.node-repair',{timeout:30000});
 await page.click('.node-repair');await page.waitForSelector('.repair-apply',{timeout:60000});
 // A proposal expires if the user changes a property while it is open.
 await page.click('.react-flow__node[data-id="conv1"]');
 const property=await page.evaluateHandle(()=>[...document.querySelectorAll('.property')].find(p=>p.querySelector('span')?.textContent.toLowerCase()==='groups')?.querySelector('input'));
 await property.asElement().click({clickCount:3});await page.keyboard.type('3');await page.keyboard.press('Tab');await pause();
 assert.equal(await page.$('.repair-apply'),null,'stale proposals cannot be applied');
 assert.equal(errors.length,0,errors.join('\n'));
 console.log('Real Python backend UI passed: red-node action, verified proposal, apply, undo, and stale-proposal invalidation.');
}finally{if(browser)await browser.close();server.kill();}
