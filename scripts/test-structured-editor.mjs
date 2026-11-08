import assert from 'node:assert/strict';
import puppeteer from 'puppeteer-core';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import net from 'node:net';

const chrome = process.env.CHROME_PATH || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser', 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find(existsSync);
if (!chrome) throw new Error('No se encontró Chrome/Chromium. Define CHROME_PATH con la ruta del ejecutable.');
const output = resolve(process.env.CAPTURE_DIR || 'capturas/redesign');
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

 const settle=()=>new Promise(r=>setTimeout(r,1000));
 const current=()=>page.evaluate(()=>JSON.parse(localStorage.getItem('mb-state-editor-training-qa')));
 const active=async()=>{const s=await current();return s.models.find(m=>m.id===s.activeModelId)};
 await settle();await shot('1600-builder');
 await page.click('.edge-insert');await page.waitForSelector('.insertion-picker');
 await page.type('.insertion-picker input','BatchNorm');await page.click('.insertion-options section>button');await settle();
 let g=await active();assert.equal(g.nodes.length,9);const added=g.nodes.find(n=>n.data.blockType==='batchnorm2d');assert.ok(added);assert.ok(g.edges.some(e=>e.target===added.id));assert.ok(g.edges.some(e=>e.source===added.id));
 await shot('1600-inserted');
 await page.keyboard.press('Delete');await settle();g=await active();assert.equal(g.nodes.length,8);assert.ok(g.edges.some(e=>e.source==='input'&&e.target==='conv1'));
 // Library choices wait for an explicit position and never create an orphan.
 await page.evaluate(()=>document.querySelector('.library-item button').click());await settle();assert.equal((await active()).nodes.length,8);
 await page.keyboard.press('Escape');
 await page.click('.react-flow__node[data-id="conv1"]');await page.keyboard.down('Control');await page.keyboard.press('d');await page.keyboard.up('Control');await settle();assert.equal((await active()).nodes.length,9);
 await page.keyboard.press('Delete');await settle();assert.equal((await active()).nodes.length,8);
 // Dragging a row uses only allowed insertion positions.
 await page.click('.react-flow__node[data-id="relu1"]');
 const grip=await page.$('.react-flow__node[data-id="relu1"] .structured-card');const gb=await grip.boundingBox();
 const dest=await page.$eval('.edge-insert[data-edge-id="conv2-pool"]',edge=>{const b=edge.getBoundingClientRect();return {x:b.x+b.width/2,y:b.y+b.height/2}});
 await page.mouse.move(gb.x+gb.width/2,gb.y+gb.height/2);await page.mouse.down();await page.mouse.move(dest.x+110,dest.y,{steps:20});await page.waitForSelector('.edge-insert[data-edge-id="conv2-pool-in-relu1"]');await page.mouse.up();await settle();
 g=await active();assert.ok(g.edges.some(e=>e.source==='conv2'&&e.target==='relu1'),'drag reorders within a connected sequence');
 await page.click('.react-flow__node[data-id="conv1"]');await settle();
 await page.evaluate(()=>document.querySelectorAll('.builder-toolbar>button')[2].click());await settle();
 g=await active();assert.ok(g.nodes.some(n=>n.data.blockType==='add'));for(const n of g.nodes){if(n.data.blockType!=='input')assert.ok(g.edges.some(e=>e.target===n.id));if(n.data.blockType!=='output')assert.ok(g.edges.some(e=>e.source===n.id));}
 await shot('1600-residual');
 const residualMerge=g.nodes.find(n=>n.data.blockType==='add');await page.click(`.react-flow__node[data-id="${residualMerge.id}"]`);await page.keyboard.press('Delete');await page.waitForSelector('.merge-removal-dialog');await page.click('.merge-removal-options button');await settle();
 g=await active();assert.equal(g.nodes.length,8);assert.ok(!g.nodes.some(n=>n.data.blockType==='add'));

 // A concat action creates two visible parallel paths, joined immediately.
 await page.click('.react-flow__node[data-id="conv2"]');await settle();await page.evaluate(()=>document.querySelectorAll('.builder-toolbar>button')[3].click());await settle();
 g=await active();assert.ok(g.nodes.some(n=>n.data.blockType==='concat'));assert.equal(g.nodes.filter(n=>n.data.blockType==='batchnorm2d').length,2);
 await page.click('.react-flow__controls-fitview');await shot('1600-branches');
 const concat=g.nodes.find(n=>n.data.blockType==='concat');await page.click(`.react-flow__node[data-id="${concat.id}"]`);await page.keyboard.press('Delete');await page.waitForSelector('.merge-removal-dialog');await shot('remove-merge-options');await page.click('.merge-removal-options button');await settle();g=await active();assert.ok(!g.nodes.some(n=>n.data.blockType==='concat'));assert.equal(g.nodes.filter(n=>n.data.blockType==='batchnorm2d').length,1);

 // Group boundaries reserve visible space above and below their cards.
 await page.click('.react-flow__node[data-id="conv1"]');await page.keyboard.down('Shift');await page.click('.react-flow__node[data-id="conv2"]');await page.keyboard.up('Shift');
 await page.click('.canvas-selection-toolbar button:nth-child(2)');await page.waitForSelector('.app-dialog input');await page.type('.app-dialog input','Grupo de prueba');await page.click('.app-dialog .primary');await settle();
 const clearGroup=await page.evaluate(()=>{const group=document.querySelector('.visual-group').getBoundingClientRect(),input=document.querySelector('.react-flow__node[data-id="input"]').getBoundingClientRect(),next=document.querySelector('.react-flow__node[data-id="relu1"]').getBoundingClientRect();return group.top>input.bottom+4&&next.top>group.bottom+4;});assert.ok(clearGroup,'group clears both neighboring cards');await shot('group-spacing');
 await page.evaluate(()=>document.querySelectorAll('.sidebar nav button')[0].click());await settle();await shot('1600-catalog');
 const before=(await current()).models.length;
 await page.click('.available-tasks button');await settle();assert.equal((await current()).models.length,before,'task preview creates no model');
 await page.setViewport({width:1280,height:800,deviceScaleFactor:1});await shot('1280-catalog');
 await page.type('.catalog-tools input','zzzz');await page.waitForSelector('.catalog-empty');
 await page.click('.catalog-tools input',{clickCount:3});await page.keyboard.press('Backspace');
 await page.evaluate(()=>document.querySelectorAll('.sidebar nav button')[2].click());await settle();await shot('1280-builder');
 // The catalog can preview every supported architecture without creating models.
 await page.evaluate(()=>document.querySelectorAll('.sidebar nav button')[0].click());
 const architectureCount=await page.$$eval('.catalog-architecture-list>button',buttons=>buttons.length);
 for(const [width,height] of [[1733,907],[1600,750],[1280,800],[1280,650]]){
 await page.setViewport({width,height,deviceScaleFactor:1});
 for(let i=0;i<architectureCount;i++){
  await page.evaluate(index=>document.querySelectorAll('.catalog-architecture-list>button')[index].click(),i);
  assert.equal(await page.$$eval('.architecture-visual text',labels=>labels.length),4);
  const fits=await page.evaluate(()=>{const panel=document.querySelector('.detail-scroll'),rect=panel.getBoundingClientRect();return panel.scrollHeight<=panel.clientHeight+1&&[...document.querySelectorAll('.available-tasks button')].every(b=>{const r=b.getBoundingClientRect();return r.top>=rect.top&&r.bottom<=rect.bottom+1});});assert.ok(fits,`all tasks visible at ${width}x${height}, architecture ${i}`);
 }
 await page.evaluate(()=>document.querySelectorAll(".catalog-architecture-list>button")[5].click());await shot(`${width}-${height}-tasks-fit`);
 }
 await page.setViewport({width:1280,height:800,deviceScaleFactor:1});

 await page.evaluate(()=>document.querySelectorAll('.catalog-architecture-list>button')[5].click());await page.click('.available-tasks button:nth-child(2)');await page.click('.architecture-detail-panel footer .primary');await page.waitForSelector('.app-dialog');await page.click('.app-dialog .secondary');assert.ok(await page.$('.model-catalog-page'));assert.equal((await current()).task,'image.classification');
 // Confirming the first choice in a fresh project must create its graph and navigate.

 await page.evaluate(()=>{const key='mb-state-editor-training-qa';const s=JSON.parse(localStorage.getItem(key));Object.assign(s,{step:'model',task:null,architecture:null,dataset:null,nodes:[],edges:[],models:[],activeModelId:null,trainingRuns:[]});localStorage.setItem(key,JSON.stringify(s));const projects=JSON.parse(localStorage.getItem('mb-projects'));projects[0].architecture=null;projects[0].taskId=null;localStorage.setItem('mb-projects',JSON.stringify(projects));});
 await page.reload({waitUntil:'networkidle2'});await page.click('.project-row');await page.waitForSelector('.model-catalog-page');
 await page.click('.available-tasks button');await page.click('.architecture-detail-panel footer .primary');await page.waitForSelector('.data-page');await settle();assert.equal((await current()).models.length,1);assert.ok((await active()).nodes.length>2);
 assert.equal(errors.length,0,errors.join('\n'));
 console.log('UI passed: live drag before release, clickable arrows, residual/concat removal, group clearance, all architecture tasks visible at 1733x907 / 1600x750 / 1280x800 / 1280x650, and model confirmation.');
}finally{if(browser)await browser.close();server.kill();}
