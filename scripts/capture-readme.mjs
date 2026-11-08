import assert from 'node:assert/strict';
import puppeteer from 'puppeteer-core';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import net from 'node:net';

const chrome = process.env.CHROME_PATH || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser', 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find(existsSync);
if (!chrome) throw new Error('No se encontró Chrome/Chromium. Define CHROME_PATH con la ruta del ejecutable.');
const output = resolve(process.env.CAPTURE_DIR || 'capturas/readme');
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


const palette=['#fb923c','#a78bfa','#38bdf8'];
const shapes=['<circle cx="64" cy="64" r="34"/>','<rect x="30" y="30" width="68" height="68" rx="7"/>','<path d="M64 22 L108 102 H20 Z"/>'];
const artwork=i=>'data:image/svg+xml,'+encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128"><rect width="128" height="128" fill="#111418"/><g fill="${palette[i%3]}" transform="rotate(${i*6-9} 64 64)">${shapes[i%3]}</g></svg>`);
Object.assign(project,{name:'Clasificador de formas',description:'CNN de demostración · tres clases',createdAt:'2026-11-07T12:00:00Z',updatedAt:'2026-11-07T12:00:00Z'});
state.dataset={...state.dataset,samples:1200,inputShape:[3,64,64],outputShape:[3],classes:['Círculo','Cuadrado','Triángulo'],options:{channels:3,resolution:64},imageResolution:{count:1200,min:{width:64,height:64},mean:{width:64,height:64},max:{width:64,height:64},nonSquareCount:0,unavailable:0},preview:{type:'image',items:[0,1,2,3].map(i=>({url:artwork(i),label:['Círculo','Cuadrado','Triángulo'][i%3]}))}};
state.models=state.models.slice(0,1);state.models[0].name='CNN · formas';
state.nodes.forEach((n,i)=>{n.position={x:40+i*306,y:170};n.data.label=({input:'Imagen',conv1:'Conv 8 · 3×3',relu1:'ReLU',conv2:'Conv 8 · 3×3',pool:'Global AvgPool',flatten:'Flatten',head:'Clasificador',output:'Clases'})[n.id];if(n.id==='head')n.data.properties.out_features=3});state.models[0].nodes=state.nodes;
const curve=Array.from({length:40},(_,i)=>({epoch:i+1,trainLoss:1.05*Math.exp(-i/11)+.07,valLoss:1.08*Math.exp(-i/13)+.13+Math.sin(i*.7)*.015,metric:.48+.45*(1-Math.exp(-i/10)),trainMetric:.5+.47*(1-Math.exp(-i/8)),valMetric:.48+.45*(1-Math.exp(-i/10))}));
state.trainingRuns=[{id:'demo-run',name:'Entrenamiento 1',status:'completed',config:{epochs:40,batch_size:32,early_stopping_patience:5},history:curve,result:{runId:'demo-run',status:'completed',checkpoint:'demo-checkpoint',metricName:'accuracy',history:curve,test:{loss:.19,metric:.927}},createdAt:project.createdAt,modelId:'model-a'}];state.activeRunId='demo-run';
try{
 const url=`http://127.0.0.1:${port}`;
 for(let i=0;i<100;i++){try{if((await fetch(url)).ok)break}catch{}await new Promise(r=>setTimeout(r,200))}
 browser=await puppeteer.launch({executablePath:chrome,headless:true,args:['--no-sandbox','--disable-dev-shm-usage'],defaultViewport:{width:1600,height:1000,deviceScaleFactor:1}});
 const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.evaluateOnNewDocument((project,state)=>{localStorage.setItem('mb-language','es');localStorage.setItem('mb-theme','dark');localStorage.setItem('mb-accent','orange');localStorage.setItem('mb-projects',JSON.stringify([project]));localStorage.setItem(`mb-state-${project.id}`,JSON.stringify(state))},project,state);
 await page.setRequestInterception(true);page.on('request',r=>{if(r.url().includes('/api/backend')){const b=JSON.parse(r.postData()||'{}');let result=null;
 if(b.action==='project.list')result=[];if(b.action==='system.gpu')result={cuda:false,name:'CPU',total_vram:0,allocated_vram:0};if(b.action==='catalog.get')result={tasks:{},architectures:{},block_types:[]};if(b.action==='data.exists')result={exists:true};if(b.action==='graph.validate')result={valid:true,message:'Grafo válido',parameters:835,trainable_parameters:835,node_shapes:{input:[3,64,64],conv1:[8,64,64],relu1:[8,64,64],conv2:[8,64,64],pool:[8,1,1],flatten:[8],head:[3],output:[3]},backward:true};
 if(b.action==='inference.run')result={prediction:'Círculo',trueLabel:'Círculo',probabilities:[.963,.024,.013],inputPreview:artwork(0),source:'Muestra de evaluación'};
 r.respond({status:200,contentType:'application/json',body:JSON.stringify({ok:true,result})})}else if(r.url().includes('/api/training-events'))r.abort();else r.continue()});
 const shot=async name=>{await page.evaluate(()=>document.fonts.ready);await new Promise(r=>setTimeout(r,700));await page.screenshot({path:resolve(output,name+'.png')});console.log(name)};
 const stage=async i=>{await page.evaluate(i=>document.querySelectorAll('.sidebar nav button')[i].click(),i);await new Promise(r=>setTimeout(r,200))};
 await page.goto(url,{waitUntil:'networkidle2'});await page.waitForSelector('.project-row');await shot('inicio');await page.click('.project-row');await page.waitForSelector('.builder-page');
 await stage(0);await shot('arquitecturas');await stage(1);await shot('datos');await stage(2);await page.waitForSelector('.react-flow__node');await page.click('.react-flow__node[data-id="conv1"] .node-title');await shot('constructor');
 await stage(3);await shot('entrenamiento');await stage(4);await page.click('.inference-run-button');await page.waitForSelector('.classification-result-content');await shot('inferencia');
 if(errors.length)throw Error(errors.join('\n'));console.log('Orange README captures:',output);
}finally{if(browser)await browser.close();server.kill()}
