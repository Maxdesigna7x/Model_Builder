import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {build} from 'esbuild';
const bundled=await build({entryPoints:[new URL('../frontend/src/structured-graph.ts',import.meta.url).pathname],bundle:true,write:false,format:'esm',platform:'node'});
const code=bundled.outputFiles[0].text;
const {arrangeModel,insertConnected,removeConnected,moveConnected,branchConnected}=await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const node=(id,blockType='relu')=>({id,type:'modelNode',position:{x:Math.random()*100,y:0},data:{blockType,label:id,properties:{}}});
const nodes=[node('i','input'),node('a'),node('b'),node('c'),node('o','output')];
const edges=nodes.slice(1).map((n,i)=>({id:`e${i}`,source:nodes[i].id,target:n.id}));
const inserted=insertConnected(nodes,edges,'e1',[node('new')]);
assert.equal(inserted.nodes.length,6);assert.equal(inserted.edges.length,5);
assert.ok(inserted.edges.some(e=>e.source==='a'&&e.target==='new'));
assert.ok(inserted.edges.some(e=>e.source==='new'&&e.target==='b'));
const removed=removeConnected(inserted.nodes,inserted.edges,new Set(['new']));
assert.ok(removed.edges.some(e=>e.source==='a'&&e.target==='b'));
assert.equal(removeConnected(nodes,edges,new Set(['i'])),null);
assert.equal(removeConnected(nodes,edges,new Set(['o'])),null);
assert.equal(removeConnected(nodes,edges,new Set(['a','c'])),null);
const moved=moveConnected(nodes,edges,'a','e3');
assert.ok(moved.edges.some(e=>e.source==='c'&&e.target==='a'));
assert.ok(moved.edges.some(e=>e.source==='a'&&e.target==='o'));
const branchNodes=[node('i','input'),node('a'),node('b'),node('m','concat'),node('o','output')];
const branchEdges=[{id:'ia',source:'i',target:'a'},{id:'ib',source:'i',target:'b'},{id:'am',source:'a',target:'m',targetHandle:'input_0'},{id:'bm',source:'b',target:'m',targetHandle:'input_1'},{id:'mo',source:'m',target:'o'}];
assert.equal(removeConnected(branchNodes,branchEdges,new Set(['m'])),null);
assert.equal(moveConnected(branchNodes,branchEdges,'a','bm'),null);
const intoMerge=insertConnected(branchNodes,branchEdges,'bm',[node('extra')]);
assert.equal(intoMerge.edges.find(e=>e.source==='extra').targetHandle,'input_1');
const layout=arrangeModel(branchNodes,branchEdges),byId=new Map(layout.map(n=>[n.id,n]));
assert.equal(byId.get('a').position.y,byId.get('b').position.y);
assert.ok(Math.abs(byId.get('a').position.x-byId.get('b').position.x)>360);
assert.equal(byId.get('i').position.x,byId.get('m').position.x);
assert.deepEqual(arrangeModel(layout,branchEdges),layout,'layout is stable regardless of previous coordinates');
for(const graph of [inserted,removed,moved,intoMerge])for(const n of graph.nodes){
 if(n.data.blockType!=='input')assert.ok(graph.edges.some(e=>e.target===n.id));
 if(n.data.blockType!=='output')assert.ok(graph.edges.some(e=>e.source===n.id));
}
console.log('Structured graph: insertion, port preservation, deletion, reorder and parallel layout passed.');

let serial=0;const branched=branchConnected(nodes.map(n=>({...n,data:{...n.data,shape:'[B, 16, 32, 32]'}})),edges,'a','concat',type=>type+'-'+serial++,type=>type);
assert.equal(branched.nodes.filter(n=>n.data.blockType==='conv2d').length,2);
const merge=branched.nodes.find(n=>n.data.blockType==='concat');assert.equal(branched.edges.filter(e=>e.target===merge.id).length,2);
for(const n of branched.nodes){if(n.data.blockType!=='input')assert.ok(branched.edges.some(e=>e.target===n.id));if(n.data.blockType!=='output')assert.ok(branched.edges.some(e=>e.source===n.id));}
console.log('Parallel branch creation preserves a complete connected model.');

const generic=branchConnected(nodes,edges,'a','concat',type=>type+'-'+serial++,type=>type);
assert.equal(generic.nodes.filter(n=>n.data.blockType==='identity').length,2,'generic tensor branches remain connected before configuring their layers');

const withOutput=[...nodes,node('o2','output')], outputEdges=[...edges,{id:'extra-output',source:'a',target:'o2'}];
const removedOutput=removeConnected(withOutput,outputEdges,new Set(['o2']));assert.equal(removedOutput.nodes.length,nodes.length);assert.equal(removedOutput.edges.length,edges.length);

const {removeMerge,reorderAtPosition}=await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
for(const kind of ['add','concat']){
 const branched=branchConnected(nodes.map(n=>({...n,data:{...n.data,shape:'[B, 16, 32, 32]'}})),edges,'a',kind,type=>type+'-'+serial++,type=>type);
 const join=branched.nodes.find(n=>n.data.blockType===kind);
 for(const keep of branched.edges.filter(e=>e.target===join.id)){
  const graph=removeMerge(branched.nodes,branched.edges,join.id,keep.id);
  assert.ok(graph);assert.ok(!graph.nodes.some(n=>n.id===join.id));
  const remaining=new Set(graph.nodes.map(n=>n.id));
  for(const e of graph.edges){assert.ok(remaining.has(e.source));assert.ok(remaining.has(e.target));}
  for(const n of graph.nodes){if(n.data.blockType!=='input')assert.ok(graph.edges.some(e=>e.target===n.id));if(n.data.blockType!=='output')assert.ok(graph.edges.some(e=>e.source===n.id));}
 }
}
const residual=branchConnected(nodes,edges,'a','add',type=>type+'-'+serial++,type=>type);
const identity=residual.nodes.find(n=>n.data.blockType==='identity');
const withoutResidual=removeConnected(residual.nodes,residual.edges,new Set([identity.id]));
assert.deepEqual(new Set(withoutResidual.nodes.map(n=>n.id)),new Set(nodes.map(n=>n.id)),'deleting the final residual layer also collapses the redundant merge');
assert.equal(withoutResidual.edges.length,edges.length);
const grouped=arrangeModel(nodes.map(n=>({...n,data:{...n.data,...(['a','b'].includes(n.id)?{groupId:'g'}:{})}})),edges);
const map=new Map(grouped.map(n=>[n.id,n]));
assert.ok(map.get('a').position.y-48>map.get('i').position.y+56,'group heading clears its preceding card');
assert.ok(map.get('c').position.y>map.get('b').position.y+56+24,'group bottom clears its following card');
const live=reorderAtPosition(nodes,edges,'a',arrangeModel(nodes,edges).find(n=>n.id==='c').position.y+56);
assert.ok(live.edges.some(e=>e.source==='c'&&e.target==='a'),'reorder is based on vertical row crossing, not hitting a wire');
console.log('Merge deletion, residual cleanup, live row reorder and group spacing passed.');
const sharedNodes=[...branchNodes,node('other-output','output')];
const sharedEdges=[...branchEdges,{id:'b-other',source:'b',target:'other-output'}];
const preserved=removeMerge(sharedNodes,sharedEdges,'m','am');
assert.ok(preserved.nodes.some(n=>n.id==='b'),'removing a merge never prunes a branch still used by another output');
assert.ok(preserved.edges.some(e=>e.id==='b-other'));
const groupedSequence=nodes.map(n=>({...n,data:{...n.data,...(['a','b'].includes(n.id)?{groups:[{id:'block',name:'Block'}],groupId:'block',groupName:'Block'}:{})}}));
const insertedInGroup=insertConnected(groupedSequence,edges,'e1',[node('inside')]);
assert.equal(insertedInGroup.nodes.find(n=>n.id==='inside').data.groups[0].id,'block','insertion within a group inherits its parent membership');
assert.equal(moveConnected(groupedSequence,edges,'a','e3'),null,'moving a layer across group boundaries cannot split its original group');
