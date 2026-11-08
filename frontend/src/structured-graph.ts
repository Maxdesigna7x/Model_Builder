import type { ModelNode, ModelEdge } from './types';
import {groupPath,withGroups} from './model-groups';

export const CARD_WIDTH = 360;
export const CARD_HEIGHT = 56;

/** Stable topological rows; branches reserve independent columns until their merge. */
export function arrangeModel(nodes: ModelNode[], edges: ModelEdge[]): ModelNode[] {
  const byId = new Map(nodes.map(n => [n.id, n]));
  const children = new Map(nodes.map(n => [n.id, [] as string[]]));
  const parents = new Map(nodes.map(n => [n.id, [] as string[]]));
  for (const e of edges) if (byId.has(e.source) && byId.has(e.target)) {
    children.get(e.source)!.push(e.target); parents.get(e.target)!.push(e.source);
  }
  const degree = new Map(nodes.map(n => [n.id, parents.get(n.id)!.length]));
  const rank = new Map(nodes.map(n => [n.id, 0]));
  const queue = nodes.filter(n => !degree.get(n.id)).map(n => n.id);
  const order: string[] = [];
  while (queue.length) {
    const id = queue.shift()!; order.push(id);
    for (const child of children.get(id)!) {
      rank.set(child, Math.max(rank.get(child)!, rank.get(id)! + 1));
      degree.set(child, degree.get(child)! - 1);
      if (!degree.get(child)) queue.push(child);
    }
  }
  // Keep imported invalid graphs visible for inspection without inventing connections.
  nodes.filter(n => !order.includes(n.id)).forEach((n, i) => { order.push(n.id); rank.set(n.id, i); });
  const lanes = new Map<string, number>();
  const rows = new Map<number, string[]>();
  for (const id of order) {
    const ps = parents.get(id)!;
    const desired = ps.length ? ps.reduce((sum, p) => {
      const siblings = children.get(p)!;
      return sum + (lanes.get(p) || 0) + (siblings.indexOf(id) - (siblings.length - 1) / 2);
    }, 0) / ps.length : 0;
    const row = rows.get(rank.get(id)!) || [];
    let lane = desired;
    while (row.some(other => Math.abs(lanes.get(other)! - lane) < .98)) lane += 1;
    lanes.set(id, lane); row.push(id); rows.set(rank.get(id)!, row);
  }
  const min = Math.min(0, ...lanes.values());
  const groupBounds = new Map<string,{first:number;last:number}>();
  for(const n of nodes) for(const group of groupPath(n)) {
    const r=rank.get(n.id)||0, bounds=groupBounds.get(group.id);
    groupBounds.set(group.id,{first:Math.min(bounds?.first??r,r),last:Math.max(bounds?.last??r,r)});
  }
  const rowY=new Map<number,number>();let y=90;
  for(let r=0;r<=Math.max(0,...rank.values());r++){
    y += [...groupBounds.values()].filter(g=>g.first===r).length * 48;
    rowY.set(r,y);y+=86;
    y += [...groupBounds.values()].filter(g=>g.last===r).length * 24;
  }
  return nodes.map(n => ({ ...n, width: CARD_WIDTH, height: CARD_HEIGHT,
    position: { x: 48 + ((lanes.get(n.id) || 0) - min) * (CARD_WIDTH + 48), y: rowY.get(rank.get(n.id)||0)||90 } }));
}

export function insertConnected(nodes: ModelNode[], edges: ModelEdge[], edgeId: string, additions: ModelNode[], internal: ModelEdge[] = []) {
  const edge = edges.find(e => e.id === edgeId);
  if (!edge || !additions.length) return null;
  const targets = new Set(internal.map(e => e.target)), sources = new Set(internal.map(e => e.source));
  const entries = additions.filter(n => !targets.has(n.id));
  const exits = additions.filter(n => !sources.has(n.id));
  if (entries.length !== 1 || exits.length !== 1) return null;
  const sourcePath=groupPath(nodes.find(n=>n.id===edge.source)!),targetPath=groupPath(nodes.find(n=>n.id===edge.target)!);
  let shared=0;while(sourcePath[shared]&&sourcePath[shared].id===targetPath[shared]?.id)shared++;
  const parentPath=sourcePath.slice(0,shared);
  additions=additions.map(n=>{const own=groupPath(n);return withGroups(n,[...parentPath,...own.filter(g=>!parentPath.some(parent=>parent.id===g.id))]);});
  const nextEdges = [...edges.filter(e => e.id !== edge.id), ...internal,
    { id: `${edge.id}-in-${entries[0].id}`, source: edge.source, sourceHandle: edge.sourceHandle, target: entries[0].id, type: 'smoothstep' },
    { id: `${edge.id}-out-${exits[0].id}`, source: exits[0].id, sourceHandle: exits[0].data.blockType === "multi_scale_cnn" ? "p2" : exits[0].data.blockType === "hf_vision_backbone" && exits[0].data.properties.expose_feature_maps ? "map_0" : undefined, target: edge.target, targetHandle: edge.targetHandle, type: 'smoothstep' }];
  return { nodes: arrangeModel([...nodes.map(n => ({ ...n, selected: false })), ...additions], nextEdges), edges: nextEdges };
}

/** Only bypass a connected region with one entry and one exit. Never tear a fork apart. */
export function removeConnected(nodes: ModelNode[], edges: ModelEdge[], ids: Set<string>) {
  const removing = nodes.filter(n => ids.has(n.id));
  if (!removing.length || removing.some(n => n.data.blockType === 'input')) return null;
  if (removing.length === 1 && removing[0].data.blockType === 'output') {
    const incoming = edges.filter(e=>e.target===removing[0].id);
    if (nodes.filter(n=>n.data.blockType==='output').length<2 || incoming.length!==1 || edges.filter(e=>e.source===incoming[0].source).length<2) return null;
    const remainingEdges=edges.filter(e=>e.target!==removing[0].id);
    return {nodes:arrangeModel(nodes.filter(n=>n.id!==removing[0].id),remainingEdges),edges:remainingEdges};
  }
  if (removing.some(n => n.data.blockType === 'output')) return null;
  const incoming = edges.filter(e => !ids.has(e.source) && ids.has(e.target));
  const outgoing = edges.filter(e => ids.has(e.source) && !ids.has(e.target));
  if (incoming.length !== 1 || outgoing.length !== 1) return null;
  const reachable = new Set<string>(), pending = [incoming[0].target];
  while (pending.length) { const id = pending.pop()!; if (reachable.has(id)) continue; reachable.add(id); pending.push(...edges.filter(e => e.source === id && ids.has(e.target)).map(e => e.target)); }
  if (removing.some(n => !reachable.has(n.id))) return null;
  if (edges.some(e => !ids.has(e.source) && !ids.has(e.target) && e.source === incoming[0].source && e.target === outgoing[0].target && e.targetHandle === outgoing[0].targetHandle)) return null;
  const nextEdges = [...edges.filter(e => !ids.has(e.source) && !ids.has(e.target)), {
    id: `bypass-${incoming[0].id}-${outgoing[0].id}`, source: incoming[0].source, sourceHandle: incoming[0].sourceHandle,
    target: outgoing[0].target, targetHandle: outgoing[0].targetHandle, type: 'smoothstep' }];
  let result={nodes:nodes.filter(n=>!ids.has(n.id)),edges:nextEdges};
  // Removing the last layer of a parallel path can make both inputs identical.
  // Collapse that join instead of leaving two overlapping wires into it.
  const target=result.nodes.find(n=>n.id===outgoing[0].target);
  if(target && ['add','concat'].includes(target.data.blockType)) {
    const inputs=result.edges.filter(e=>e.target===target.id);
    const unique=inputs.filter((e,i)=>inputs.findIndex(other=>other.source===e.source&&(other.sourceHandle||'source')===(e.sourceHandle||'source'))===i);
    if(unique.length===1){const collapsed=removeMerge(result.nodes,result.edges,target.id,unique[0].id);if(collapsed)result=collapsed;}
    else result.edges=result.edges.filter(e=>e.target!==target.id||unique.includes(e));
  }
  return {nodes:arrangeModel(result.nodes,result.edges),edges:result.edges};
}

export function moveConnected(nodes: ModelNode[], edges: ModelEdge[], nodeId: string, edgeId: string) {
  const node = nodes.find(n => n.id === nodeId), target = edges.find(e => e.id === edgeId);
  if (!node || !target || target.source === nodeId || target.target === nodeId) return null;
  const membership=groupPath(node).map(g=>g.id);
  const sourceGroups=groupPath(nodes.find(n=>n.id===target.source)!).map(g=>g.id),targetGroups=groupPath(nodes.find(n=>n.id===target.target)!).map(g=>g.id);
  if(membership.some(id=>!sourceGroups.includes(id)&&!targetGroups.includes(id)))return null;
  if(sourceGroups.some(id=>targetGroups.includes(id)&&!membership.includes(id)))return null;
  // Reordering remains inside a single uninterrupted branch.
  const reachable = (from: string, to: string) => {
    const seen = new Set<string>(); let id = from;
    while (!seen.has(id)) { if (id === to) return true; seen.add(id);
      const out = edges.filter(e => e.source === id);
      if (out.length !== 1 || edges.filter(e => e.target === out[0].target).length !== 1) return false;
      id = out[0].target;
    } return false;
  };
  if (!reachable(nodeId, target.source) && !reachable(target.target, nodeId)) return null;
  const removed = removeConnected(nodes, edges, new Set([nodeId]));
  return removed ? insertConnected(removed.nodes, removed.edges, edgeId, [{ ...node, selected: true }]) : null;
}

/** Create a complete fork and join in one edit, preserving the replaced edge's ports. */
export function branchConnected(nodes: ModelNode[], edges: ModelEdge[], sourceId: string, kind: 'add'|'concat', makeId: (type:string)=>string, label: (type:string)=>string) {
  const source = nodes.find(n=>n.id===sourceId), outgoing=edges.filter(e=>e.source===sourceId);
  const shape=source?.data.shape?.match(/^\[B,\s*(\d+),\s*\d+,\s*\d+\]$/);
  if (!source || outgoing.length!==1 || source.data.blockType==='output') return null;
  const edge=outgoing[0], channels=shape?Number(shape[1]):0;
  const merge:ModelNode={id:makeId(kind),type:'modelNode',position:{x:0,y:0},selected:true,data:{blockType:kind,label:label(kind),category:'Branches',properties:kind==='concat'?{axis:1}:{}}};
  const additions:ModelNode[]=[merge];
  const nextEdges:ModelEdge[]=edges.filter(e=>e.id!==edge.id);
  const connect=(from:string,to:string,targetHandle?:string,sourceHandle?:string)=>nextEdges.push({id:makeId('edge'),source:from,target:to,targetHandle,sourceHandle,type:'smoothstep'});
  const path=(types:string[],port:string)=>{
    let previous=sourceId;
    for(const type of types){const id=makeId(type);const properties:Record<string,string|number|boolean>=type==='conv2d'?{out_channels:channels,kernel_size:3,stride:1,padding:1,bias:false}:{};
      additions.push({id,type:'modelNode',position:{x:0,y:0},data:{blockType:type,label:label(type),category:'Model',properties}});
      connect(previous,id,undefined,previous===sourceId?edge.sourceHandle||undefined:undefined);previous=id;
    }connect(previous,merge.id,port);
  };
  if(kind==='add'){connect(sourceId,merge.id,'input_0',edge.sourceHandle||undefined);path(shape?['conv2d','relu','conv2d']:['identity'],'input_1');}
  else {path(shape?['conv2d','batchnorm2d','relu']:['identity'],'input_0');path(shape?['conv2d','batchnorm2d','relu']:['identity'],'input_1');}
  connect(merge.id,edge.target,edge.targetHandle||undefined);
  return {nodes:arrangeModel([...nodes.map(n=>({...n,selected:false})),...additions],nextEdges),edges:nextEdges,selectedId:merge.id};
}

/** Keep the chosen input path, bypass the join and prune only its now-unused ancestors. */
export function removeMerge(nodes:ModelNode[],edges:ModelEdge[],mergeId:string,keepEdgeId:string){
  const merge=nodes.find(n=>n.id===mergeId),keep=edges.find(e=>e.id===keepEdgeId&&e.target===mergeId);
  if(!merge||!['add','concat'].includes(merge.data.blockType)||!keep)return null;
  const ancestors=new Set<string>(),pending=[mergeId];
  while(pending.length){const id=pending.pop()!;if(ancestors.has(id))continue;ancestors.add(id);pending.push(...edges.filter(e=>e.target===id).map(e=>e.source));}
  const replacements=edges.filter(e=>e.source===mergeId).map(e=>({...e,id:`unmerge-${e.id}`,source:keep.source,sourceHandle:keep.sourceHandle,selected:false}));
  const nextEdges=[...edges.filter(e=>e.source!==mergeId&&e.target!==mergeId),...replacements];
  const needed=new Set<string>(),queue=nodes.filter(n=>n.data.blockType==='output').map(n=>n.id);
  while(queue.length){const id=queue.pop()!;if(needed.has(id))continue;needed.add(id);queue.push(...nextEdges.filter(e=>e.target===id).map(e=>e.source));}
  const remaining=nodes.filter(n=>n.id!==mergeId&&(!ancestors.has(n.id)||needed.has(n.id)||n.data.blockType==='input'));
  const ids=new Set(remaining.map(n=>n.id));
  const remainingEdges=nextEdges.filter(e=>ids.has(e.source)&&ids.has(e.target));
  return {nodes:arrangeModel(remaining,remainingEdges),edges:remainingEdges};
}

/** The pointer crosses rows to reorder; it need not hit a connector. */
export function reorderAtPosition(nodes:ModelNode[],edges:ModelEdge[],nodeId:string,y:number){
  const layout=arrangeModel(nodes,edges),node=layout.find(n=>n.id===nodeId);
  if(!node || (y>=node.position.y-8&&y<=node.position.y+CARD_HEIGHT+8))return null;
  const byId=new Map(layout.map(n=>[n.id,n]));
  const candidates=edges.map(edge=>({edge,graph:moveConnected(nodes,edges,nodeId,edge.id)})).filter(c=>c.graph);
  candidates.sort((a,b)=>Math.abs(y-(byId.get(a.edge.source)!.position.y+CARD_HEIGHT+byId.get(a.edge.target)!.position.y)/2)-Math.abs(y-(byId.get(b.edge.source)!.position.y+CARD_HEIGHT+byId.get(b.edge.target)!.position.y)/2));
  return candidates[0]?.graph||null;
}
