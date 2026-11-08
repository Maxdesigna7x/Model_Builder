import type {ModelNode,ModelEdge} from './types';
export type ModelGroup={id:string;name:string};

/** Outer-to-inner path; old single-group projects remain readable. */
export function groupPath(node:ModelNode):ModelGroup[]{
  const groups=node.data.groups;
  if(Array.isArray(groups))return groups.filter((g):g is ModelGroup=>Boolean(g&&typeof g.id==='string'&&typeof g.name==='string'));
  return typeof node.data.groupId==='string'?[{id:node.data.groupId,name:String(node.data.groupName||'Group')}]:[];
}
export function withGroups(node:ModelNode,groups:ModelGroup[]):ModelNode{
  const leaf=groups.at(-1);
  return {...node,data:{...node.data,groups,groupId:leaf?.id,groupName:leaf?.name}};
}
export function modelGroups(nodes:ModelNode[]){
  const groups=new Map<string,{id:string;name:string;depth:number;members:ModelNode[];parent?:string}>();
  for(const node of nodes)groupPath(node).forEach((g,depth)=>{
    if(!groups.has(g.id))groups.set(g.id,{...g,depth,members:[],parent:groupPath(node)[depth-1]?.id});
    groups.get(g.id)!.members.push(node);
  });return [...groups.values()];
}
export function canGroup(nodes:ModelNode[],edges:ModelEdge[],ids:Set<string>):boolean{
  const selected=nodes.filter(n=>ids.has(n.id));if(selected.length<2)return false;
  const groups=modelGroups(nodes);
  if(groups.some(g=>g.members.length===selected.length&&g.members.every(n=>ids.has(n.id))))return false;
  // Cross a group boundary only when that entire group is selected.
  if(groups.some(g=>{const count=g.members.filter(n=>ids.has(n.id)).length;return count>0&&count<g.members.length&&count!==selected.length;}))return false;
  const incoming=new Map(selected.map(n=>[n.id,0])),outgoing=new Map(selected.map(n=>[n.id,0]));
  const neighbors=new Map(selected.map(n=>[n.id,[] as string[]]));
  for(const e of edges)if(ids.has(e.source)&&ids.has(e.target)){
    incoming.set(e.target,incoming.get(e.target)!+1);outgoing.set(e.source,outgoing.get(e.source)!+1);
    neighbors.get(e.source)!.push(e.target);neighbors.get(e.target)!.push(e.source);
  }
  const visited=new Set<string>(),pending=[selected[0].id];
  while(pending.length){const id=pending.pop()!;if(visited.has(id))continue;visited.add(id);pending.push(...neighbors.get(id)!);}
  if(visited.size!==selected.length)return false;
  const entries=selected.filter(n=>!incoming.get(n.id)),exits=selected.filter(n=>!outgoing.get(n.id));
  if(entries.length!==1||exits.length!==1)return false;
  // No skipped branch or layer may leave the region and return to it.
  if(edges.some(e=>!ids.has(e.source)&&ids.has(e.target)&&e.target!==entries[0].id))return false;
  if(edges.some(e=>ids.has(e.source)&&!ids.has(e.target)&&e.source!==exits[0].id))return false;
  return true;
}
export function createModelGroup(nodes:ModelNode[],edges:ModelEdge[],ids:Set<string>,group:ModelGroup){
  if(!canGroup(nodes,edges,ids))return null;
  const paths=nodes.filter(n=>ids.has(n.id)).map(groupPath);let prefix=0;
  while(paths[0][prefix]&&paths.every(p=>p[prefix]?.id===paths[0][prefix].id))prefix++;
  return nodes.map(n=>{if(!ids.has(n.id))return n;const p=groupPath(n);return withGroups(n,[...p.slice(0,prefix),group,...p.slice(prefix)]);});
}
export function groupRectangles(nodes:ModelNode[]){
  const groups=modelGroups(nodes).sort((a,b)=>b.depth-a.depth);
  const rects=new Map<string,{x:number;y:number;right:number;bottom:number}>();
  return groups.map(g=>{
    const children=groups.filter(child=>child.parent===g.id).map(child=>rects.get(child.id)!);
    const rect={x:Math.min(...g.members.map(n=>n.position.x),...children.map(r=>r.x))-24,
      y:Math.min(...g.members.map(n=>n.position.y),...children.map(r=>r.y))-48,
      right:Math.max(...g.members.map(n=>n.position.x+(n.width||360)),...children.map(r=>r.right))+24,
      bottom:Math.max(...g.members.map(n=>n.position.y+(n.height||56)),...children.map(r=>r.bottom))+24};
    rects.set(g.id,rect);return {...g,...rect};
  }).sort((a,b)=>a.depth-b.depth);
}
