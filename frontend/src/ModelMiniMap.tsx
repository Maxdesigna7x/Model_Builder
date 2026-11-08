import {useRef} from 'react';
import {useReactFlow,useStore,useViewport} from '@xyflow/react';
import {Maximize2} from 'lucide-react';
import {useTranslation} from 'react-i18next';
import type {ModelNode,ModelEdge} from './types';

type Rect={x:number;y:number;right:number;bottom:number;id:string};
export default function ModelMiniMap({nodes,edges,groups}:{nodes:ModelNode[];edges:ModelEdge[];groups:Rect[]}){
 const {t}=useTranslation('app');const flow=useReactFlow();const viewport=useViewport();
 const width=useStore(s=>s.width),height=useStore(s=>s.height),svg=useRef<SVGSVGElement>(null),dragging=useRef(false);
 const rects=[...nodes.map(n=>({x:n.position.x,y:n.position.y,right:n.position.x+(n.width||360),bottom:n.position.y+(n.height||56)})),...groups];
 const left=Math.min(0,...rects.map(r=>r.x)),top=Math.min(0,...rects.map(r=>r.y));
 const right=Math.max(1,...rects.map(r=>r.right)),bottom=Math.max(1,...rects.map(r=>r.bottom));
 // The bounds depend exclusively on model elements, never on camera position or zoom.
 const minX=rects.length?Math.min(...rects.map(r=>r.x)):left,minY=rects.length?Math.min(...rects.map(r=>r.y)):top;
 const padding=Math.max(16,Math.max(right-minX,bottom-minY)*.035);
 const viewBox=[minX-padding,minY-padding,right-minX+padding*2,bottom-minY+padding*2].join(' ');
 const byId=new Map(nodes.map(n=>[n.id,n]));
 const pan=(event:React.PointerEvent<SVGSVGElement>)=>{
  const matrix=svg.current?.getScreenCTM();if(!matrix)return;
  const point=new DOMPoint(event.clientX,event.clientY).matrixTransform(matrix.inverse());
  void flow.setCenter(point.x,point.y,{zoom:viewport.zoom});
 };
 return <aside className="model-minimap nodrag nopan" aria-label={t('groups.minimap')}>
  <button className="minimap-fit" onClick={()=>void flow.fitView({padding:.15,duration:200})} title={t('groups.fitAll')} aria-label={t('groups.fitAll')}><Maximize2 size={12}/></button>
  <svg ref={svg} width="160" height="110" viewBox={viewBox} role="img" aria-label={t('groups.minimap')}
   onPointerDown={e=>{if(e.button!==0)return;e.preventDefault();e.stopPropagation();dragging.current=true;e.currentTarget.setPointerCapture(e.pointerId);pan(e);}}
   onPointerMove={e=>{if(dragging.current)pan(e);}} onPointerUp={e=>{dragging.current=false;if(e.currentTarget.hasPointerCapture(e.pointerId))e.currentTarget.releasePointerCapture(e.pointerId);}} onPointerCancel={()=>{dragging.current=false;}} onWheel={e=>e.stopPropagation()}>
   {groups.map(g=><rect key={g.id} className="minimap-group" x={g.x} y={g.y} width={g.right-g.x} height={g.bottom-g.y} rx={10}/>)}
   {edges.map(e=>{const a=byId.get(e.source),b=byId.get(e.target);return a&&b?<path key={e.id} className="minimap-edge" d={`M${a.position.x+180} ${a.position.y+56} L${b.position.x+180} ${b.position.y}`}/>:null;})}
   {nodes.map(n=><rect key={n.id} data-node-id={n.id} className={`minimap-node ${n.selected?'selected':''}`} x={n.position.x} y={n.position.y} width={n.width||360} height={n.height||56} rx={7}/>)}
   <rect className="minimap-viewport" x={-viewport.x/viewport.zoom} y={-viewport.y/viewport.zoom} width={width/viewport.zoom} height={height/viewport.zoom}/>
  </svg>
 </aside>;
}
