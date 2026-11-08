import {useEffect,useRef,useState} from 'react';
import {useTranslation} from 'react-i18next';
import {Wrench,X,CheckCircle2,LoaderCircle} from 'lucide-react';
import {backend} from './bridge';
import {syncOutputContract} from './catalog';
import {translateBackendError} from './i18n/errors';
import type {Architecture,TaskId,DatasetSummary,ModelNode,ModelEdge,GraphRepairResult} from './types';

export type RepairGraph={nodes:ModelNode[];edges:ModelEdge[]};
export const repairGraphKey=(graph:RepairGraph)=>JSON.stringify({nodes:graph.nodes.map(n=>[n.id,n.data.blockType,n.data.label,n.data.properties,n.data.groupId,n.data.groupName,n.data.groups]),edges:graph.edges.map(e=>[e.id,e.source,e.target,e.sourceHandle||null,e.targetHandle||null])});
export default function GraphRepairPanel({graph,dataset,architecture,task,onApply,onClose}:{graph:RepairGraph;dataset:DatasetSummary|null;architecture:Architecture;task:TaskId;onApply:(graph:RepairGraph)=>void;onClose:()=>void}){
 const {t}=useTranslation('app');
 const [result,setResult]=useState<GraphRepairResult|null>(null),[busy,setBusy]=useState(false),[error,setError]=useState('');
 const key=repairGraphKey(graph)+JSON.stringify([dataset?.id,dataset?.inputShape,dataset?.outputShape,dataset?.revision,dataset?.path,dataset?.options,architecture,task]);
 const current=useRef(key),request=useRef(0);current.current=key;
 const [resultKey,setResultKey]=useState('');
 useEffect(()=>{request.current++;setResult(null);setError('');setBusy(false);return()=>{request.current++;}},[key]);
 const inspect=async()=>{
  const id=++request.current,checkedKey=key;setBusy(true);setError('');setResult(null);
  try{
   const reply=await backend<GraphRepairResult>('graph.solutions',{graph,dataset,architecture,task_id:task});
   // Use the same dataset contract as the editor before offering or storing a repair.
   const verified:GraphRepairResult['solutions']=[];
   for(const solution of reply.solutions){
    if(request.current!==id||current.current!==checkedKey)return;
    const normalized={...solution.graph,nodes:syncOutputContract(solution.graph.nodes,solution.graph.edges,dataset)};
    try{
     const validation=repairGraphKey(normalized)===repairGraphKey(solution.graph)?solution.validation:await backend<typeof solution.validation>('graph.validate',{graph:normalized,dataset,architecture,task_id:task});
     if(validation.valid)verified.push({...solution,graph:normalized,validation});
    }catch{/* A repair that conflicts with the editor's dataset contract is not offered. */}
   }
   reply.solutions=verified;
   if(reply.status==='repairable'&&!verified.length)reply.status='unsupported';
   if(request.current===id&&current.current===checkedKey){setResult(reply);setResultKey(checkedKey);}
  }catch(e){if(request.current===id&&current.current===checkedKey)setError(translateBackendError(String(e)));}
  finally{if(request.current===id&&current.current===checkedKey)setBusy(false);}
 };
 useEffect(()=>{if(dataset)void inspect();},[]);
 return <section className="graph-repair-panel nodrag nopan" role="dialog" aria-labelledby="graph-repair-title">
  <header><div><Wrench size={18}/><strong id="graph-repair-title">{t('repair.title')}</strong></div><button aria-label={t('repair.close')} onClick={onClose}><X size={16}/></button></header>
  <p>{t('repair.purpose')}</p>
  {!dataset?<p role="status">{t('repair.needsData')}</p>:<button className="secondary repair-inspect" disabled={busy} onClick={inspect}>{busy?<LoaderCircle size={15} className="repair-spinner"/>:<Wrench size={15}/>} {t(busy?'repair.checking':'repair.recheck')}</button>}
  {error&&<p className="repair-error" role="alert">{error}</p>}
  {result&&resultKey===key&&<div className="repair-result" role="status">
   {result.problem&&<div className="repair-problem">{translateBackendError(result.problem.message)}</div>}
   {result.status==='valid'&&<p><CheckCircle2 size={15}/> {t('repair.alreadyValid')}</p>}
   {result.status==='unsupported'&&<p>{t('repair.unsupported')}</p>}
   {result.solutions.map(solution=><article key={solution.id}><h4>{t('repair.proposal')}</h4><ol>{solution.steps.map((step,index)=><li key={index}>{t(`repair.steps.${step.key}`,step)}</li>)}</ol><p className="repair-verified"><CheckCircle2 size={15}/>{t('repair.verified')}</p><button className="primary repair-apply" onClick={()=>{if(current.current===resultKey)onApply(solution.graph);}}>{t('repair.apply')}</button></article>)}
  </div>}
 </section>;
}
