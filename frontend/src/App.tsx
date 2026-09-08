import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow, Background, Controls, MiniMap, addEdge, useEdgesState, useNodesState,
  type Connection, type NodeMouseHandler, type ReactFlowInstance
} from "@xyflow/react";
import { Activity, AlertCircle, ArrowLeft, BarChart3, Box, BrainCircuit, Check, CheckCircle2, ChevronRight, Clock3, Cpu, Database, FolderOpen, GitBranch, Info, Layers3, Moon, Palette, Play, Plus, RefreshCw, Save, Sparkles, Sun, Upload, WandSparkles, X } from "lucide-react";
import { ARCHITECTURES, BLOCK_INFO, BLOCKS, TASKS, presetsFor, templateFor } from "./catalog";
import { backend, isTauri, onTrainingEvent, pickDatasetDirectory, startTraining } from "./bridge";
import ModelNode from "./ModelNode";
import { TrainingChart } from "./Chart";
import DataPipelineEditor from "./DataPipelineEditor";
import DataCharts from "./DataCharts";
import type { Architecture, DataPipeline, DatasetAnalytics, DatasetOptions, DatasetPreview, DatasetSummary, GpuInfo, MetricPoint, ModelEdge, ModelNode as ModelNodeT, Project, ProjectModel, RunResult, Step, TaskId, TrainingRun } from "./types";

const nodeTypes = { modelNode: ModelNode };

import { BaseEdge, EdgeLabelRenderer, getSmoothStepPath, type EdgeProps } from "@xyflow/react";
import { Trash2 } from "lucide-react";

function CustomEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, selected, style, markerEnd, data }: EdgeProps) {
  const [edgePath, labelX, labelY] = getSmoothStepPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });
  return (
    <>
      <BaseEdge path={edgePath} markerEnd={markerEnd} style={style} className={selected ? "selected" : ""} />
      {selected && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: "all",
            }}
            className="nodrag nopan"
          >
            <button
              className="edge-delete-btn"
              onClick={(e) => {
                e.stopPropagation();
                if (data && typeof data.onDelete === "function") {
                  (data.onDelete as (edgeId: string) => void)(id);
                }
              }}
              title="Eliminar conexión"
            >
              <Trash2 size={11} />
            </button>
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

const edgeTypes = { customEdge: CustomEdge, smoothstep: CustomEdge };

const STEPS: Array<{ id: Step; label: string; icon: typeof BrainCircuit }> = [
  { id: "model", label: "Modelo y tarea", icon: BrainCircuit }, { id: "data", label: "Datos", icon: Database },
  { id: "builder", label: "Constructor", icon: GitBranch }, { id: "training", label: "Entrenamiento", icon: Activity }, { id: "inference", label: "Inferencia", icon: Play }
];

const uid = () => crypto.randomUUID();
const now = () => new Date().toISOString();
const newTrainingRun = (index: number): TrainingRun => ({ id: uid(), name: `Corrida ${index}`, status: "draft", config: {}, history: [], result: null, createdAt: now() });
const newProjectModel = (index: number, architecture: Architecture, task: TaskId): ProjectModel => {
  const graph = templateFor(architecture, task);
  return { id: uid(), name: `Modelo ${index}`, architecture, nodes: graph.nodes, edges: graph.edges, graphValid: false };
};
const syncOutputContract = (nodes: ModelNodeT[], edges: ModelEdge[], dataset: DatasetSummary | null): ModelNodeT[] => {
  if (!dataset) return nodes;
  const outputIds = new Set(nodes.filter(node => node.data.blockType === "output").map(node => node.id));
  const byId=new Map(nodes.map(node=>[node.id,node]));const headIds=new Set<string>();let frontier=edges.filter(edge=>outputIds.has(edge.target)).map(edge=>edge.source);const visited=new Set<string>();
  while(frontier.length){const id=frontier.shift()!;if(visited.has(id))continue;visited.add(id);const block=byId.get(id)?.data.blockType;if(["linear","conv2d","convtranspose2d"].includes(block||"")){headIds.add(id);continue}frontier.push(...edges.filter(edge=>edge.target===id).map(edge=>edge.source))}
  // La última proyección aprendible dicta el contrato, incluso si después hay
  // Reshape o Sigmoide. En lenguaje causal la dimensión de clases es la última.
  return nodes.map(node => {
    if(node.data.blockType==="embedding"&&dataset.options?.vocab_size)return {...node,data:{...node.data,properties:{...node.data.properties,vocab_size:dataset.options.vocab_size}}};
    if(node.data.blockType==="positional_encoding"&&dataset.options?.max_length)return {...node,data:{...node.data,properties:{...node.data.properties,max_length:dataset.options.max_length}}};
    if (!headIds.has(node.id)) return node;
    const key = node.data.blockType === "linear" ? "out_features" : "out_channels";const required=node.data.blockType==="linear"&&dataset.outputShape.length===2?(dataset.inputShape.length===1?dataset.outputShape.at(-1)!:dataset.outputShape.reduce((a,b)=>a*b,1)):dataset.outputShape[0];
    return { ...node, data: { ...node.data, label: `${node.data.label.split(" ·")[0]} · ${required}`, properties: { ...node.data.properties, [key]: required } } };
  });
};
const loadProjects = (): Project[] => JSON.parse(localStorage.getItem("mb-projects") || "[]");
type Theme = "dark" | "light";
type Accent = "blue" | "orange" | "green" | "violet";
const loadTheme = (): Theme => localStorage.getItem("mb-theme") === "light" ? "light" : "dark";
const loadAccent = (): Accent => {
  const value = localStorage.getItem("mb-accent");
  return value === "orange" || value === "green" || value === "violet" ? value : "blue";
};

function estimateNodeParameters(node:ModelNodeT):number {
  const props=node.data.properties || {};
  if(node.data.blockType==="linear"){const input=Number(props.in_features)||64;const output=Number(props.out_features)||64;return input*output+(props.bias===false?0:output)}
  if(node.data.blockType==="conv2d"||node.data.blockType==="convtranspose2d"){const input=Number(props.in_channels)||16;const output=Number(props.out_channels)||16;const kernel=Number(props.kernel_size)||3;return input*output*kernel*kernel+(props.bias===false?0:output)}
  if(node.data.blockType==="conv1d"){const input=Number(props.in_channels)||16;const output=Number(props.out_channels)||32;const kernel=Number(props.kernel_size)||3;return input*output*kernel+(props.bias===false?0:output)}
  if(["lstm","gru","rnn"].includes(node.data.blockType)){const gates=node.data.blockType==="lstm"?4:node.data.blockType==="gru"?3:1;const input=Number(props.input_size)||64;const hidden=Number(props.hidden_size)||64;const layers=Number(props.num_layers)||1;return gates*(input+hidden+2)*hidden*layers*(props.bidirectional?2:1)}
  if(node.data.blockType==="embedding")return (Number(props.vocab_size)||2048)*(Number(props.d_model)||64);
  if(["transformer_encoder","causal_transformer"].includes(node.data.blockType)){const d=Number(props.d_model)||64;const ff=Number(props.dim_feedforward)||4*d;const layers=Number(props.num_layers)||2;return layers*(4*d*d+2*d*ff+9*d+ff)}
  return 0;
}

function graphErrorMap(nodes:ModelNodeT[],edges:ModelEdge[]):Map<string,string>{
  const errors=new Map<string,string>();const byId=new Map(nodes.map(node=>[node.id,node]));
  const inputs=nodes.filter(node=>node.data.blockType==="input"),outputs=nodes.filter(node=>node.data.blockType==="output");
  if(inputs.length!==1) inputs.forEach(node=>errors.set(node.id,"Debe existir exactamente un bloque de entrada."));
  if(outputs.length!==1) outputs.forEach(node=>errors.set(node.id,"Debe existir exactamente un bloque de salida."));
  const seen=new Set<string>();
  edges.forEach(edge=>{const key=`${edge.source}:${edge.sourceHandle||""}->${edge.target}:${edge.targetHandle||""}`;if(seen.has(key)){errors.set(edge.source,"Conexión duplicada.");errors.set(edge.target,"Conexión duplicada.")}seen.add(key);if(!byId.has(edge.source)||!byId.has(edge.target))return;if(byId.get(edge.source)?.data.blockType==="output")errors.set(edge.source,"Una salida no puede iniciar conexiones.");if(byId.get(edge.target)?.data.blockType==="input")errors.set(edge.target,"Una entrada no puede recibir conexiones.")});
  nodes.forEach(node=>{const incoming=edges.filter(edge=>edge.target===node.id);const outgoing=edges.filter(edge=>edge.source===node.id);if(node.data.blockType!=="input"&&!incoming.length)errors.set(node.id,"El bloque no tiene entrada.");if(node.data.blockType!=="output"&&!outgoing.length)errors.set(node.id,"El bloque no tiene salida.");if(!["concat","add"].includes(node.data.blockType)&&incoming.length>1)errors.set(node.id,"Este bloque solo admite una entrada.");if(["concat","add"].includes(node.data.blockType)&&incoming.length<2)errors.set(node.id,"Este bloque necesita al menos dos entradas.")});
  const indegree=new Map(nodes.map(node=>[node.id,0]));edges.forEach(edge=>indegree.set(edge.target,(indegree.get(edge.target)||0)+1));const queue=[...indegree].filter(([,degree])=>degree===0).map(([id])=>id);let visited=0;while(queue.length){const id=queue.shift()!;visited++;edges.filter(edge=>edge.source===id).forEach(edge=>{const degree=(indegree.get(edge.target)||1)-1;indegree.set(edge.target,degree);if(degree===0)queue.push(edge.target)})}if(visited!==nodes.length)[...indegree].filter(([,degree])=>degree>0).forEach(([id])=>errors.set(id,"El bloque forma parte de un ciclo."));
  return errors;
}

export default function App() {
  const [projects, setProjects] = useState<Project[]>(loadProjects);
  const [project, setProject] = useState<Project | null>(null);
  const [step, setStep] = useState<Step>("model");
  const [task, setTask] = useState<TaskId | null>(null);
  const [architecture, setArchitecture] = useState<Architecture | null>(null);
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<ModelNodeT>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<ModelEdge>([]);
  const [models, setModels] = useState<ProjectModel[]>([]);
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const firstRun = useRef<TrainingRun>(newTrainingRun(1));
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([firstRun.current]);
  const [activeRunId, setActiveRunId] = useState(firstRun.current.id);
  const [graphValid, setGraphValid] = useState(false);
  const [training, setTraining] = useState(false);
  const runningRunId = useRef<string | null>(null);
  const [theme, setTheme] = useState<Theme>(loadTheme);
  const [accent, setAccent] = useState<Accent>(loadAccent);
  const [notice, setNotice] = useState<string>("");
  const [gpuInfo, setGpuInfo] = useState<GpuInfo | null>(null);

  useEffect(() => {
    if (isTauri()) {
      backend<GpuInfo>("system.gpu").then(setGpuInfo).catch(() => {});
    }
  }, []);


  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem("mb-theme", theme); }, [theme]);
  useEffect(() => { document.documentElement.dataset.accent = accent; localStorage.setItem("mb-accent", accent); }, [accent]);
  useEffect(()=>{if(!notice)return;const timer=window.setTimeout(()=>setNotice(""),5000);return()=>window.clearTimeout(timer)},[notice]);
  useEffect(() => { localStorage.setItem("mb-projects", JSON.stringify(projects)); }, [projects]);
  useEffect(() => {
    if (!project) return;
    const state={ step, task, architecture, dataset, nodes, edges, models, activeModelId, trainingRuns, activeRunId, graphValid };
    localStorage.setItem(`mb-state-${project.id}`, JSON.stringify(state));
    if(isTauri()) { const timer=window.setTimeout(()=>backend("project.state.save",{project,state}).catch(()=>{}),600); return ()=>window.clearTimeout(timer); }
  }, [project, step, task, architecture, dataset, nodes, edges, models, activeModelId, trainingRuns, activeRunId, graphValid]);
  useEffect(() => {
    if (!activeModelId || !architecture) return;
    setModels(items => items.map(model => model.id === activeModelId ? { ...model, architecture, nodes, edges, graphValid } : model));
  }, [activeModelId, architecture, nodes, edges, graphValid]);
  useEffect(() => { let off = () => {}; onTrainingEvent(event => {
    const targetId = runningRunId.current;
    if (!targetId) return;
    if (event.type === "metric") setTrainingRuns(runs => runs.map(item => item.id === targetId ? { ...item, history: [...item.history.filter(p => p.epoch !== Number(event.epoch)), { epoch: Number(event.epoch), trainLoss: Number(event.train_loss), valLoss: Number(event.val_loss), metric: Number(event.metric) }].sort((a,b) => a.epoch-b.epoch) } : item));
    if (event.type === "complete") { const result=event.result as unknown as RunResult; setTraining(false); setTrainingRuns(runs => runs.map(item => item.id === targetId ? { ...item, status:"completed", result, history:result.history || item.history, completedAt:now() } : item)); runningRunId.current=null; setNotice("Entrenamiento completado y checkpoint guardado."); }
    if (event.type === "error") { setTraining(false); setTrainingRuns(runs => runs.map(item => item.id === targetId ? { ...item, status:"error", error:String(event.message), completedAt:now() } : item)); runningRunId.current=null; setNotice(String(event.message)); }
  }).then(fn => off = fn); return () => off(); }, []);

  const activeTrainingRun = trainingRuns.find(item => item.id === activeRunId) || trainingRuns[0];
  const history = activeTrainingRun?.history || [];
  const run = activeTrainingRun?.result || null;
  const completed = useMemo(() => ({ model: !!task && !!architecture, data: !!dataset, builder: graphValid, training: trainingRuns.some(item=>!!item.result), inference: false }), [task, architecture, dataset, graphValid, trainingRuns]);
  const selected = nodes.find(n => n.id === selectedId) || null;

  const createProject = async (name: string, description: string) => {
    const p: Project = { id: uid(), name, description, path: `projects/${name.toLowerCase().replace(/[^a-z0-9áéíóúñ]+/gi, "-")}`, createdAt: now(), updatedAt: now() };
    if (isTauri()) { try { await backend("project.create", { project: p }); } catch (e) { setNotice(String(e)); return; } }
    const freshRun=newTrainingRun(1); setProjects(v => [p, ...v]); setProject(p); setStep("model"); setTask(null); setArchitecture(null); setDataset(null); setNodes([]); setEdges([]); setModels([]); setActiveModelId(null); setTrainingRuns([freshRun]); setActiveRunId(freshRun.id); setGraphValid(false);
  };

  const openProject = (p: Project) => {
    const stored = localStorage.getItem(`mb-state-${p.id}`);
    setProject(p);
    if (stored || p.savedState) {
      const state = stored ? JSON.parse(stored) : p.savedState!; const restoredModels=(state.models as ProjectModel[] | undefined) || []; const legacyArchitecture=state.architecture as Architecture || p.architecture || null; const legacyTask=state.task as TaskId || p.taskId || null; const fallback=legacyArchitecture&&legacyTask?[{id:uid(),name:"Modelo 1",architecture:legacyArchitecture,nodes:state.nodes as ModelNodeT[] || [],edges:state.edges as ModelEdge[] || [],graphValid:Boolean(state.graphValid)}]:[]; const loadedModels=restoredModels.length?restoredModels:fallback; const selectedModel=loadedModels.find(model=>model.id===state.activeModelId)||loadedModels[0]; setStep(state.step as Step || "model"); setTask(legacyTask); setArchitecture(selectedModel?.architecture || legacyArchitecture); setDataset(state.dataset as DatasetSummary || null); setNodes(selectedModel?.nodes || []); setEdges(selectedModel?.edges || []); setModels(loadedModels); setActiveModelId(selectedModel?.id || null); const restoredRuns=(state.trainingRuns as TrainingRun[] | undefined)?.length ? state.trainingRuns as TrainingRun[] : [{ ...newTrainingRun(1), history:state.history as MetricPoint[] || [], result:state.run as RunResult || null, status:state.run ? "completed":"draft" } as TrainingRun]; setTrainingRuns(restoredRuns); setActiveRunId((state.activeRunId as string) || restoredRuns[0].id); setGraphValid(Boolean(selectedModel?.graphValid));
    } else if (p.architecture && p.taskId) {
      setTask(p.taskId); setArchitecture(p.architecture); const template=templateFor(p.architecture,p.taskId); setNodes(template.nodes); setEdges(template.edges); const freshRun=newTrainingRun(1);setTrainingRuns([freshRun]);setActiveRunId(freshRun.id);setStep("model"); setGraphValid(false);
    }
  };

  const openPath = async () => {
    if (!isTauri()) { window.alert("Abrir una carpeta está disponible en la aplicación Tauri."); return; }
    const path = window.prompt("Ruta de la carpeta del proyecto:");
    if (!path) return;
    try { const p=await backend<Project>("project.open",{path}); setProjects(v=>[p,...v.filter(item=>item.id!==p.id)]); openProject(p); }
    catch(e) { window.alert(String(e)); }
  };

  const deleteProject = async (p: Project) => {
    if (!window.confirm(`¿Estás seguro de que deseas eliminar el proyecto "${p.name}"?`)) return;
    if (isTauri()) {
      try {
        await backend("project.delete", { project: p });
      } catch (e) {
        console.warn("Error borrando proyecto:", e);
      }
    }
    localStorage.removeItem(`mb-state-${p.id}`);
    setProjects(v => v.filter(item => item.id !== p.id));
    if (project?.id === p.id) {
      setProject(null);
    }
  };

  const choose = (a: Architecture, t: TaskId) => {
    const freshRun=newTrainingRun(1); const template = templateFor(a, t); setArchitecture(a); setTask(t); setDataset(null); setTrainingRuns([freshRun]);setActiveRunId(freshRun.id);setGraphValid(false); setNodes(template.nodes); setEdges(template.edges);
    const model=activeModelId ? { id:activeModelId, name:models.find(item=>item.id===activeModelId)?.name || "Modelo 1", architecture:a, nodes:template.nodes, edges:template.edges, graphValid:false } : newProjectModel(1,a,t);
    if (!activeModelId) { model.nodes=template.nodes; model.edges=template.edges; setActiveModelId(model.id); setModels([model]); } else setModels(items=>items.map(item=>item.id===activeModelId?model:item));
    if (project) { const updated = { ...project, architecture: a, taskId: t, updatedAt: now() }; setProject(updated); setProjects(v => v.map(p => p.id === updated.id ? updated : p)); }
  };

  const generateData = async (options?: DatasetOptions) => {
    if (!task || !architecture) return;
    try {
      let result: DatasetSummary;
      if (project) result = await backend("data.generate", { project, task_id: task, architecture, seed: 42, options });
      else throw new Error("No hay un proyecto activo.");
      setDataset(result); setNodes(current => syncOutputContract(current, edges, result)); setGraphValid(false); setNotice(`Dataset listo: ${result.samples} muestras.`);
    } catch (e) { setNotice(String(e)); }
  };

  const importData = async (path: string, options?: DatasetOptions) => {
    if (!task || !architecture || !project || !path) return;
    try { const result = await backend<DatasetSummary>("data.import", { project, task_id: task, architecture, path, options }); setDataset({...result,sourcePath:path}); setNodes(current => syncOutputContract(current, edges, result)); setGraphValid(false); setNotice("Datos importados y validados."); }
    catch (e) { setNotice(String(e)); }
  };

  const applyDataPipeline = async (pipeline: DataPipeline) => {
    if (!project || !dataset || !task) throw new Error("No hay un dataset activo.");
    try {
      const result=await backend<DatasetSummary>("data.pipeline.apply",{project,dataset,task_id:task,pipeline});
      setDataset(result);setNodes(current=>syncOutputContract(current,edges,result));setGraphValid(false);setNotice(`Pipeline aplicado · revisión ${result.pipeline?.revision ?? result.revision}.`);
    } catch(e) { setNotice(String(e)); throw e; }
  };

  const validateGraph = async () => {
    const contractNodes = syncOutputContract(nodes, edges, dataset);
    if (contractNodes !== nodes) setNodes(contractNodes);
    const graphNodes = contractNodes;
    const errors=graphErrorMap(graphNodes,edges);
    setNodes(items=>items.map(node=>({...node,data:{...node.data,error:errors.get(node.id)}})));
    if (errors.size) { const [nodeId,message]=errors.entries().next().value as [string,string];setSelectedId(nodeId);setNotice(`${nodes.find(node=>node.id===nodeId)?.data.label || "Grafo"}: ${message}`);setGraphValid(false);return false; }
    if (project && dataset && task && architecture) {
      try {
        const r = await backend<{ valid: boolean; message: string; node_shapes?: Record<string, number[]> }>("graph.validate", { project, dataset, task_id: task, architecture, graph: { nodes: graphNodes, edges } });
        if (r.node_shapes) setNodes(ns => ns.map(n => ({ ...n, data: { ...n.data, shape: `[B, ${r.node_shapes?.[n.id]?.join(", ") || "?"}]` } })));
        setGraphValid(r.valid); setNotice(r.message); return r.valid;
      } catch (e) { const message=String(e);const failed=nodes.find(node=>message.includes(node.data.label)||message.includes(node.id));if(failed){setNodes(items=>items.map(node=>node.id===failed.id?{...node,data:{...node.data,error:message}}:node));setSelectedId(failed.id)}setGraphValid(false);setNotice(message); }
    }
    setGraphValid(true); return true;
  };

  const train = async (config: Record<string, number | string>) => {
    if (!project || !dataset || !task || !architecture || !(await validateGraph())) return;
    const targetId=activeTrainingRun?.id || newTrainingRun(trainingRuns.length+1).id;
    runningRunId.current=targetId;
    setTraining(true); setTrainingRuns(runs=>runs.map(item=>item.id===targetId?{...item,modelId:activeModelId || undefined,status:"running",config,history:[],result:null,error:undefined,startedAt:now(),completedAt:undefined}:item)); setNotice("Preparando entrenamiento…");
    try {
      await startTraining({ project, dataset, task_id: task, architecture, graph: { nodes: syncOutputContract(nodes, edges, dataset), edges }, config });
    } catch (e) {
      setTraining(false); runningRunId.current=null; setTrainingRuns(runs=>runs.map(item=>item.id===targetId?{...item,status:"error",error:String(e),completedAt:now()}:item));setNotice(String(e));
    }
  };

  const infer = async (payload: { mode: string; values?: string; index?: number }) => {
    if (!project || !task || !architecture || !dataset) throw new Error("Completa proyecto y datos.");
    return backend<Record<string, unknown>>("inference.run", { project, task_id: task, architecture, dataset, checkpoint: activeTrainingRun?.result?.checkpoint, ...payload });
  };

  const addTrainingRun = () => {
    const fresh=newTrainingRun(trainingRuns.length+1);
    setTrainingRuns(items=>[...items,fresh]);setActiveRunId(fresh.id);
  };
  const switchModel = (id:string) => {
    const target=models.find(model=>model.id===id); if(!target || id===activeModelId)return;
    setModels(items=>items.map(model=>model.id===activeModelId?{...model,architecture:architecture || model.architecture,nodes,edges,graphValid}:model));
    setActiveModelId(id); setArchitecture(target.architecture); setNodes(target.nodes); setEdges(target.edges); setGraphValid(target.graphValid); setSelectedId(null);
  };
  const addModel = () => {
    if(!architecture || !task)return; const defaultName=`Modelo ${models.length+1}`; const name=window.prompt("Nombre del modelo:",defaultName)?.trim(); if(!name)return;
    const model=newProjectModel(models.length+1,architecture,task); model.name=name; setModels(items=>[...items,model]); switchModelAfterCreate(model);
  };
  const switchModelAfterCreate=(model:ProjectModel)=>{setActiveModelId(model.id);setArchitecture(model.architecture);setNodes(model.nodes);setEdges(model.edges);setGraphValid(false);setSelectedId(null)};

  const stepIndex=STEPS.findIndex(item=>item.id===step);
  const goPrevious=()=>{if(stepIndex>0)setStep(STEPS[stepIndex-1].id)};
  const goNext=async()=>{
    if(step==="model"){if(!task||!architecture){setNotice("Selecciona una tarea y arquitectura para continuar.");return}setStep("data");return}
    if(step==="data"){if(!dataset){setNotice("Carga o genera un dataset para continuar.");return}setStep("builder");return}
    if(step==="builder"){if(await validateGraph())setStep("training");return}
    if(step==="training"){if(!run){setNotice("Completa la corrida activa antes de usarla en inferencia.");return}setStep("inference")}
  };

  if (!project) return <ProjectHub projects={projects} onCreate={createProject} onOpen={openProject} onDelete={deleteProject} onOpenPath={openPath} theme={theme} setTheme={setTheme} accent={accent} setAccent={setAccent} />;

  return <div className="app-shell">
    <Sidebar step={step} setStep={setStep} completed={completed} onHome={() => setProject(null)} gpuInfo={gpuInfo} nodes={nodes} dataset={dataset} theme={theme} setTheme={setTheme} accent={accent} setAccent={setAccent} />
    <main className="main-area">
      {notice && <div className="notice"><span>{notice}</span><button onClick={() => setNotice("")}><X size={14}/></button></div>}
      <section className={`content ${step === "training" ? "no-scroll" : ""}`}>
        <div className={`stage-nav ${step==="builder"?"on-toolbar":""}`}><button className="secondary" onClick={goPrevious} disabled={stepIndex===0}><ArrowLeft/> Anterior</button>{stepIndex<STEPS.length-1&&<button className="primary" onClick={goNext}>Siguiente <ChevronRight/></button>}</div>
        {step === "model" && <ModelTask architecture={architecture} task={task} onChoose={choose} />}
        {step === "data" && <DataStep task={task} dataset={dataset} onGenerate={generateData} onImport={importData} onApplyPipeline={applyDataPipeline} onSplitsChange={splits => setDataset(current => current ? { ...current, splits } : current)} />}
        {step === "builder" && architecture && task && <Builder architecture={architecture} task={task} dataset={dataset} models={models} activeModelId={activeModelId} onSelectModel={switchModel} onAddModel={addModel} theme={theme} nodes={nodes} edges={edges} setNodes={setNodes} setEdges={setEdges} onNodesChange={(changes: Parameters<typeof onNodesChange>[0]) => { setGraphValid(false); onNodesChange(changes); }} onEdgesChange={(changes: Parameters<typeof onEdgesChange>[0]) => { setGraphValid(false); onEdgesChange(changes); }} onDirty={()=>setGraphValid(false)} selected={selected} setSelectedId={setSelectedId} onValidate={validateGraph} />}
        {step === "training" && task && <Training task={task} dataset={dataset} models={models} activeModelId={activeModelId} onSelectModel={switchModel} runs={trainingRuns} activeRunId={activeRunId} training={training} accent={accent} onSelectRun={setActiveRunId} onAddRun={addTrainingRun} onTrain={train} />}
        {step === "inference" && task && dataset && <Inference task={task} dataset={dataset} run={run} runName={activeTrainingRun?.name || "Corrida activa"} onInfer={infer} />}
      </section>
    </main>
  </div>;
}

function Builder({ architecture, task, dataset, models, activeModelId, onSelectModel, onAddModel, theme, nodes, edges, setNodes, setEdges, onNodesChange, onEdgesChange, onDirty, selected, setSelectedId, onValidate }: { architecture: Architecture; task:TaskId; dataset:DatasetSummary|null; models:ProjectModel[]; activeModelId:string|null; onSelectModel:(id:string)=>void; onAddModel:()=>void; theme: Theme; nodes: ModelNodeT[]; edges: ModelEdge[]; setNodes: React.Dispatch<React.SetStateAction<ModelNodeT[]>>; setEdges: React.Dispatch<React.SetStateAction<ModelEdge[]>>; onNodesChange: any; onEdgesChange: any; onDirty: () => void; selected: ModelNodeT | null; setSelectedId: (id: string | null) => void; onValidate: () => void }) {
  const [libraryOpen, setLibraryOpen] = useState(true);
  const [flow, setFlow] = useState<ReactFlowInstance<ModelNodeT, ModelEdge> | null>(null);
  const [hoveredBlock,setHoveredBlock]=useState<(typeof BLOCKS)[Architecture][number]|null>(null);
  const [pinnedBlock,setPinnedBlock]=useState<(typeof BLOCKS)[Architecture][number]|null>(null);
  const infoCard=useRef<HTMLDivElement>(null);
  const [presetsOpen,setPresetsOpen]=useState(false);
  const [presetName,setPresetName]=useState("");
  const [customPresets,setCustomPresets]=useState<Array<{id:string;name:string;description:string;nodes:ModelNodeT[];edges:ModelEdge[]}>>(()=>JSON.parse(localStorage.getItem(`mb-presets-${architecture}`)||"[]"));
  useEffect(()=>localStorage.setItem(`mb-presets-${architecture}`,JSON.stringify(customPresets)),[architecture,customPresets]);
  useEffect(()=>{if(!flow)return;const frame=requestAnimationFrame(()=>flow.fitView({padding:.18,duration:180}));return()=>cancelAnimationFrame(frame)},[flow,libraryOpen]);
  useEffect(()=>{
    if(!pinnedBlock)return;
    const close=(event:PointerEvent)=>{if(infoCard.current&&!infoCard.current.contains(event.target as Node))setPinnedBlock(null)};
    document.addEventListener("pointerdown",close);
    return()=>document.removeEventListener("pointerdown",close);
  },[pinnedBlock]);

  const removeNode = useCallback((id: string) => {
    onDirty();
    setNodes(ns => ns.filter(n => n.id !== id || n.data.blockType === "input" || n.data.blockType === "output"));
    setEdges(es => es.filter(e => e.source !== id && e.target !== id));
    setSelectedId(null);
  }, [setNodes, setEdges, setSelectedId, onDirty]);

  const removeEdge = useCallback((edgeId: string) => {
    onDirty();
    setEdges(es => es.filter(e => e.id !== edgeId));
  }, [setEdges, onDirty]);

  const liveErrors=useMemo(()=>graphErrorMap(nodes,edges),[nodes,edges]);
  const nodesWithHandler = useMemo(() => {
    return nodes.map(n => ({
      ...n,
      deletable: n.data.blockType !== "input" && n.data.blockType !== "output",
      data: {
        ...n.data,
        error: liveErrors.get(n.id),
        onDelete: removeNode
      }
    }));
  }, [nodes, removeNode,liveErrors]);

  const edgesWithHandler = useMemo(() => {
    return edges.map(e => ({
      ...e,
      type: "smoothstep",
      data: {
        ...e.data,
        onDelete: removeEdge
      }
    }));
  }, [edges, removeEdge]);

  const isValidConnection = useCallback((connection: Connection | ModelEdge) => {
    if (!connection.source || !connection.target) return false;
    if (connection.source === connection.target) return false;
    const sourceNode=nodes.find(node=>node.id===connection.source);const targetNode=nodes.find(node=>node.id===connection.target);
    if(!sourceNode||!targetNode||sourceNode.data.blockType==="output"||targetNode.data.blockType==="input")return false;
    if(edges.some(edge=>edge.source===connection.source&&edge.target===connection.target&&edge.sourceHandle===connection.sourceHandle&&edge.targetHandle===connection.targetHandle&&edge.id!==(connection as ModelEdge).id))return false;
    const existingIncoming = edges.filter(e => e.target === connection.target && e.id !== (connection as ModelEdge).id);
    if (!["concat","add"].includes(targetNode.data.blockType)&&existingIncoming.length > 0) return false;

    const isReachable = (src: string, tgt: string, visited = new Set<string>()): boolean => {
      if (src === tgt) return true;
      if (visited.has(tgt)) return false;
      visited.add(tgt);
      const children = edges.filter(e => e.source === tgt).map(e => e.target);
      return children.some(c => isReachable(src, c, visited));
    };

    if (isReachable(connection.source, connection.target)) return false;

    return true;
  }, [edges,nodes]);

  const connect = useCallback((p: Connection) => {
    if (!isValidConnection(p)) return;
    onDirty();
    setEdges(es => addEdge({ ...p, type: "smoothstep", data: { onDelete: removeEdge } }, es));
  }, [setEdges, onDirty, isValidConnection, removeEdge]);

  const select: NodeMouseHandler<ModelNodeT> = (_, n) => setSelectedId(n.id);
  const addBlock = (b: typeof BLOCKS[Architecture][number], position = { x: 300 + Math.random() * 300, y: 120 + Math.random() * 220 }) => {
    onDirty();
    const id = `${b.type}-${uid().slice(0, 6)}`;
    setNodes(ns => [...ns, { id, type: "modelNode", position, deletable: true, data: { blockType: b.type, label: b.label, category: b.category, properties: { ...b.defaults } } }]);
    setSelectedId(id);
  };

  const updateProp = (key: string, value: string | number | boolean) => {
    onDirty();
    return selected && setNodes(ns => ns.map(n => n.id === selected.id ? { ...n, data: { ...n.data, properties: { ...n.data.properties, [key]: value }, label: key === "out_features" || key === "out_channels" ? `${n.data.label.split(" ·")[0]} · ${value}` : n.data.label } } : n));
  };

  const groups = BLOCKS[architecture].reduce<Record<string, typeof BLOCKS[Architecture]>>((all, b) => { (all[b.category] ??= []).push(b); return all; }, {});
  const onDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const type = event.dataTransfer.getData("application/modelbuilder");
    const block = BLOCKS[architecture].find(b => b.type === type);
    if (block && flow) addBlock(block, flow.screenToFlowPosition({ x: event.clientX, y: event.clientY }));
  };

  const paramEstimate = useMemo(() => {
    return nodes.reduce((count,node)=>count+estimateNodeParameters(node),0);
  }, [nodes]);
  const builtinPresets = useMemo(() => presetsFor(architecture, task), [architecture, task]);

  const applyPreset=(preset:{nodes:ModelNodeT[];edges:ModelEdge[]})=>{onDirty();setNodes(preset.nodes.map(node=>({...node,data:{...node.data,error:undefined}})));setEdges(preset.edges);setSelectedId(null);setPresetsOpen(false);window.setTimeout(()=>flow?.fitView({padding:.18}),20)};
  const savePreset=()=>{if(!presetName.trim())return;setCustomPresets(items=>[...items,{id:uid(),name:presetName.trim(),description:"Modelo personalizado guardado desde el constructor.",nodes,edges}]);setPresetName("")};

  return (
    <div className="builder-page">
      <div className="model-tabs" role="tablist" aria-label="Modelos del proyecto">{models.map(model=><button key={model.id} role="tab" aria-selected={model.id===activeModelId} className={model.id===activeModelId?"active":""} onClick={()=>onSelectModel(model.id)}><BrainCircuit size={13}/>{model.name}<small>{ARCHITECTURES[model.architecture].name}</small></button>)}<button className="add-model" onClick={onAddModel} title="Crear nuevo modelo" aria-label="Crear nuevo modelo"><Plus size={15}/></button></div>
      <div className="builder-toolbar">
        <button className={libraryOpen ? "active" : ""} onClick={() => setLibraryOpen(!libraryOpen)}><Plus /> Bloques</button>
        <button className={presetsOpen ? "active" : ""} onClick={()=>setPresetsOpen(true)}><Layers3/> Presets</button>
        <span className="toolbar-separator" />
        <button onClick={onValidate}><Check /> Validar</button>
        <button onClick={onValidate}><Save /> Guardar revisión</button>
        <div className="toolbar-spacer" />
        <span>{nodes.length} bloques · {edges.length} conexiones</span>
      </div>

      <div className="builder-workspace">
        {libraryOpen && (
          <aside className="block-library">
            <div><span className="eyebrow">BIBLIOTECA</span><h3>{ARCHITECTURES[architecture].name}</h3></div>
            {Object.entries(groups).map(([cat, blocks]) => (
              <section key={cat}>
                <small>{cat}</small>
                {blocks.map(b => <div className="library-item" key={b.type} onMouseEnter={()=>{if(!pinnedBlock)setHoveredBlock(b)}} onMouseLeave={()=>setHoveredBlock(null)}><button draggable onClick={() => addBlock(b)} onDragStart={e => { e.dataTransfer.setData("application/modelbuilder", b.type); e.dataTransfer.effectAllowed = "move"; }}><Plus size={13} />{b.label}</button><button className={`block-info-button ${pinnedBlock?.type===b.type?"active":""}`} onClick={event=>{event.stopPropagation();setPinnedBlock(b);setHoveredBlock(null)}} title={`Fijar información sobre ${b.label}`} aria-label={`Fijar información sobre ${b.label}`}><Info size={13}/></button></div>)}
              </section>
            ))}
          </aside>
        )}

        <div className="flow-wrap">
          <div className="canvas-param-badge">
            <Cpu size={13} />
            <span>Estimación: <strong>{paramEstimate > 0 ? paramEstimate.toLocaleString() : "---"}</strong> params</span>
          </div>

          <ReactFlow
            nodes={nodesWithHandler}
            edges={edgesWithHandler}
            onInit={setFlow}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={connect}
            isValidConnection={isValidConnection}
            onNodeClick={select}
            onPaneClick={() => setSelectedId(null)}
            onDragOver={e => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; }}
            onDrop={onDrop}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
            colorMode={theme === "dark" ? "dark" : "light"}
            deleteKeyCode={["Backspace", "Delete"]}
          >
            <Background color={theme === "dark" ? "#35373b" : "#d1d2d5"} gap={22} />
            <Controls />
            <MiniMap nodeColor="var(--accent)" maskColor={theme === "dark" ? "rgba(16,17,18,.75)" : "rgba(240,240,242,.75)"} pannable zoomable />
          </ReactFlow>
        </div>

        <aside className="inspector">
          {selected ? (
            <>
              <div className="inspector-head">
                <span className="eyebrow">PROPIEDADES</span>
                <h3>{selected.data.label}</h3>
                <code>{selected.data.blockType}</code>
              </div>
              {Object.entries(selected.data.properties).length ? Object.entries(selected.data.properties).map(([k, v]) => (
                <Property key={k} name={k} value={v} disabled={!!dataset && ((selected.data.blockType==="linear"&&k==="out_features")||(selected.data.blockType==="conv2d"&&k==="out_channels")) && edges.some(edge=>edge.source===selected.id&&nodes.find(node=>node.id===edge.target)?.data.blockType==="output")} onChange={nv => updateProp(k, nv)} />
              )) : <p className="muted">Este bloque no tiene parámetros editables.</p>}
              <div className="inspector-section">
                <span className="eyebrow">TENSOR</span>
                <div className="shape-box">{selected.data.shape || "Se resolverá al validar"}</div>
              </div>
              <div className="inspector-section parameter-summary"><span className="eyebrow">PARÁMETROS</span><strong>{estimateNodeParameters(selected).toLocaleString()}</strong><small>estimados en este bloque</small><div><span>Total del modelo</span><b>{paramEstimate.toLocaleString()}</b></div></div>
              {liveErrors.get(selected.id)&&<div className="inspector-error"><AlertCircle/>{liveErrors.get(selected.id)}</div>}
            </>
          ) : (
            <div className="inspector-empty">
              <GitBranch />
              <h3>Selecciona un bloque</h3>
              <p>Edita aquí sus propiedades y revisa las dimensiones.</p>
            </div>
          )}
        </aside>
      </div>
      {(pinnedBlock||hoveredBlock)&&(()=>{const block=pinnedBlock||hoveredBlock!;return <div ref={infoCard} className={`block-info-popover ${pinnedBlock?"pinned":"preview"}`}><div className="block-info-head"><span className="eyebrow">{pinnedBlock?"INFORMACIÓN FIJADA":"VISTA RÁPIDA"}</span>{pinnedBlock&&<button className="modal-close" onClick={()=>setPinnedBlock(null)} aria-label="Cerrar información"><X/></button>}</div><h2>{block.label}</h2><p>{BLOCK_INFO[block.type]?.description || "Bloque de procesamiento compatible con esta arquitectura."}</p><div className="node-diagram">{(BLOCK_INFO[block.type]?.diagram || ["Entrada",block.label,"Salida"]).map((part,index)=><div key={`${part}-${index}`}>{index>0&&<ChevronRight/>}<span className={index===1?"operation":""}>{part}</span></div>)}</div><h3>Cómo se usa</h3><p>{BLOCK_INFO[block.type]?.usage || "Conéctalo respetando las dimensiones indicadas al validar el grafo."}</p><div className="default-properties"><span>Propiedades iniciales</span><code>{Object.keys(block.defaults).length?JSON.stringify(block.defaults):"Sin parámetros editables"}</code></div>{!pinnedBlock&&<small className="pin-hint">Haz clic en <Info/> para mantener esta guía abierta.</small>}</div>})()}
      {presetsOpen&&<div className="modal-backdrop"><div className="modal presets-modal"><button className="modal-close" onClick={()=>setPresetsOpen(false)}><X/></button><span className="eyebrow">PRESETS DE MODELO</span><h2>Empieza desde una estructura fiable</h2><div className="preset-list">{builtinPresets.map(preset=><article key={preset.id}><div><Sparkles/><span><strong>{preset.name}</strong><small>{preset.description}</small></span></div><p><b>Ventaja:</b> {preset.benefit}. <b>Desventaja:</b> {preset.tradeoff}.</p><button className="secondary" onClick={()=>applyPreset(preset)}>Aplicar preset</button></article>)}{customPresets.map(preset=><article key={preset.id}><div><Save/><span><strong>{preset.name}</strong><small>{preset.description}</small></span></div><p><b>Ventaja:</b> conserva tu diseño y configuración. <b>Desventaja:</b> depende de las dimensiones del dataset actual.</p><button className="secondary" onClick={()=>applyPreset(preset)}>Aplicar preset</button></article>)}</div><div className="save-preset"><label>Guardar el modelo actual<input value={presetName} onChange={event=>setPresetName(event.target.value)} placeholder="Nombre del preset"/></label><button className="primary" disabled={!presetName.trim()} onClick={savePreset}><Save/> Guardar</button></div></div></div>}
    </div>
  );
}

function Property({ name, value, disabled=false, onChange }: { name: string; value: string | number | boolean; disabled?:boolean; onChange: (v: string | number | boolean) => void }) {
  return (
    <label className="property">
      <span>{name.replaceAll("_", " ")}</span>
      {typeof value === "boolean" ? (
        <button className={`toggle ${value ? "on" : ""}`} onClick={() => onChange(!value)} type="button"><i /></button>
      ) : (
        <input disabled={disabled} title={disabled ? "Este valor se ajusta automáticamente a la salida del dataset." : undefined} type={typeof value === "number" ? "number" : "text"} step="any" value={String(value)} onChange={e => onChange(typeof value === "number" ? Number(e.target.value) : e.target.value)} />
      )}
    </label>
  );
}

function Training({ task, dataset, models, activeModelId, onSelectModel, runs, activeRunId, training, accent, onSelectRun, onAddRun, onTrain }: { task: TaskId; dataset:DatasetSummary|null; models:ProjectModel[]; activeModelId:string|null; onSelectModel:(id:string)=>void; runs: TrainingRun[]; activeRunId: string; training: boolean; accent: Accent; onSelectRun:(id:string)=>void; onAddRun:()=>void; onTrain: (c: Record<string, number | string>) => void }) {
  const activeRun=runs.find(item=>item.id===activeRunId) || runs[0];
  const history=activeRun?.history || [];
  const run=activeRun?.result || null;
  const [epochs, setEpochs] = useState(20);
  const [lr, setLr] = useState(0.001);
  const [batch, setBatch] = useState(Number(dataset?.options?.batchSize||32));
  const [optimizer, setOptimizer] = useState("adamw");
  const [bestModelCriterion, setBestModelCriterion] = useState("none");
  useEffect(()=>{
    const config=activeRun?.config || {};
    setEpochs(Number(config.epochs || 20));setLr(Number(config.learning_rate || .001));setBatch(Number(config.batch_size || dataset?.options?.batchSize || 32));setOptimizer(String(config.optimizer || "adamw"));setBestModelCriterion(String(config.best_model_criterion || "none"));
  },[activeRunId,dataset?.options?.batchSize]);
  const currentEpoch=history.at(-1)?.epoch || 0;
  const targetEpochs=Number(activeRun?.config.epochs || epochs || 1);
  const progress=Math.min(100,Math.round(currentEpoch/targetEpochs*100));
  const elapsedSeconds=activeRun?.startedAt ? Math.max(0,Math.round(((activeRun.completedAt ? new Date(activeRun.completedAt).getTime() : Date.now())-new Date(activeRun.startedAt).getTime())/1000)) : 0;
  const etaSeconds=currentEpoch ? Math.max(0,Math.round((elapsedSeconds/currentEpoch)*(targetEpochs-currentEpoch))) : 0;
  const formatTime=(seconds:number)=>seconds<60?`${seconds}s`:`${Math.floor(seconds/60)}m ${seconds%60}s`;
  const bestLoss=history.length ? history.reduce((best, point)=>point.valLoss<best.valLoss?point:best,history[0]) : null;
  const bestMetric=history.length ? history.reduce((best, point)=>point.metric>best.metric?point:best,history[0]) : null;

  return (
    <div className="page training-page">
      <div className="page-intro">
        <div>
          <span className="eyebrow">PASO 4 DE 5</span>
          <h2>Entrena y observa</h2>
          <p>Loss de train/validation y {TASKS[task].metric} en tiempo real.</p>
        </div>
      </div>

      <div className="run-tabs" role="tablist" aria-label="Corridas de entrenamiento">
        {runs.map(item=><button key={item.id} role="tab" aria-selected={item.id===activeRunId} className={item.id===activeRunId?"active":""} onClick={()=>onSelectRun(item.id)}><i className={`run-dot ${item.status}`}/><span>{item.name}</span>{item.status==="completed"&&<Check size={12}/>}</button>)}
        <button className="add-run" onClick={onAddRun} disabled={training} title="Crear nueva corrida"><Plus size={15}/></button>
      </div>

      <div className="training-layout">
        <div className="charts">
          <article className="panel chart-panel">
            <div className="chart-title">
              <div><span className="eyebrow">CURVA PRINCIPAL</span><h3>Loss</h3></div>
              <div className="chart-status"><span className="live"><i /> {training ? "EN VIVO" : run ? "COMPLETADO" : "LISTO"}</span>{bestLoss&&<span className="best-value">Mejor val. loss <strong>{bestLoss.valLoss.toFixed(4)}</strong> · época {bestLoss.epoch}</span>}</div>
            </div>
            <TrainingChart history={history} metric="loss" accent={accent} />
          </article>
          <article className="panel chart-panel">
            <div className="chart-title">
              <div><span className="eyebrow">MÉTRICA DE TAREA</span><h3>{TASKS[task].metric}</h3></div>
              {bestMetric&&<span className="best-value">Mejor {TASKS[task].metric} <strong>{bestMetric.metric.toFixed(4)}</strong> · época {bestMetric.epoch}</span>}
            </div>
            <TrainingChart history={history} metric={TASKS[task].metric} accent={accent} compact />
          </article>
        </div>

        <aside className="run-config">
          <div className="run-config-scroll">
          <label>Modelo a entrenar
            <select value={activeModelId || ""} onChange={event=>onSelectModel(event.target.value)} disabled={training}>
              {models.map(model=><option value={model.id} key={model.id}>{model.name} · {ARCHITECTURES[model.architecture].name}</option>)}
            </select>
          </label>
          <label>Épocas
            <input type="number" min="1" max="200" value={epochs} onChange={e => setEpochs(Number(e.target.value))} />
          </label>
          <label>Learning rate
            <input type="number" step="0.0001" value={lr} onChange={e => setLr(Number(e.target.value))} />
          </label>
          <label>Batch size
            <input type="number" min="1" value={batch} onChange={e => setBatch(Number(e.target.value))} />
          </label>
          <label>Optimizador
            <select value={optimizer} onChange={e => setOptimizer(e.target.value)}>
              <option value="adamw">AdamW</option>
              <option value="adam">Adam</option>
              <option value="sgd">SGD</option>
            </select>
          </label>          <label>Criterio de mejor modelo
            <select value={bestModelCriterion} onChange={e => setBestModelCriterion(e.target.value)}>
              <option value="none">Ninguno (Última época)</option>
              <option value="val_loss">Menor Pérdida de Validación (Val Loss)</option>
              <option value="train_loss">Menor Pérdida de Entrenamiento (Train Loss)</option>
              <option value="metric">Mejor Métrica ({TASKS[task].metric})</option>
            </select>
          </label>

          <div className="resource">
            <Cpu />
            <span><small>DISPOSITIVO</small><strong>Auto · CUDA si disponible</strong></span>
          </div>
          {(history.length > 0 || activeRun?.startedAt) && <><div className="run-progress-head"><span>Época {currentEpoch} de {targetEpochs}</span><strong>{progress}%</strong></div><div className="run-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}><i style={{width:`${progress}%`}}/></div><div className="time-summary"><span><Clock3/> Transcurrido <strong>{formatTime(elapsedSeconds)}</strong></span>{activeRun?.status==="running"&&<span>Restante aprox. <strong>{currentEpoch?formatTime(etaSeconds):"calculando…"}</strong></span>}</div></>}
          {run?.test&&<div className="test-summary"><CheckCircle2/><span><small>RESULTADO DE TEST</small><strong>{run.metricName}: {run.test.metric.toFixed(4)}</strong></span></div>}
          {activeRun?.error&&<div className="run-error"><AlertCircle/>{activeRun.error}</div>}
          </div>
          <div className="run-config-footer"><button className="primary full" disabled={training || activeRun?.status==="completed"} onClick={() => onTrain({ epochs, learning_rate: lr, batch_size: batch, optimizer, best_model_criterion: bestModelCriterion })}>{activeRun?.status==="running" ? <><Activity className="spin" /> Entrenando…</> : activeRun?.status==="completed" ? <><Check/> Corrida completada</> : <><Play /> Iniciar entrenamiento</>}</button>{activeRun?.status==="completed"&&<small>Crea una nueva corrida con + para probar otra configuración.</small>}</div>
        </aside>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function Blocked({ message }: { message: string }) { return <div className="blocked-state"><div><X size={32} /><h3>{message}</h3></div></div>; }

function AppearanceMenu({ theme, setTheme, accent, setAccent }: { theme:Theme; setTheme:(value:Theme)=>void; accent:Accent; setAccent:(value:Accent)=>void }) {
  const [open,setOpen]=useState(false);
  const root=useRef<HTMLDivElement>(null);
  useEffect(()=>{
    if(!open)return;
    const close=(event:PointerEvent)=>{if(root.current&&!root.current.contains(event.target as Node))setOpen(false)};
    const escape=(event:KeyboardEvent)=>{if(event.key==="Escape")setOpen(false)};
    document.addEventListener("pointerdown",close); document.addEventListener("keydown",escape);
    return()=>{document.removeEventListener("pointerdown",close);document.removeEventListener("keydown",escape)};
  },[open]);
  const colors:Array<{id:Accent;label:string}>=[{id:"blue",label:"Azul"},{id:"orange",label:"Naranja"},{id:"green",label:"Verde"},{id:"violet",label:"Violeta"}];
  return <div className="appearance" ref={root}>
    <button className={`icon-button ${open?"active":""}`} onClick={()=>setOpen(value=>!value)} aria-label="Apariencia" aria-haspopup="menu" aria-expanded={open}><Palette/></button>
    {open&&<div className="appearance-menu" role="menu">
      <div className="appearance-title"><Palette/><span><strong>Apariencia</strong><small>Tema y color de acento</small></span></div>
      <span className="menu-label">TEMA</span>
      <div className="theme-options"><button className={theme==="dark"?"selected":""} onClick={()=>setTheme("dark")}><Moon/> Oscuro</button><button className={theme==="light"?"selected":""} onClick={()=>setTheme("light")}><Sun/> Claro</button></div>
      <span className="menu-label">COLOR DE ACENTO</span>
      <div className="accent-options">{colors.map(color=><button key={color.id} className={accent===color.id?"selected":""} data-color={color.id} onClick={()=>setAccent(color.id)} title={color.label} aria-label={color.label} aria-pressed={accent===color.id}><i/>{accent===color.id&&<Check/>}</button>)}</div>
    </div>}
  </div>;
}

function ProjectHub({ projects, onCreate, onOpen, onDelete, onOpenPath, theme, setTheme, accent, setAccent }: { projects: Project[]; onCreate: (n:string,d:string)=>void; onOpen:(p:Project)=>void; onDelete:(p:Project)=>void; onOpenPath:()=>void; theme:Theme; setTheme:(v:Theme)=>void; accent:Accent; setAccent:(v:Accent)=>void }) {
  const [creating, setCreating] = useState(false); const [name, setName] = useState(""); const [description, setDescription] = useState("");
  return <div className="hub"><header className="hub-header"><div className="brand" title="ModelBuilder" aria-label="ModelBuilder"><span className="brand-mark"><BrainCircuit /></span></div><AppearanceMenu theme={theme} setTheme={setTheme} accent={accent} setAccent={setAccent}/></header>
    <div className="hub-content"><div className="hero"><span className="eyebrow">TALLER LOCAL DE MACHINE LEARNING</span><h1>Construye, entrena y entiende<br/><em>tu propio modelo.</em></h1><p>Del dataset al checkpoint mediante un constructor visual conectado a PyTorch.</p><button className="primary" onClick={() => setCreating(true)}><Plus size={17}/> Nuevo proyecto</button></div>
      <div className="section-head"><div><h2>Proyectos recientes</h2><p>{projects.length ? `${projects.length} proyectos en este equipo` : "Aún no hay proyectos"}</p></div><button className="secondary" onClick={onOpenPath}><FolderOpen size={16}/> Abrir carpeta</button></div>
      {projects.length ? (
        <div className="project-table">
          {projects.map(p => (
            <div className="project-row-wrap" key={p.id}>
              <button className="project-row" onClick={() => onOpen(p)}>
                <span className="project-icon"><Box/></span>
                <span><strong>{p.name}</strong><small>{p.description || "Proyecto de ModelBuilder"}</small></span>
                <span className="pill">{p.architecture?.toUpperCase() || "Sin configurar"}</span>
                <time>{new Date(p.updatedAt).toLocaleDateString()}</time>
              </button>
              <button className="project-delete-btn" onClick={(e) => { e.stopPropagation(); onDelete(p); }} title="Eliminar proyecto" aria-label="Eliminar proyecto">
                <Trash2 size={16} />
              </button>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty-state"><Sparkles/><h3>Tu primer experimento empieza aquí</h3><p>Crea un proyecto y elige qué quieres aprender de tus datos.</p></div>
      )}
    </div>
    {creating && <div className="modal-backdrop"><form className="modal" onSubmit={e => { e.preventDefault(); if(name.trim()) { onCreate(name.trim(), description.trim()); setCreating(false); } }}><button type="button" className="modal-close" onClick={() => setCreating(false)}><X/></button><span className="eyebrow">NUEVO PROYECTO</span><h2>Prepara tu espacio</h2><label>Nombre<input autoFocus value={name} onChange={e=>setName(e.target.value)} placeholder="Clasificador de señales"/></label><label>Descripción opcional<textarea value={description} onChange={e=>setDescription(e.target.value)} placeholder="Qué quieres probar…"/></label><div className="modal-actions"><button type="button" className="secondary" onClick={()=>setCreating(false)}>Cancelar</button><button className="primary" disabled={!name.trim()}>Crear proyecto <ChevronRight size={16}/></button></div></form></div>}
  </div>;
}

function computeVramEstimate(nodes: ModelNodeT[], dataset: DatasetSummary | null, totalVramBytes: number) {
  let baseMB = 350;
  let paramMemoryMB = nodes.length * 12;
  if (dataset) {
    const inputElements = (dataset.inputShape || []).reduce((a, b) => a * b, 1);
    const activationMB = (32 * inputElements * Math.max(1, nodes.length) * 4) / (1024 * 1024);
    paramMemoryMB += Math.round(activationMB);
  }
  const estimatedMB = Math.max(394, Math.round(baseMB + paramMemoryMB));
  const totalGB = (totalVramBytes / (1024 * 1024 * 1024)).toFixed(2);
  const percent = ((estimatedMB / (totalVramBytes / (1024 * 1024))) * 100).toFixed(1);
  return {
    text: `${estimatedMB} MB de ${totalGB} GB · ${percent}%`,
    percent: Math.min(100, Number(percent))
  };
}

function Sidebar({ step, setStep, completed, onHome, gpuInfo, nodes, dataset, theme, setTheme, accent, setAccent }: { step:Step; setStep:(s:Step)=>void; completed:Record<Step,boolean>; onHome:()=>void; gpuInfo: GpuInfo | null; nodes: ModelNodeT[]; dataset: DatasetSummary | null; theme:Theme; setTheme:(value:Theme)=>void; accent:Accent; setAccent:(value:Accent)=>void }) {
  const totalVram = gpuInfo?.total_vram || 12487661158;
  const gpuName = gpuInfo?.name || "NVIDIA GeForce RTX 3060";
  const vramEst = computeVramEstimate(nodes, dataset, totalVram);

  return <aside className="sidebar">
    <div className="sidebar-top">
      <button className="brand sidebar-brand" onClick={onHome} title="Volver a proyectos" aria-label="Volver a proyectos"><span className="brand-mark"><BrainCircuit/></span></button>
      <AppearanceMenu theme={theme} setTheme={setTheme} accent={accent} setAccent={setAccent}/>
    </div>
    <nav><small>FLUJO</small>{STEPS.map((s,i) => { const Icon=s.icon; return <button key={s.id} className={step===s.id?"active":""} onClick={()=>setStep(s.id)}><span className="step-index">{completed[s.id]?<Check size={13}/>:i+1}</span><Icon size={16}/><span>{s.label}</span></button>; })}</nav>
    <div className="sidebar-bottom">
      <div><Cpu size={16}/><span><strong>Motor local</strong><small>PyTorch · CPU/CUDA</small></span></div>
      <div className="vram-widget">
        <span className="vram-header">MEMORIA ESTIMADA</span>
        <span className="vram-value">{vramEst.text}</span>
        <div className="vram-track">
          <div className="vram-fill" style={{ width: `${vramEst.percent}%` }} />
        </div>
        <div className="vram-gpu-info">
          <i className="vram-gpu-dot" />
          <span>{gpuName}</span>
        </div>
      </div>
    </div>
  </aside>;
}

function ModelTask({ architecture, task, onChoose }: { architecture:Architecture|null; task:TaskId|null; onChoose:(a:Architecture,t:TaskId)=>void }) {
  const [filter,setFilter]=useState("all");
  return <div className="page model-page"><div className="page-intro"><div><span className="eyebrow">PASO 1 DE 5</span><h2>¿Qué quieres construir?</h2><p>La tarea define tus datos, la salida y cómo mediremos el aprendizaje.</p></div></div>
    <div className="filter-tabs">{["all","Tabla","Señales y series","Visión","Texto","Reconstrucción"].map(f=><button className={filter===f?"selected":""} onClick={()=>setFilter(f)} key={f}>{f==="all"?"Todas":f}</button>)}</div>
    <div className="architecture-grid">{(Object.entries(ARCHITECTURES) as [Architecture,typeof ARCHITECTURES[Architecture]][]).map(([id,a]) => {
      const visible=a.tasks.filter(t=>filter==="all"||TASKS[t].category===filter); if(!visible.length)return null;
      return <article className={`architecture-card ${architecture===id?"chosen":""}`} key={id}><div className="arch-icon"><BrainCircuit/></div><div><span className="eyebrow">ARQUITECTURA</span><h3>{a.name}</h3><p>{a.description}</p></div><div className="task-list">{visible.map(t=><button key={t} className={task===t?"selected":""} onClick={()=>onChoose(id,t)}><span><strong>{TASKS[t].name}</strong><small>{TASKS[t].input} → {TASKS[t].output}</small></span><ChevronRight size={15}/></button>)}</div></article>;
    })}</div>{task && <div className="contract-panel"><div><small>ENTRADA COMPATIBLE</small><strong>{TASKS[task].format}</strong></div><div><small>MÉTRICA PRINCIPAL</small><strong>{TASKS[task].metric}</strong></div><div><small>MODALIDAD</small><strong>{TASKS[task].modality}</strong></div></div>}
  </div>;
}

function DataStep({ task, dataset, onGenerate, onImport, onApplyPipeline, onSplitsChange }: {
  task: TaskId | null;
  dataset: DatasetSummary | null;
  onGenerate: (opts?: DatasetOptions) => Promise<void>;
  onImport: (path: string, opts?: DatasetOptions) => Promise<void>;
  onApplyPipeline: (pipeline: DataPipeline) => Promise<void>;
  onSplitsChange: (splits: DatasetSummary["splits"]) => void;
}) {
  const [path, setPath] = useState(dataset?.sourcePath || "");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [loadingLabel, setLoadingLabel] = useState("");
  const [dataView,setDataView]=useState<"preview"|"charts">("preview");
  const [analytics,setAnalytics]=useState<DatasetAnalytics|undefined>(dataset?.analytics);
  const isImage = task ? (task.startsWith("image") || task.includes("segmentation")) : false;
  const isText = task?.startsWith("text.") || false;

  const [channels, setChannels] = useState<number>(dataset?.options?.channels || (dataset?.inputShape?.[0] === 1 ? 1 : 3));
  const [resolution, setResolution] = useState<number>(dataset?.options?.resolution || (isImage && dataset?.inputShape?.[1] ? dataset.inputShape[1] : 256));
  const [normalization, setNormalization] = useState<"none" | "standard" | "minmax" | "minmax_sym">(dataset?.options?.normalization || "none");
  const [batchSize, setBatchSize] = useState<number>(dataset?.options?.batchSize || 32);
  const [maxLength,setMaxLength]=useState<number>(dataset?.options?.max_length || 64);
  const [vocabSize,setVocabSize]=useState<number>(dataset?.options?.vocab_size || 2048);

  useEffect(() => {
    if (dataset) {
      if (dataset.options?.channels) setChannels(dataset.options.channels);
      else if (dataset.inputShape?.[0]) setChannels(dataset.inputShape[0] === 1 ? 1 : 3);

      if (dataset.options?.resolution) setResolution(dataset.options.resolution);
      else if (isImage && dataset.inputShape?.[1]) setResolution(dataset.inputShape[1]);

      if (dataset.options?.normalization) setNormalization(dataset.options.normalization);
      if (dataset.options?.batchSize) setBatchSize(dataset.options.batchSize);
    }
  }, [dataset, isImage]);
  useEffect(()=>{if(!dataset||!task)return;setAnalytics(dataset.analytics);if(!dataset.analytics)backend<DatasetAnalytics>("data.analytics",{dataset,task_id:task}).then(setAnalytics).catch(()=>{})},[dataset?.id,dataset?.revision,task]);

  if (!task) return <Blocked message="Selecciona primero una tarea y arquitectura." />;

  const trainCount = dataset ? Math.round(dataset.samples * dataset.splits.train / 100) : 0;
  const validationCount = dataset ? Math.round(dataset.samples * dataset.splits.validation / 100) : 0;

  const currentOpts: DatasetOptions = {
    ...(isImage ? { channels, resolution } : isText ? {max_length:maxLength,vocab_size:vocabSize} : { normalization }),
    batchSize
  };

  const runLoading = async (label:string, operation:()=>Promise<void>) => {
    setLoading(true);setLoadingLabel(label);setProgress(6);
    const timer=window.setInterval(()=>setProgress(value=>Math.min(92,value+Math.max(1,Math.round((94-value)/8)))),180);
    try { await operation();setProgress(100); }
    finally { window.clearInterval(timer);window.setTimeout(()=>{setLoading(false);setProgress(0)},300); }
  };

  const browseFolder=async()=>{
    const selectedPath=await pickDatasetDirectory();
    if(selectedPath)setPath(selectedPath);
    else if(!isTauri()) window.alert("El selector de carpetas está disponible en la aplicación de escritorio. En navegador, escribe una ruta accesible por el servidor local.");
  };

  const handleApplyOptions = () => {
    if (dataset?.source === "imported" && (path || dataset.sourcePath)) {
      runLoading("Aplicando propiedades…",()=>onImport(path || dataset.sourcePath!, currentOpts));
    } else {
      runLoading("Regenerando datos…",()=>onGenerate(currentOpts));
    }
  };

  const reloadData=()=>dataset?.source==="imported"&&(path||dataset.sourcePath)
    ? runLoading("Volviendo a cargar el dataset…",()=>onImport(path||dataset.sourcePath!,currentOpts))
    : runLoading("Regenerando el dataset…",()=>onGenerate(currentOpts));

  return (
    <div className="page data-page">
      {loading&&<div className="loading-banner"><RefreshCw className="spin"/><span>{loadingLabel}</span><div><i style={{width:`${progress}%`}}/></div><strong>{progress}%</strong></div>}
      <div className="page-intro">
        <div>
          <span className="eyebrow">PASO 2 DE 5</span>
          <h2>Prepara los datos</h2>
        </div>
      </div>

      {!dataset ? (
        <div className="source-grid">
          <article className="source-card">
            <Upload />
            <h3>Importar datos propios</h3>
            <p>Usa el formato compatible y valida cada muestra antes de crear el modelo.</p>
            <label className="app-field">
              <span>Ruta local</span>
              <div className="path-picker"><input value={path} onChange={e => setPath(e.target.value)} placeholder="/ruta/al/dataset" /><button type="button" className="secondary" onClick={browseFolder}><FolderOpen size={15}/> Buscar</button></div>
            </label>
            <button className="secondary" disabled={!path||loading} onClick={() => runLoading("Importando y validando datos…",()=>onImport(path, currentOpts))}>
              Inspeccionar e importar
            </button>
          </article>

          <article className="source-card featured">
            <WandSparkles />
            <h3>Generar muestra sintética</h3>
            <p>Dataset reproducible diseñado para esta tarea. Seed 42 y splits 70/15/15.</p>
            <button className="primary" disabled={loading} onClick={() => runLoading("Generando datos sintéticos…",()=>onGenerate(currentOpts))}>Generar ahora</button>
          </article>
        </div>
      ) : (
        <div className="data-step-layout">
          <div className="data-main">
            <DataPipelineEditor task={task} dataset={dataset} onApply={onApplyPipeline}/>
            <article className="panel">
              <div className="data-view-header"><span className="eyebrow">EXPLORADOR DE DATOS</span><div className="data-view-switch" role="tablist" aria-label="Vista del dataset"><button role="tab" aria-selected={dataView==="preview"} className={dataView==="preview"?"active":""} onClick={()=>setDataView("preview")}>Muestras</button><button role="tab" aria-selected={dataView==="charts"} className={dataView==="charts"?"active":""} onClick={()=>setDataView("charts")}>Gráficas</button></div></div>

              {dataView==="charts"?<DataCharts analytics={analytics} task={task}/>:dataset.preview?.type === "image" && dataset.preview.items?.length ? (
                <div className="preview-image-grid">
                  {dataset.preview.items.slice(0,4).map((item, i) => (
                    <div className="preview-image-card" key={i}>
                      <div className={`dataset-image-pair ${item.targetUrl ? "with-target" : ""}`}><figure><img src={item.url} alt={item.label || `Muestra ${i + 1}`} /><figcaption>Entrada</figcaption></figure>{item.targetUrl&&<figure><img src={item.targetUrl} alt={item.targetLabel || "Objetivo"} /><figcaption>{item.targetLabel || "Salida"}</figcaption></figure>}</div>
                      <span className="preview-image-label">Muestra #{i + 1}</span>
                    </div>
                  ))}
                </div>
              ) : dataset.preview?.type === "tabular" && dataset.preview.items?.length ? (
                <div className="preview-table-wrap">
                  <table className="preview-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        {dataset.preview.columns?.map((c, i) => <th key={i}>{c}</th>)}
                        <th>Target</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dataset.preview.items.map((item, i) => (
                        <tr key={i}>
                          <td>{i + 1}</td>
                          {item.features?.map((f, j) => <td key={j}>{f}</td>)}
                          <td className="target-cell">{item.target}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="preview-art">
                  <div className="scatter">
                    {Array.from({ length: 38 }, (_, i) => (
                      <i key={i} style={{ left: `${7 + (i * 37) % 87}%`, top: `${8 + (i * 61) % 78}%`, background: ["var(--accent)", "#a78bfa", "#4ade80"][i % 3] }} />
                    ))}
                  </div>
                </div>
              )}
            </article>

            <article className="panel">
              <span className="eyebrow">PARTICIONES DE DATOS</span>
              <SplitControl splits={dataset.splits} onChange={onSplitsChange} />
              <div className="split-legend">
                <span><i className="cyan" /> Train <strong>{trainCount}</strong></span>
                <span><i className="amber" /> Validation <strong>{validationCount}</strong></span>
                <span><i className="green" /> Test <strong>{dataset.samples - trainCount - validationCount}</strong></span>
              </div>
            </article>
          </div>

          <aside className="data-sidebar"><div className="panel data-config-panel"><span className="eyebrow">PROPIEDADES DEL NODO</span><div id="data-node-inspector"/>{dataset.classes&&<div className="data-classes-section"><span className="eyebrow">CLASES ({dataset.classes.length})</span><div className="class-tags">{dataset.classes.map(c=><span key={c}>{c}</span>)}</div></div>}</div></aside>
        </div>
      )}
    </div>
  );
}

function SplitControl({ splits, onChange, disabled=false }: { splits: DatasetSummary["splits"]; onChange: (value: DatasetSummary["splits"]) => void; disabled?: boolean }) {
  const bar = useRef<HTMLDivElement>(null);
  const dragging = useRef<"train" | "validation" | null>(null);
  const minimum = 5;

  const move = useCallback((clientX: number, kind: "train" | "validation") => {
    if (!bar.current) return;
    const rect = bar.current.getBoundingClientRect();
    const position = Math.round(((clientX - rect.left) / rect.width) * 100);
    if (kind === "train") {
      const train = Math.max(minimum, Math.min(position, 100 - splits.test - minimum));
      onChange({ train, validation: 100 - train - splits.test, test: splits.test });
    } else {
      const boundary = Math.max(splits.train + minimum, Math.min(position, 100 - minimum));
      onChange({ train: splits.train, validation: boundary - splits.train, test: 100 - boundary });
    }
  }, [splits, onChange]);

  useEffect(() => {
    const pointerMove = (event: PointerEvent) => {
      if (dragging.current) {
        event.preventDefault();
        move(event.clientX, dragging.current);
      }
    };
    const pointerUp = () => { dragging.current = null; };
    window.addEventListener("pointermove", pointerMove);
    window.addEventListener("pointerup", pointerUp);
    return () => {
      window.removeEventListener("pointermove", pointerMove);
      window.removeEventListener("pointerup", pointerUp);
    };
  }, [move]);

  const keyboard = (kind: "train" | "validation", event: React.KeyboardEvent) => {
    if (disabled) return;
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const delta = event.key === "ArrowRight" ? 1 : -1;
    const width = bar.current?.getBoundingClientRect().width || 100;
    const boundary = kind === "train" ? splits.train : splits.train + splits.validation;
    move((bar.current?.getBoundingClientRect().left || 0) + (boundary + delta) * width / 100, kind);
  };

  return (
    <div className="split-control" ref={bar}>
      <div className="split-bar">
        <span style={{ width: `${splits.train}%` }}>{splits.train}%</span>
        <span style={{ width: `${splits.validation}%` }}>{splits.validation}%</span>
        <span style={{ width: `${splits.test}%` }}>{splits.test}%</span>
      </div>
      <button
        disabled={disabled}
        className="split-handle"
        style={{ left: `${splits.train}%` }}
        onPointerDown={event => { if(disabled)return;event.preventDefault(); dragging.current = "train"; }}
        onKeyDown={event => keyboard("train", event)}
        role="slider"
        aria-label="Límite entre entrenamiento y validación"
        aria-valuemin={minimum}
        aria-valuemax={100 - splits.test - minimum}
        aria-valuenow={splits.train}
      />
      <button
        disabled={disabled}
        className="split-handle"
        style={{ left: `${splits.train + splits.validation}%` }}
        onPointerDown={event => { if(disabled)return;event.preventDefault(); dragging.current = "validation"; }}
        onKeyDown={event => keyboard("validation", event)}
        role="slider"
        aria-label="Límite entre validación y prueba"
        aria-valuemin={splits.train + minimum}
        aria-valuemax={100 - minimum}
        aria-valuenow={splits.train + splits.validation}
      />
    </div>
  );
}

function Inference({ task, dataset, run, runName, onInfer }: {
  task: TaskId;
  dataset: DatasetSummary;
  run: RunResult | null;
  runName:string;
  onInfer: (payload: { mode: string; values?: string; index?: number }) => Promise<Record<string, unknown>>;
}) {
  const isImageTask = task.startsWith("image") || task.includes("segmentation");
  const isSegmentation = task.includes("segmentation");
  const isTabularTask = task.startsWith("tabular");
  const isTextTask = task.startsWith("text.");

  const [mode, setMode] = useState<"test" | "manual">("test");
  const [manualValues, setManualValues] = useState(isTabularTask ? Array.from({ length: dataset.inputShape[0] }, () => "0.5").join(", ") : "");
  const [manualImageB64, setManualImageB64] = useState<string>("");
  const [inputPreview, setInputPreview] = useState<string>("");
  const [trueLabel, setTrueLabel] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const b64 = event.target?.result as string;
      setManualImageB64(b64);
      setInputPreview(b64);
    };
    reader.readAsDataURL(file);
  };

  const execute = async () => {
    setBusy(true);
    try {
      const payload = {
        mode,
        values: mode === "manual" ? (isImageTask ? manualImageB64 : manualValues) : ""
      };
      const res = await onInfer(payload);
      setResult(res);
      if (res.inputPreview) setInputPreview(res.inputPreview as string);
      if (res.trueLabel !== undefined && res.trueLabel !== null) setTrueLabel(String(res.trueLabel));
      else if (mode === "manual") setTrueLabel(null);
    } catch (e) {
      setResult({ error: String(e) });
    } finally {
      setBusy(false);
    }
  };

  if (!run) return <Blocked message="Entrena el modelo o carga un checkpoint compatible antes de inferir." />;
  const prediction=result?.prediction;
  const comparisonAvailable=trueLabel!==null&&prediction!==undefined;
  const predictionCorrect=comparisonAvailable&&String(prediction).trim().toLowerCase()===String(trueLabel).trim().toLowerCase();
  const manualModeLabel=isImageTask ? "Cargar imagen" : isTextTask ? "Texto" : isTabularTask ? "Valores" : "Secuencia";
  const manualModeHelp=isImageTask
    ? `Una imagen se ajustará a ${dataset.inputShape.slice(1).join(" × ") || "la resolución del modelo"}.`
    : isTextTask
      ? `Escribe texto libre; se tokenizará con el vocabulario del dataset y se ajustará a ${dataset.inputShape[0]} tokens.`
      : isTabularTask
      ? `Escribe ${dataset.inputShape[0]} valores separados por comas.`
      : `Introduce una secuencia compatible con ${dataset.inputShape.join(" × ")}.`;
  const changeMode=(next:"test"|"manual")=>{setMode(next);setResult(null);setTrueLabel(null);setInputPreview(next==="manual"?manualImageB64:"")};

  return (
    <div className="page inference-page">
      <div className="page-intro">
        <div>
          <span className="eyebrow">PASO 5 DE 5</span>
          <h2>Prueba tu modelo</h2>
          <p>{TASKS[task].name} · checkpoint de {runName}</p>
        </div>
      </div>

      <article className={`panel inference-card ${isSegmentation?"segmentation-inference":""} ${comparisonAvailable?(predictionCorrect?"correct":"incorrect"):""}`}>
        <div className="inference-workbench-header">
          <div><span className="eyebrow">INFERENCIA</span><h3>Elige una entrada y prueba el modelo</h3></div>
          <span className="input-shape">Entrada: {dataset.inputShape.join(" × ")}</span>
        </div>

        <div className="inference-mode-selector" role="tablist" aria-label="Origen de los datos de inferencia">
          <button type="button" role="tab" aria-selected={mode === "test"} className={mode === "test" ? "active" : ""} onClick={() => changeMode("test")}>
            <Database size={15}/><span>Muestra eval.</span><small>Test</small>
          </button>
          <button type="button" role="tab" aria-selected={mode === "manual"} className={mode === "manual" ? "active" : ""} onClick={() => changeMode("manual")}>
            {isImageTask ? <Upload size={15}/> : <Activity size={15}/>}<span>{manualModeLabel}</span><small>Manual</small>
          </button>
        </div>

        <div className={`inference-work-area ${isImageTask && inputPreview ? "has-image-preview" : ""}`}>
          <div className="inference-middle">
          {result ? <div className="inference-results-inline">
          <div className="inference-section-heading result-heading"><div><span className="eyebrow">RESULTADO</span><h3>{result?.source ? String(result.source) : "Predicción del modelo"}</h3></div>{result&&<span className={`result-status ${comparisonAvailable?(predictionCorrect?"correct":"incorrect"):""}`}>{comparisonAvailable?(predictionCorrect?"Correcta":"Revisar"):"Calculada"}</span>}</div>
          {result.error ? <div className="inference-error-state"><AlertCircle/><span><strong>No se pudo calcular la predicción</strong><small>{String(result.error)}</small></span></div> : <>
              {comparisonAvailable&&<div className={`prediction-feedback ${predictionCorrect?"correct":"incorrect"}`}>{predictionCorrect?<CheckCircle2/>:<AlertCircle/>}<span><strong>{predictionCorrect?"Predicción correcta":"Predicción incorrecta"}</strong><small>{predictionCorrect?"Coincide con la etiqueta esperada.":`Se esperaba ${trueLabel}.`}</small></span></div>}
              <div className="prediction">
                <small>PREDICCIÓN DEL MODELO</small>
                <strong>{String(result.prediction ?? "Sin resultado")}</strong>
                {trueLabel && <span className="true-val-subtitle">Etiqueta real esperada: <strong>{trueLabel}</strong></span>}
              </div>

              {isImageTask && inputPreview && (
                result.outputPreview ? (
                  <div className="inference-visual-pair">
                    <figure><figcaption>Imagen de entrada</figcaption><img src={inputPreview} alt="Entrada usada en la inferencia" /></figure>
                    <figure><figcaption>{isSegmentation ? "Máscara generada" : "Salida del modelo"}</figcaption><img src={String(result.outputPreview)} alt="Salida generada por el modelo" /></figure>
                  </div>
                ) : (
                  <figure className="inference-input-visual">
                    <figcaption>Imagen de entrada</figcaption>
                    <img src={inputPreview} alt="Entrada usada en la inferencia" />
                  </figure>
                )
              )}

              {Array.isArray(result.probabilities) && (
                <div className="probabilities">
                  {result.probabilities.map((p, i) => (
                    <div key={i}>
                      <span>{dataset.classes?.[i] || `Clase ${i}`}</span>
                      <div>
                        <i style={{ width: `${Number(p) * 100}%` }} />
                      </div>
                      <strong>{(Number(p) * 100).toFixed(1)}%</strong>
                    </div>
                  ))}
                </div>
              )}
            </>}
        </div> : mode === "test" ? <div className="inference-preparation"><div className="source-copy"><Database/><span><strong>Conjunto reservado de evaluación</strong><small>Se elegirá una muestra aleatoria que el modelo no utilizó durante el entrenamiento.</small></span></div><div className="awaiting-result"><BarChart3/><span>La entrada y el resultado aparecerán aquí al ejecutar la inferencia.</span></div></div> : <div className="inference-preparation"><div className="source-copy"><Activity/><span><strong>{manualModeLabel}</strong><small>{manualModeHelp}</small></span></div>{isImageTask?<div className="image-upload-box"><label className="input-dropzone"><Upload size={24}/><span>Selecciona o arrastra un archivo</span><small>JPG, PNG, WebP o BMP</small><input type="file" accept="image/*" onChange={handleFileUpload}/></label>{inputPreview&&<div className="image-preview-container"><img src={inputPreview} alt="Imagen a inferir"/></div>}</div>:<label className="manual-values-field"><span>{manualModeHelp}</span><textarea value={manualValues} onChange={e=>setManualValues(e.target.value)} placeholder={isTextTask?"Escribe aquí una frase o párrafo…":"ej. 0.12, 0.45, -0.8, ..."}/></label>}</div>}
          </div>
          <button className="primary inference-run-button" onClick={execute} disabled={busy || (mode === "manual" && isImageTask && !manualImageB64)}><Play size={16}/> {busy ? "Calculando predicción…" : mode === "test" ? "Elegir muestra y calcular" : "Ejecutar inferencia"}</button>
        </div>
      </article>
    </div>
  );
}

function CircleAlertIcon(){return <div className="big-icon"><Activity/></div>}

function generateMockImageSvg(clsName: string, color: string): string {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256"><rect width="256" height="256" fill="#0c0e12"/><circle cx="128" cy="128" r="70" fill="${color}" opacity="0.85"/><text x="128" y="220" fill="#a1a2a5" font-size="16" text-anchor="middle" font-family="sans-serif">${clsName}</text></svg>`;
  return "data:image/svg+xml;base64," + btoa(svg);
}

function mockDataset(task: TaskId, options?: DatasetOptions): DatasetSummary {
  const image = task.startsWith("image") || task.includes("segmentation");
  const seq = task.startsWith("sequence");
  const segmentation = task.includes("segmentation");
  const multi = task.endsWith("multiclass");

  const resolution = options?.resolution || 256;
  const channels = options?.channels || (image ? 3 : undefined);
  const normalization = options?.normalization || "none";

  const classes = task.includes("classification") || segmentation
    ? (multi ? ["Fondo", "Círculo", "Cuadrado", "Triángulo"] : ["Clase 0", "Clase 1", "Clase 2"])
    : undefined;

  let preview: DatasetPreview | undefined;
  if (image) {
    const colors = ["#22d3ee", "#a78bfa", "#4ade80", "#fb923c"];
    const items = Array.from({ length: 8 }, (_, i) => ({
      url: generateMockImageSvg(classes ? classes[i % classes.length] : `Forma #${i + 1}`, colors[i % colors.length]),
      label: classes ? classes[i % classes.length] : `Muestra #${i + 1}`
    }));
    preview = { type: "image", items };
  } else {
    const cols = Array.from({ length: 8 }, (_, k) => `var_${k + 1}`);
    const items = Array.from({ length: 6 }, (_, i) => ({
      features: Array.from({ length: 8 }, (_, j) => Number(((Math.sin(i + j) * 1.5) + (j * 0.2)).toFixed(3))),
      target: classes ? classes[i % classes.length] : String((0.42 + i * 0.1).toFixed(2))
    }));
    preview = { type: "tabular", items, columns: cols };
  }

  const inputShape = image
    ? [channels || 3, resolution, resolution]
    : seq
      ? [48, 3]
      : [12];

  const outputShape = segmentation
    ? [task.endsWith("binary") ? 1 : 4, resolution, resolution]
    : task === "sequence.forecast"
      ? [8, 1]
      : task.includes("classification")
        ? [3]
        : [1];

  return {
    id: uid(),
    source: "synthetic",
    samples: image ? 120 : seq ? 240 : 450,
    inputShape,
    outputShape,
    classes,
    description: segmentation
      ? "Formas geométricas y máscaras alineadas"
      : image
        ? "Formas con variaciones de color, posición y ruido"
        : seq
          ? "Señales temporales con tendencia y estacionalidad"
          : "Patrones tabulares reproducibles con ruido controlado",
    splits: { train: 70, validation: 15, test: 15 },
    preview,
    options: { channels: channels || 3, resolution, normalization }
  };
}

function mockPoint(epoch:number,task:TaskId):MetricPoint { const train=1.4*Math.exp(-epoch/5)+.08+Math.sin(epoch)*.02; const val=1.5*Math.exp(-epoch/5.6)+.1+Math.sin(epoch*.7)*.025; const high=task.includes("classification")||task.includes("segmentation"); return {epoch,trainLoss:+train.toFixed(4),valLoss:+val.toFixed(4),metric:+(high?(1-val/1.8):val*.65).toFixed(4)} }
