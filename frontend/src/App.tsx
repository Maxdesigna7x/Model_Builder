import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  ReactFlow, Background, Controls, MiniMap, addEdge, useEdgesState, useNodesState,
  type Connection, type NodeMouseHandler, type ReactFlowInstance
} from "@xyflow/react";
import { Activity, AlertCircle, ArrowLeft, ArrowUpDown, BarChart3, Binary, Box, BrainCircuit, Check, CheckCircle2, ChevronRight, Clock3, Cloud, Cpu, Database, Download, FileText, FolderOpen, GitBranch, GitFork, Grid2X2, Image, LayoutGrid, Layers3, List, Network, Info, Moon, Palette, Play, Plus, RefreshCw, Repeat2, ScanLine, Search, Shield, Shrink, SlidersHorizontal, Save, Sparkles, Split, Sun, Table2, Upload, WandSparkles, Waves, Waypoints, X, Zap } from "lucide-react";
import { ARCHITECTURES, BLOCK_INFO, BLOCKS, TASKS, presetsFor, syncOutputContract, templateFor } from "./catalog";
import { getPresetBenefit, getPresetDescription, getPresetName, getPresetTradeoff } from "./catalog-i18n";
import { backend, isTauri, onTrainingEvent, pickDatasetDirectory, startTraining } from "./bridge";
import ModelNode from "./ModelNode";
import { TrainingChart } from "./Chart";
import DataPipelineEditor from "./DataPipelineEditor";
import DataCharts from "./DataCharts";
import { translateBackendError } from "./i18n/errors";
import type { Architecture, DataPipeline, DatasetAnalytics, DatasetOptions, DatasetPreview, DatasetSummary, GpuInfo, HuggingFaceDataset, MetricPoint, ModelEdge, ModelNode as ModelNodeT, Project, ProjectModel, RunResult, Step, TaskId, TrainingRun } from "./types";

const ARCHITECTURE_ICONS: Record<Architecture, typeof BrainCircuit> = {
  mlp: Binary,
  cnn1d: Waves,
  lstm: Repeat2,
  transformer: Waypoints,
  transformer_causal: GitFork,
  cnn: ScanLine,
  vit: LayoutGrid,
  unet: Split,
  autoencoder: Shrink,
};

const nodeTypes = { modelNode: ModelNode };

import { BaseEdge, EdgeLabelRenderer, getSmoothStepPath, type EdgeProps } from "@xyflow/react";
import { Trash2 } from "lucide-react";

function CustomEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, selected, style, markerEnd, data }: EdgeProps) {
  const { t } = useTranslation(["app", "common"]);
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
              title={t("app:edge.deleteConnectionTitle")}
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

function useSteps() {
  const { t } = useTranslation("app");
  return useMemo(() => [
    { id: "model" as Step, label: t("app:step.model"), icon: BrainCircuit },
    { id: "data" as Step, label: t("app:step.data"), icon: Database },
    { id: "builder" as Step, label: t("app:step.builder"), icon: GitBranch },
    { id: "training" as Step, label: t("app:step.training"), icon: Activity },
    { id: "inference" as Step, label: t("app:step.inference"), icon: Play },
  ], [t]);
}

const uid = () => crypto.randomUUID();
const now = () => new Date().toISOString();
const formatBytes = (bytes: number) => bytes < 1024 ? `${bytes} B` : bytes < 1024 ** 2 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 ** 2).toFixed(bytes < 10 * 1024 ** 2 ? 2 : 0)} MB`;
const newTrainingRun = (index: number, name: string): TrainingRun => ({ id: uid(), name, status: "draft", config: {}, history: [], result: null, createdAt: now() });
const newProjectModel = (index: number, architecture: Architecture, task: TaskId, name: string): ProjectModel => {
  const graph = templateFor(architecture, task);
  return { id: uid(), name, architecture, nodes: graph.nodes, edges: graph.edges, graphValid: false };
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

function graphErrorMap(nodes:ModelNodeT[],edges:ModelEdge[],t:(key:string)=>string):Map<string,string>{
  const errors=new Map<string,string>();const byId=new Map(nodes.map(node=>[node.id,node]));
  const inputs=nodes.filter(node=>node.data.blockType==="input"),outputs=nodes.filter(node=>node.data.blockType==="output");
  if(inputs.length!==1) inputs.forEach(node=>errors.set(node.id,t("app:graphError.singleInput")));
  if(outputs.length!==1) outputs.forEach(node=>errors.set(node.id,t("app:graphError.singleOutput")));
  const seen=new Set<string>();
  edges.forEach(edge=>{const key=`${edge.source}:${edge.sourceHandle||""}->${edge.target}:${edge.targetHandle||""}`;if(seen.has(key)){errors.set(edge.source,t("app:graphError.duplicateConnection"));errors.set(edge.target,t("app:graphError.duplicateConnection"))}seen.add(key);if(!byId.has(edge.source)||!byId.has(edge.target))return;if(byId.get(edge.source)?.data.blockType==="output")errors.set(edge.source,t("app:graphError.outputCannotSource"));if(byId.get(edge.target)?.data.blockType==="input")errors.set(edge.target,t("app:graphError.inputCannotTarget"))});
  nodes.forEach(node=>{const incoming=edges.filter(edge=>edge.target===node.id);const outgoing=edges.filter(edge=>edge.source===node.id);if(node.data.blockType!=="input"&&!incoming.length)errors.set(node.id,t("app:graphError.blockNoInput"));if(node.data.blockType!=="output"&&!outgoing.length)errors.set(node.id,t("app:graphError.blockNoOutput"));if(!["concat","add"].includes(node.data.blockType)&&incoming.length>1)errors.set(node.id,t("app:graphError.singleInputOnly"));if(["concat","add"].includes(node.data.blockType)&&incoming.length<2)errors.set(node.id,t("app:graphError.needsTwoInputs"))});
  const indegree=new Map(nodes.map(node=>[node.id,0]));edges.forEach(edge=>indegree.set(edge.target,(indegree.get(edge.target)||0)+1));const queue=[...indegree].filter(([,degree])=>degree===0).map(([id])=>id);let visited=0;while(queue.length){const id=queue.shift()!;visited++;edges.filter(edge=>edge.source===id).forEach(edge=>{const degree=(indegree.get(edge.target)||1)-1;indegree.set(edge.target,degree);if(degree===0)queue.push(edge.target)})}if(visited!==nodes.length)[...indegree].filter(([,degree])=>degree>0).forEach(([id])=>errors.set(id,t("app:graphError.cycle")));
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
  const { t } = useTranslation(["app", "common", "catalog"]);
  const steps = useSteps();
  const firstRun = useRef<TrainingRun>(newTrainingRun(1, t("app:trainingRun.defaultName", { index: 1 })));
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
    if (event.type === "complete") { const result=event.result as unknown as RunResult; setTraining(false); setTrainingRuns(runs => runs.map(item => item.id === targetId ? { ...item, status:"completed", result, history:result.history || item.history, completedAt:now() } : item)); runningRunId.current=null; setNotice(t("app:notice.trainingComplete")); }
    if (event.type === "error") { setTraining(false); setTrainingRuns(runs => runs.map(item => item.id === targetId ? { ...item, status:"error", error:String(event.message), completedAt:now() } : item)); runningRunId.current=null; setNotice(translateBackendError(String(event.message))); }
  }).then(fn => off = fn); return () => off(); }, []);

  const activeTrainingRun = trainingRuns.find(item => item.id === activeRunId) || trainingRuns[0];
  const history = activeTrainingRun?.history || [];
  const run = activeTrainingRun?.result || null;
  const completed = useMemo(() => ({ model: !!task && !!architecture, data: !!dataset, builder: graphValid, training: trainingRuns.some(item=>!!item.result), inference: false }), [task, architecture, dataset, graphValid, trainingRuns]);
  const selected = nodes.find(n => n.id === selectedId) || null;

  const createProject = async (name: string, description: string) => {
    const p: Project = { id: uid(), name, description, path: `projects/${name.toLowerCase().replace(/[^a-z0-9áéíóúñ]+/gi, "-")}`, createdAt: now(), updatedAt: now() };
    if (isTauri()) { try { await backend("project.create", { project: p }); } catch (e) { setNotice(translateBackendError(String(e))); return; } }
    const freshRun=newTrainingRun(1, t("app:trainingRun.defaultName", { index: 1 })); setProjects(v => [p, ...v]); setProject(p); setStep("model"); setTask(null); setArchitecture(null); setDataset(null); setNodes([]); setEdges([]); setModels([]); setActiveModelId(null); setTrainingRuns([freshRun]); setActiveRunId(freshRun.id); setGraphValid(false);
  };

  const openProject = (p: Project) => {
    const stored = localStorage.getItem(`mb-state-${p.id}`);
    setProject(p);
    if (stored || p.savedState) {
      const state = stored ? JSON.parse(stored) : p.savedState!; const restoredModels=(state.models as ProjectModel[] | undefined) || []; const legacyArchitecture=state.architecture as Architecture || p.architecture || null; const legacyTask=state.task as TaskId || p.taskId || null; const fallback=legacyArchitecture&&legacyTask?[{id:uid(),name:t("app:projectModel.defaultName", { index: 1 }),architecture:legacyArchitecture,nodes:state.nodes as ModelNodeT[] || [],edges:state.edges as ModelEdge[] || [],graphValid:Boolean(state.graphValid)}]:[]; const loadedModels=restoredModels.length?restoredModels:fallback; const selectedModel=loadedModels.find(model=>model.id===state.activeModelId)||loadedModels[0]; setStep(state.step as Step || "model"); setTask(legacyTask); setArchitecture(selectedModel?.architecture || legacyArchitecture); setDataset(state.dataset as DatasetSummary || null); setNodes(selectedModel?.nodes || []); setEdges(selectedModel?.edges || []); setModels(loadedModels); setActiveModelId(selectedModel?.id || null); const restoredRuns=(state.trainingRuns as TrainingRun[] | undefined)?.length ? state.trainingRuns as TrainingRun[] : [{ ...newTrainingRun(1, t("app:trainingRun.defaultName", { index: 1 })), history:state.history as MetricPoint[] || [], result:state.run as RunResult || null, status:state.run ? "completed":"draft" } as TrainingRun]; setTrainingRuns(restoredRuns); setActiveRunId((state.activeRunId as string) || restoredRuns[0].id); setGraphValid(Boolean(selectedModel?.graphValid));
    } else if (p.architecture && p.taskId) {
      setTask(p.taskId); setArchitecture(p.architecture); const template=templateFor(p.architecture,p.taskId); setNodes(template.nodes); setEdges(template.edges); const freshRun=newTrainingRun(1, t("app:trainingRun.defaultName", { index: 1 }));setTrainingRuns([freshRun]);setActiveRunId(freshRun.id);setStep("model"); setGraphValid(false);
    }
  };

  const openPath = async () => {
    if (!isTauri()) { window.alert(t("app:openFolder.notTauri")); return; }
    const path = window.prompt(t("app:openFolder.prompt"));
    if (!path) return;
    try { const p=await backend<Project>("project.open",{path}); setProjects(v=>[p,...v.filter(item=>item.id!==p.id)]); openProject(p); }
    catch(e) { window.alert(translateBackendError(String(e))); }
  };

  const deleteProject = async (p: Project) => {
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

  const choose = (a: Architecture, taskId: TaskId) => {
    const freshRun=newTrainingRun(1, t("app:trainingRun.defaultName", { index: 1 })); const template = templateFor(a, taskId); setArchitecture(a); setTask(taskId); setDataset(null); setTrainingRuns([freshRun]);setActiveRunId(freshRun.id);setGraphValid(false); setNodes(template.nodes); setEdges(template.edges);
    const model=activeModelId ? { id:activeModelId, name:models.find(item=>item.id===activeModelId)?.name || t("app:projectModel.defaultName", { index: 1 }), architecture:a, nodes:template.nodes, edges:template.edges, graphValid:false } : newProjectModel(1,a,taskId,t("app:projectModel.defaultName", { index: 1 }));
    if (!activeModelId) { model.nodes=template.nodes; model.edges=template.edges; setActiveModelId(model.id); setModels([model]); } else setModels(items=>items.map(item=>item.id===activeModelId?model:item));
    if (project) { const updated = { ...project, architecture: a, taskId, updatedAt: now() }; setProject(updated); setProjects(v => v.map(p => p.id === updated.id ? updated : p)); }
  };

  const downloadData = async (datasetId: string, options?: DatasetOptions) => {
    if (!task || !architecture) return;
    try {
      let result: DatasetSummary;
      if (project) result = await backend("data.download", { project, task_id: task, architecture, dataset_id: datasetId, options });
      else throw new Error(t("app:data.noActiveProject"));
      setDataset(result); setNodes(current => syncOutputContract(current, edges, result)); setGraphValid(false); setNotice(t("app:data.datasetReady", { samples: result.samples }));
    } catch (e) { setNotice(translateBackendError(String(e))); }
  };

  const importData = async (path: string, options?: DatasetOptions) => {
    if (!task || !architecture || !project || !path) return;
    try { const result = await backend<DatasetSummary>("data.import", { project, task_id: task, architecture, path, options }); setDataset({...result,sourcePath:path}); setNodes(current => syncOutputContract(current, edges, result)); setGraphValid(false); setNotice(t("app:data.importedNotice")); }
    catch (e) { setNotice(translateBackendError(String(e))); }
  };

  const applyDataPipeline = async (pipeline: DataPipeline) => {
    if (!project || !dataset || !task) throw new Error(t("app:data.noActiveDataset"));
    try {
      const result=await backend<DatasetSummary>("data.pipeline.apply",{project,dataset,task_id:task,pipeline});
      setDataset(result);setNodes(current=>syncOutputContract(current,edges,result));setGraphValid(false);setNotice(t("app:data.pipelineApplied", { revision: result.pipeline?.revision ?? result.revision }));
    } catch(e) { setNotice(translateBackendError(String(e))); throw e; }
  };

  const validateGraph = async () => {
    const contractNodes = syncOutputContract(nodes, edges, dataset);
    if (contractNodes !== nodes) setNodes(contractNodes);
    const graphNodes = contractNodes;
    const errors=graphErrorMap(graphNodes,edges,t);
    setNodes(items=>items.map(node=>({...node,data:{...node.data,error:errors.get(node.id)}})));
    if (errors.size) { const [nodeId,message]=errors.entries().next().value as [string,string];setSelectedId(nodeId);setNotice(`${nodes.find(node=>node.id===nodeId)?.data.label || t("app:builder.graph")}: ${message}`);setGraphValid(false);return false; }
    if (project && dataset && task && architecture) {
      try {
        const r = await backend<{ valid: boolean; message: string; node_shapes?: Record<string, number[]> }>("graph.validate", { project, dataset, task_id: task, architecture, graph: { nodes: graphNodes, edges } });
        if (r.node_shapes) setNodes(ns => ns.map(n => ({ ...n, data: { ...n.data, shape: `[B, ${r.node_shapes?.[n.id]?.join(", ") || "?"}]` } })));
        setGraphValid(r.valid); setNotice(r.message); return r.valid;
      } catch (e) { const message=translateBackendError(String(e));const failed=nodes.find(node=>message.includes(node.data.label)||message.includes(node.id));if(failed){setNodes(items=>items.map(node=>node.id===failed.id?{...node,data:{...node.data,error:message}}:node));setSelectedId(failed.id)}setGraphValid(false);setNotice(message); }
    }
    setGraphValid(true); return true;
  };

  const train = async (config: Record<string, number | string>) => {
    if (!project || !dataset || !task || !architecture || !(await validateGraph())) return;
    const targetId=activeTrainingRun?.id || newTrainingRun(trainingRuns.length+1, t("app:trainingRun.defaultName", { index: trainingRuns.length + 1 })).id;
    runningRunId.current=targetId;
    setTraining(true); setTrainingRuns(runs=>runs.map(item=>item.id===targetId?{...item,modelId:activeModelId || undefined,status:"running",config,history:[],result:null,error:undefined,startedAt:now(),completedAt:undefined}:item)); setNotice(t("app:training.preparing"));
    try {
      await startTraining({ project, dataset, task_id: task, architecture, graph: { nodes: syncOutputContract(nodes, edges, dataset), edges }, config });
    } catch (e) {
      setTraining(false); runningRunId.current=null; setTrainingRuns(runs=>runs.map(item=>item.id===targetId?{...item,status:"error",error:translateBackendError(String(e)),completedAt:now()}:item));setNotice(translateBackendError(String(e)));
    }
  };

  const infer = async (payload: { mode: string; values?: string; index?: number }) => {
    if (!project || !task || !architecture || !dataset) throw new Error(t("app:data.completeProjectData"));
    return backend<Record<string, unknown>>("inference.run", { project, task_id: task, architecture, dataset, checkpoint: activeTrainingRun?.result?.checkpoint, ...payload });
  };

  const addTrainingRun = () => {
    const fresh=newTrainingRun(trainingRuns.length+1, t("app:trainingRun.defaultName", { index: trainingRuns.length + 1 }));
    setTrainingRuns(items=>[...items,fresh]);setActiveRunId(fresh.id);
  };
  const switchModel = (id:string) => {
    const target=models.find(model=>model.id===id); if(!target || id===activeModelId)return;
    setModels(items=>items.map(model=>model.id===activeModelId?{...model,architecture:architecture || model.architecture,nodes,edges,graphValid}:model));
    setActiveModelId(id); setArchitecture(target.architecture); setNodes(target.nodes); setEdges(target.edges); setGraphValid(target.graphValid); setSelectedId(null);
  };
  const addModel = () => {
    if(!architecture || !task)return; const defaultName=t("app:projectModel.defaultName", { index: models.length + 1 }); const name=window.prompt(t("app:model.namePrompt"),defaultName)?.trim(); if(!name)return;
    const model=newProjectModel(models.length+1,architecture,task,defaultName); model.name=name; model.nodes=syncOutputContract(model.nodes,model.edges,dataset); setModels(items=>[...items,model]); switchModelAfterCreate(model);
  };
  const switchModelAfterCreate=(model:ProjectModel)=>{setActiveModelId(model.id);setArchitecture(model.architecture);setNodes(model.nodes);setEdges(model.edges);setGraphValid(false);setSelectedId(null)};

  const stepIndex=steps.findIndex(item=>item.id===step);
  const goPrevious=()=>{if(stepIndex>0)setStep(steps[stepIndex-1].id)};
  const goNext=async()=>{
    if(step==="model"){if(!task||!architecture){setNotice(t("app:notice.selectTaskAndArchitecture"));return}setStep("data");return}
    if(step==="data"){if(!dataset){setNotice(t("app:notice.loadDataset"));return}setStep("builder");return}
    if(step==="builder"){if(await validateGraph())setStep("training");return}
    if(step==="training"){if(!run){setNotice(t("app:notice.completeRun"));return}setStep("inference")}
  };

  if (!project) return <ProjectHub projects={projects} onCreate={createProject} onOpen={openProject} onDelete={deleteProject} theme={theme} setTheme={setTheme} accent={accent} setAccent={setAccent} />;

  return <div className="app-shell">
    <Sidebar step={step} setStep={setStep} completed={completed} onHome={() => setProject(null)} gpuInfo={gpuInfo} nodes={nodes} dataset={dataset} theme={theme} setTheme={setTheme} accent={accent} setAccent={setAccent} />
    <main className="main-area">
      {notice && <div className="notice"><span>{notice}</span><button onClick={() => setNotice("")}><X size={14}/></button></div>}
      <section className={`content ${step === "training" ? "no-scroll" : ""}`}>
        <div className={`stage-nav ${step==="builder"?"on-toolbar":""}`}><button className="secondary" onClick={goPrevious} disabled={stepIndex===0}><ArrowLeft/> {t("common:previous")}</button>{stepIndex<steps.length-1&&<button className="primary" onClick={goNext}>{t("common:next")} <ChevronRight/></button>}</div>
        {step === "model" && <ModelTask architecture={architecture} task={task} onChoose={choose} onContinue={goNext} />}
        {step === "data" && <DataStep project={project} task={task} dataset={dataset} onDownload={downloadData} onImport={importData} onChooseAnother={()=>setDataset(null)} onApplyPipeline={applyDataPipeline} onSplitsChange={splits => setDataset(current => current ? { ...current, splits } : current)} />}
        {step === "builder" && architecture && task && <Builder architecture={architecture} task={task} dataset={dataset} models={models} activeModelId={activeModelId} onSelectModel={switchModel} onAddModel={addModel} theme={theme} nodes={nodes} edges={edges} setNodes={setNodes} setEdges={setEdges} onNodesChange={(changes: Parameters<typeof onNodesChange>[0]) => { setGraphValid(false); onNodesChange(changes); }} onEdgesChange={(changes: Parameters<typeof onEdgesChange>[0]) => { setGraphValid(false); onEdgesChange(changes); }} onDirty={()=>setGraphValid(false)} selected={selected} setSelectedId={setSelectedId} onValidate={validateGraph} />}
        {step === "training" && task && <Training task={task} dataset={dataset} models={models} activeModelId={activeModelId} onSelectModel={switchModel} runs={trainingRuns} activeRunId={activeRunId} training={training} accent={accent} onSelectRun={setActiveRunId} onAddRun={addTrainingRun} onTrain={train} />}
        {step === "inference" && task && dataset && <Inference task={task} dataset={dataset} run={run} runName={activeTrainingRun?.name || t("app:inference.activeRun")} onInfer={infer} />}
      </section>
    </main>
  </div>;
}

function Builder({ architecture, task, dataset, models, activeModelId, onSelectModel, onAddModel, theme, nodes, edges, setNodes, setEdges, onNodesChange, onEdgesChange, onDirty, selected, setSelectedId, onValidate }: { architecture: Architecture; task:TaskId; dataset:DatasetSummary|null; models:ProjectModel[]; activeModelId:string|null; onSelectModel:(id:string)=>void; onAddModel:()=>void; theme: Theme; nodes: ModelNodeT[]; edges: ModelEdge[]; setNodes: React.Dispatch<React.SetStateAction<ModelNodeT[]>>; setEdges: React.Dispatch<React.SetStateAction<ModelEdge[]>>; onNodesChange: any; onEdgesChange: any; onDirty: () => void; selected: ModelNodeT | null; setSelectedId: (id: string | null) => void; onValidate: () => void }) {
  const { t } = useTranslation(["app", "common", "catalog"]);
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

  const liveErrors=useMemo(()=>graphErrorMap(nodes,edges,t),[nodes,edges,t]);
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

  const applyPreset=(preset:{nodes:ModelNodeT[];edges:ModelEdge[]})=>{onDirty();setNodes(syncOutputContract(preset.nodes,preset.edges,dataset).map(node=>({...node,data:{...node.data,error:undefined}})));setEdges(preset.edges);setSelectedId(null);setPresetsOpen(false);window.setTimeout(()=>flow?.fitView({padding:.18}),20)};
  const savePreset=()=>{if(!presetName.trim())return;setCustomPresets(items=>[...items,{id:uid(),name:presetName.trim(),description:t("app:customPreset.description"),nodes,edges}]);setPresetName("")};
  const presetI18n = (preset: typeof builtinPresets[number]) => {
    return {
      name: getPresetName(architecture, preset.id),
      description: getPresetDescription(architecture, preset.id),
      benefit: getPresetBenefit(architecture, preset.id),
      tradeoff: getPresetTradeoff(architecture, preset.id)
    };
  };

  return (
    <div className="builder-page">
      <div className="model-tabs" role="tablist" aria-label={t("app:model.modelsAria")}>{models.map(model=><button key={model.id} role="tab" aria-selected={model.id===activeModelId} className={model.id===activeModelId?"active":""} onClick={()=>onSelectModel(model.id)}><BrainCircuit size={13}/>{model.name}<small>{t(`catalog:architectures.${model.architecture}.name`)}</small></button>)}<button className="add-model" onClick={onAddModel} title={t("app:model.createTitle")} aria-label={t("app:model.createTitle")}><Plus size={15}/></button></div>
      <div className="builder-toolbar">
        <button className={libraryOpen ? "active" : ""} onClick={() => setLibraryOpen(!libraryOpen)}><Plus /> {t("common:blocks")}</button>
        <button className={presetsOpen ? "active" : ""} onClick={()=>setPresetsOpen(true)}><Layers3/> {t("common:presets")}</button>
        <span className="toolbar-separator" />
        <button onClick={onValidate}><Check /> {t("common:validate")}</button>
        <button onClick={onValidate}><Save /> {t("app:builder.saveRevision")}</button>
        <div className="toolbar-spacer" />
        <span>{t("app:builder.graphSummary", { blocks: nodes.length, connections: edges.length })}</span>
      </div>

      <div className="builder-workspace">
        {libraryOpen && (
          <aside className="block-library">
            <div><span className="eyebrow">{t("app:builder.libraryEyebrow")}</span><h3>{t(`catalog:architectures.${architecture}.name`)}</h3></div>
            {Object.entries(groups).map(([cat, blocks]) => (
              <section key={cat}>
                <small>{t(`catalog:blockCategories.${cat}`)}</small>
                {blocks.map(b => <div className="library-item" key={b.type} onMouseEnter={()=>{if(!pinnedBlock)setHoveredBlock(b)}} onMouseLeave={()=>setHoveredBlock(null)}><button draggable onClick={() => addBlock(b)} onDragStart={e => { e.dataTransfer.setData("application/modelbuilder", b.type); e.dataTransfer.effectAllowed = "move"; }}><Plus size={13} />{t(`catalog:blocks.${b.type}.label`)}</button><button className={`block-info-button ${pinnedBlock?.type===b.type?"active":""}`} onClick={event=>{event.stopPropagation();setPinnedBlock(b);setHoveredBlock(null)}} title={t("app:blockInfo.pinTitle", { block: t(`catalog:blocks.${b.type}.label`) })} aria-label={t("app:blockInfo.pinTitle", { block: t(`catalog:blocks.${b.type}.label`) })}><Info size={13}/></button></div>)}
              </section>
            ))}
          </aside>
        )}

        <div className="flow-wrap">
          <div className="canvas-param-badge">
            <Cpu size={13} />
            <span>{t("app:builder.paramEstimate", { count: paramEstimate > 0 ? paramEstimate.toLocaleString() : "---" })}</span>
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
                <span className="eyebrow">{t("app:inspector.properties")}</span>
                <h3>{t(`catalog:blocks.${selected.data.blockType}.label`, { defaultValue: selected.data.label })}</h3>
                <code>{selected.data.blockType}</code>
              </div>
              {Object.entries(selected.data.properties).length ? Object.entries(selected.data.properties).map(([k, v]) => (
                <Property key={k} name={k} value={v} disabled={!!dataset && ((selected.data.blockType==="linear"&&k==="out_features")||(selected.data.blockType==="conv2d"&&k==="out_channels")) && edges.some(edge=>edge.source===selected.id&&nodes.find(node=>node.id===edge.target)?.data.blockType==="output")} onChange={nv => updateProp(k, nv)} />
              )) : <p className="muted">{t("app:inspector.noParams")}</p>}
              <div className="inspector-section">
                <span className="eyebrow">{t("app:inspector.tensor")}</span>
                <div className="shape-box">{selected.data.shape || t("app:inspector.shapePlaceholder")}</div>
              </div>
              <div className="inspector-section parameter-summary"><span className="eyebrow">{t("app:inspector.parameters")}</span><strong>{estimateNodeParameters(selected).toLocaleString()}</strong><small>{t("app:inspector.parametersPerBlock")}</small><div><span>{t("app:inspector.totalModel")}</span><b>{paramEstimate.toLocaleString()}</b></div></div>
              {liveErrors.get(selected.id)&&<div className="inspector-error"><AlertCircle/>{translateBackendError(liveErrors.get(selected.id)!)}</div>}
            </>
          ) : (
            <div className="inspector-empty">
              <GitBranch />
              <h3>{t("app:inspector.emptyTitle")}</h3>
              <p>{t("app:inspector.emptyHint")}</p>
            </div>
          )}
        </aside>
      </div>
      {(pinnedBlock||hoveredBlock)&&(()=>{const block=pinnedBlock||hoveredBlock!;const blockLabel=t(`catalog:blocks.${block.type}.label`, { defaultValue: block.label });const description=t(`catalog:blocks.${block.type}.description`, { defaultValue: BLOCK_INFO[block.type]?.description });const usage=t(`catalog:blocks.${block.type}.usage`, { defaultValue: BLOCK_INFO[block.type]?.usage });const diagram=t(`catalog:blocks.${block.type}.diagram`, { returnObjects: true, defaultValue: BLOCK_INFO[block.type]?.diagram || [t("common:input"), blockLabel, t("common:output")] }) as string[];return <div ref={infoCard} className={`block-info-popover ${pinnedBlock?"pinned":"preview"}`}><div className="block-info-head"><span className="eyebrow">{pinnedBlock?t("app:blockInfo.pinnedEyebrow"):t("app:blockInfo.previewEyebrow")}</span>{pinnedBlock&&<button className="modal-close" onClick={()=>setPinnedBlock(null)} aria-label={t("app:blockInfo.closeInfo")}><X/></button>}</div><h2>{blockLabel}</h2><p>{description || t("app:blockInfo.fallbackDescription")}</p><div className="node-diagram">{diagram.map((part,index)=><div key={`${part}-${index}`}>{index>0&&<ChevronRight/>}<span className={index===1?"operation":""}>{part}</span></div>)}</div><h3>{t("app:blockInfo.usageTitle")}</h3><p>{usage || t("app:blockInfo.fallbackUsage")}</p><div className="default-properties"><span>{t("app:blockInfo.defaults")}</span><code>{Object.keys(block.defaults).length?JSON.stringify(block.defaults):t("app:blockInfo.noEditableDefaults")}</code></div>{!pinnedBlock&&<small className="pin-hint">{t("app:blockInfo.pinHint")}</small>}</div>})()}
      {presetsOpen&&<div className="modal-backdrop"><div className="modal presets-modal"><button className="modal-close" onClick={()=>setPresetsOpen(false)}><X/></button><span className="eyebrow">{t("app:presets.modalEyebrow")}</span><h2>{t("app:presets.modalTitle")}</h2><div className="preset-list">{builtinPresets.map(preset=>{const info=presetI18n(preset);return <article key={preset.id}><div><Sparkles/><span><strong>{info.name}</strong><small>{info.description}</small></span></div><p><b>{t("app:presets.benefitLabel")}</b> {info.benefit}. <b>{t("app:presets.tradeoffLabel")}</b> {info.tradeoff}.</p><button className="secondary" onClick={()=>applyPreset(preset)}>{t("app:presets.apply")}</button></article>;})}{customPresets.map(preset=><article key={preset.id}><div><Save/><span><strong>{preset.name}</strong><small>{preset.description}</small></span></div><p><b>{t("app:presets.benefitLabel")}</b> {t("app:customPreset.benefit")}. <b>{t("app:presets.tradeoffLabel")}</b> {t("app:customPreset.tradeoff")}.</p><button className="secondary" onClick={()=>applyPreset(preset)}>{t("app:presets.apply")}</button></article>)}</div><div className="save-preset"><label>{t("app:savePreset.label")}<input value={presetName} onChange={event=>setPresetName(event.target.value)} placeholder={t("app:savePreset.placeholder")}/></label><button className="primary" disabled={!presetName.trim()} onClick={savePreset}><Save/> {t("common:save")}</button></div></div></div>}
    </div>
  );
}

function Property({ name, value, disabled=false, onChange }: { name: string; value: string | number | boolean; disabled?:boolean; onChange: (v: string | number | boolean) => void }) {
  const { t } = useTranslation(["app", "common"]);
  return (
    <label className="property">
      <span>{name.replaceAll("_", " ")}</span>
      {typeof value === "boolean" ? (
        <button className={`toggle ${value ? "on" : ""}`} onClick={() => onChange(!value)} type="button"><i /></button>
      ) : (
        <input disabled={disabled} title={disabled ? t("app:property.autoAdjusted") : undefined} type={typeof value === "number" ? "number" : "text"} step="any" value={String(value)} onChange={e => onChange(typeof value === "number" ? Number(e.target.value) : e.target.value)} />
      )}
    </label>
  );
}

function Training({ task, dataset, models, activeModelId, onSelectModel, runs, activeRunId, training, accent, onSelectRun, onAddRun, onTrain }: { task: TaskId; dataset:DatasetSummary|null; models:ProjectModel[]; activeModelId:string|null; onSelectModel:(id:string)=>void; runs: TrainingRun[]; activeRunId: string; training: boolean; accent: Accent; onSelectRun:(id:string)=>void; onAddRun:()=>void; onTrain: (c: Record<string, number | string>) => void }) {
  const { t } = useTranslation(["app", "common", "catalog"]);
  const activeRun=runs.find(item=>item.id===activeRunId) || runs[0];
  const history=activeRun?.history || [];
  const run=activeRun?.result || null;
  const defaultEpochs=task==="text.language_model"?10:task.startsWith("image.segmentation")?40:20;
  const defaultBestCriterion="val_loss";
  const [epochs, setEpochs] = useState(defaultEpochs);
  const [lr, setLr] = useState(0.001);
  const [batch, setBatch] = useState(Number(dataset?.options?.batchSize||32));
  const [optimizer, setOptimizer] = useState("adamw");
  const [bestModelCriterion, setBestModelCriterion] = useState(defaultBestCriterion);
  useEffect(()=>{
    const config=activeRun?.config || {};
    setEpochs(Number(config.epochs || defaultEpochs));setLr(Number(config.learning_rate || .001));setBatch(Number(config.batch_size || dataset?.options?.batchSize || 32));setOptimizer(String(config.optimizer || "adamw"));setBestModelCriterion(String(config.best_model_criterion || defaultBestCriterion));
  },[activeRunId,dataset?.options?.batchSize,defaultEpochs,defaultBestCriterion]);
  const currentEpoch=history.at(-1)?.epoch || 0;
  const targetEpochs=Number(activeRun?.config.epochs || epochs || 1);
  const progress=Math.min(100,Math.round(currentEpoch/targetEpochs*100));
  const elapsedSeconds=activeRun?.startedAt ? Math.max(0,Math.round(((activeRun.completedAt ? new Date(activeRun.completedAt).getTime() : Date.now())-new Date(activeRun.startedAt).getTime())/1000)) : 0;
  const etaSeconds=currentEpoch ? Math.max(0,Math.round((elapsedSeconds/currentEpoch)*(targetEpochs-currentEpoch))) : 0;
  const formatTime=(seconds:number)=>seconds<60?t("app:time.seconds", { count: seconds }):t("app:time.minutes", { minutes: Math.floor(seconds/60), seconds: seconds%60 });
  const bestLoss=history.length ? history.reduce((best, point)=>point.valLoss<best.valLoss?point:best,history[0]) : null;
  const bestMetric=history.length ? history.reduce((best, point)=>point.metric>best.metric?point:best,history[0]) : null;

  const metricName = t(`catalog:tasks.${task}.metric`);
  return (
    <div className="page training-page">
      <div className="page-intro">
        <div>
          <span className="eyebrow">{t("app:training.stepEyebrow")}</span>
          <h2>{t("app:training.title")}</h2>
          <p>{t("app:training.subtitle", { metric: metricName })}</p>
        </div>
      </div>

      <div className="run-tabs" role="tablist" aria-label={t("app:training.runsAriaLabel")}>
        {runs.map(item=><button key={item.id} role="tab" aria-selected={item.id===activeRunId} className={item.id===activeRunId?"active":""} onClick={()=>onSelectRun(item.id)}><i className={`run-dot ${item.status}`}/><span>{item.name}</span>{item.status==="completed"&&<Check size={12}/>}</button>)}
        <button className="add-run" onClick={onAddRun} disabled={training} title={t("app:training.addRunTitle")}><Plus size={15}/></button>
      </div>

      <div className="training-layout">
        <div className="charts">
          <article className="panel chart-panel">
            <div className="chart-title">
              <div><span className="eyebrow">{t("app:training.chartMainEyebrow")}</span><h3>{t("app:training.loss")}</h3></div>
              <div className="chart-status"><span className="live"><i /> {training ? t("app:training.statusLive") : run ? t("app:training.statusCompleted") : t("app:training.statusReady")}</span>{bestLoss&&<span className="best-value">{t("app:training.bestValLoss", { loss: bestLoss.valLoss.toFixed(4), epoch: bestLoss.epoch })}</span>}</div>
            </div>
            <TrainingChart history={history} metric="loss" accent={accent} />
          </article>
          <article className="panel chart-panel">
            <div className="chart-title">
              <div><span className="eyebrow">{t("app:training.taskMetricEyebrow")}</span><h3>{metricName}</h3></div>
              {bestMetric&&<span className="best-value">{t("app:training.bestMetric", { metric: metricName, value: bestMetric.metric.toFixed(4), epoch: bestMetric.epoch })}</span>}
            </div>
            <TrainingChart history={history} metric={metricName} accent={accent} compact />
          </article>
        </div>

        <aside className="run-config">
          <div className="run-config-scroll">
          <label>{t("app:training.modelLabel")}
            <select value={activeModelId || ""} onChange={event=>onSelectModel(event.target.value)} disabled={training}>
              {models.map(model=><option value={model.id} key={model.id}>{model.name} · {t(`catalog:architectures.${model.architecture}.name`)}</option>)}
            </select>
          </label>
          <label>{t("app:training.epochsLabel")}
            <input type="number" min="1" max="200" value={epochs} onChange={e => setEpochs(Number(e.target.value))} />
          </label>
          <label>{t("app:training.lrLabel")}
            <input type="number" step="0.0001" value={lr} onChange={e => setLr(Number(e.target.value))} />
          </label>
          <label>{t("app:training.batchLabel")}
            <input type="number" min="1" value={batch} onChange={e => setBatch(Number(e.target.value))} />
          </label>
          <label>{t("app:training.optimizerLabel")}
            <select value={optimizer} onChange={e => setOptimizer(e.target.value)}>
              <option value="adamw">AdamW</option>
              <option value="adam">Adam</option>
              <option value="sgd">SGD</option>
            </select>
          </label>          <label>{t("app:training.bestModelCriterionLabel")}
            <select value={bestModelCriterion} onChange={e => setBestModelCriterion(e.target.value)}>
              <option value="none">{t("app:training.criterion.none")}</option>
              <option value="val_loss">{t("app:training.criterion.val_loss")}</option>
              <option value="train_loss">{t("app:training.criterion.train_loss")}</option>
              <option value="metric">{t("app:training.criterion.metric", { metric: metricName })}</option>
            </select>
          </label>

          <div className="resource">
            <Cpu />
            <span><small>{t("app:training.deviceLabel")}</small><strong>{t("app:training.deviceValue")}</strong></span>
          </div>
          {(history.length > 0 || activeRun?.startedAt) && <><div className="run-progress-head"><span>{t("app:training.epochProgress", { current: currentEpoch, target: targetEpochs })}</span><strong>{progress}%</strong></div><div className="run-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}><i style={{width:`${progress}%`}}/></div><div className="time-summary"><span><Clock3/> {t("app:training.elapsed")} <strong>{formatTime(elapsedSeconds)}</strong></span>{activeRun?.status==="running"&&<span>{t("app:training.remaining")} <strong>{currentEpoch?formatTime(etaSeconds):t("app:training.calculating")}</strong></span>}</div></>}
          {run?.test&&<div className="test-summary"><CheckCircle2/><span><small>{t("app:training.testResult")}</small><strong>{run.metricName}: {run.test.metric.toFixed(4)}</strong></span></div>}
          {activeRun?.error&&<div className="run-error"><AlertCircle/>{translateBackendError(activeRun.error)}</div>}
          </div>
          <div className="run-config-footer"><button className="primary full" disabled={training || activeRun?.status==="completed"} onClick={() => onTrain({ epochs, learning_rate: lr, batch_size: batch, optimizer, best_model_criterion: bestModelCriterion })}>{activeRun?.status==="running" ? <><Activity className="spin" /> {t("app:training.trainingButton")}</> : activeRun?.status==="completed" ? <><Check/> {t("app:training.completedButton")}</> : <><Play /> {t("app:training.startButton")}</>}</button>{activeRun?.status==="completed"&&<small>{t("app:training.newRunHint")}</small>}</div>
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
  const { t, i18n } = useTranslation(["app", "common"]);
  const [open,setOpen]=useState(false);
  const root=useRef<HTMLDivElement>(null);
  useEffect(()=>{
    if(!open)return;
    const close=(event:PointerEvent)=>{if(root.current&&!root.current.contains(event.target as Node))setOpen(false)};
    const escape=(event:KeyboardEvent)=>{if(event.key==="Escape")setOpen(false)};
    document.addEventListener("pointerdown",close); document.addEventListener("keydown",escape);
    return()=>{document.removeEventListener("pointerdown",close);document.removeEventListener("keydown",escape)};
  },[open]);
  const colors:Array<{id:Accent;label:string}>=[{id:"blue",label:t("common:blue")},{id:"orange",label:t("common:orange")},{id:"green",label:t("common:green")},{id:"violet",label:t("common:violet")}];
  const languages:Array<{id:string;label:string}>=[{id:"en",label:"English"},{id:"es",label:"Español"}];
  return <div className="appearance" ref={root}>
    <button className={`icon-button ${open?"active":""}`} onClick={()=>setOpen(value=>!value)} aria-label={t("common:settings")} aria-haspopup="menu" aria-expanded={open}><Palette/></button>
    {open&&<div className="appearance-menu" role="menu">
      <div className="appearance-title"><Palette/><span><strong>{t("common:settings")}</strong><small>{t("common:appearanceDescription")}</small></span></div>
      <span className="menu-label">{t("common:language")}</span>
      <div className="theme-options">{languages.map(lang=><button key={lang.id} className={i18n.language===lang.id?"selected":""} onClick={()=>i18n.changeLanguage(lang.id)}>{lang.label}</button>)}</div>
      <span className="menu-label">{t("app:appearance.themeLabel")}</span>
      <div className="theme-options"><button className={theme==="dark"?"selected":""} onClick={()=>setTheme("dark")}><Moon/> {t("common:dark")}</button><button className={theme==="light"?"selected":""} onClick={()=>setTheme("light")}><Sun/> {t("common:light")}</button></div>
      <span className="menu-label">{t("app:appearance.accentColorLabel")}</span>
      <div className="accent-options">{colors.map(color=><button key={color.id} className={accent===color.id?"selected":""} data-color={color.id} onClick={()=>setAccent(color.id)} title={color.label} aria-label={color.label} aria-pressed={accent===color.id}><i/>{accent===color.id&&<Check/>}</button>)}</div>
    </div>}
  </div>;
}

function ProjectHub({ projects, onCreate, onOpen, onDelete, theme, setTheme, accent, setAccent }: { projects: Project[]; onCreate: (n:string,d:string)=>void; onOpen:(p:Project)=>void; onDelete:(p:Project)=>void; theme:Theme; setTheme:(v:Theme)=>void; accent:Accent; setAccent:(v:Accent)=>void }) {
  const { t } = useTranslation(["app", "common", "catalog"]);
  const [creating, setCreating] = useState(false); const [name, setName] = useState(""); const [description, setDescription] = useState("");
  const [projectPendingDeletion,setProjectPendingDeletion]=useState<Project|null>(null);
  const [query,setQuery]=useState("");
  const [modelFilter,setModelFilter]=useState("all");
  const [projectView,setProjectView]=useState<"list"|"grid">("list");
  const filteredProjects=projects.filter(item=>(modelFilter==="all"||item.architecture===modelFilter)&&`${item.name} ${item.description}`.toLowerCase().includes(query.toLowerCase()));
  const modelCounts=projects.reduce<Record<string,number>>((counts,item)=>{const key=item.architecture||"unconfigured";counts[key]=(counts[key]||0)+1;return counts},{});
  const maxCount=Math.max(1,...Object.values(modelCounts));
  const recentActivity=projects.slice().sort((a,b)=>new Date(b.updatedAt).getTime()-new Date(a.updatedAt).getTime()).slice(0,4);
  return <div className="hub"><header className="hub-header"><div className="brand" title={t("common:appName")} aria-label={t("common:appName")}><span className="brand-mark"><BrainCircuit /></span></div><AppearanceMenu theme={theme} setTheme={setTheme} accent={accent} setAccent={setAccent}/></header>
    <div className="hub-content">
      <section className="hub-dashboard-top">
        <div className="hub-hero-card"><div className="hero"><span className="eyebrow">{t("app:hub.heroEyebrow")}</span><h1>{t("app:hub.heroTitle")}<br/><em>{t("app:hub.heroTitleEmphasis")}</em></h1><p>{t("app:hub.heroSubtitle")}</p><div className="hero-actions"><button className="primary" onClick={() => setCreating(true)}><Plus size={17}/> {t("app:hub.newProjectButton")}</button></div></div><div className="hub-cubes" aria-hidden="true"><i/><i/><i/></div><div className="hub-benefits"><span><Box/><b>{t("app:hub.flexible")}</b><small>{t("app:hub.flexibleHint")}</small></span><span><Zap/><b>{t("app:hub.local")}</b><small>{t("app:hub.localHint")}</small></span><span><Shield/><b>{t("app:hub.open")}</b><small>{t("app:hub.openHint")}</small></span></div></div>
        <div className="hub-summary-stack">
          <article className="hub-summary-card"><h2>{t("app:hub.recentModelTypes")}</h2><div className="model-usage-list">{Object.entries(modelCounts).sort((a,b)=>b[1]-a[1]).slice(0,4).map(([id,count])=><div key={id}><span>{id==="unconfigured"?t("app:hub.unconfigured"):t(`catalog:architectures.${id}.name`)}</span><i><b style={{width:`${Math.max(12,count/maxCount*100)}%`}}/></i><strong>{count}</strong></div>)}{!Object.keys(modelCounts).length&&<p>{t("app:hub.noModelStats")}</p>}</div></article>
          <article className="hub-summary-card activity-card"><div className="summary-title"><h2>{t("app:hub.recentActivity")}</h2><small>{t("app:hub.viewAll")}</small></div><div className="activity-list">{recentActivity.map(item=><button key={item.id} onClick={()=>onOpen(item)}><span><CheckCircle2/></span><b>{item.name}</b><time>{new Date(item.updatedAt).toLocaleDateString()}</time></button>)}{!recentActivity.length&&<p>{t("app:hub.noActivity")}</p>}</div></article>
        </div>
      </section>
      <section className="projects-card">
        <div className="projects-toolbar"><div><h2>{t("app:hub.recentProjects")}</h2><p>{projects.length ? t("app:hub.projectCount", { count: projects.length }) : t("app:hub.noProjects")}</p></div><div className="project-controls"><label><Search/><input value={query} onChange={event=>setQuery(event.target.value)} placeholder={t("app:hub.searchProjects")}/></label><select value={modelFilter} onChange={event=>setModelFilter(event.target.value)} aria-label={t("app:hub.filterModels")}><option value="all">{t("app:hub.allModels")}</option>{Object.keys(modelCounts).filter(id=>id!=="unconfigured").map(id=><option value={id} key={id}>{t(`catalog:architectures.${id}.name`)}</option>)}</select><div className="view-toggle"><button className={projectView==="list"?"active":""} onClick={()=>setProjectView("list")} aria-label={t("app:hub.listView")}><List/></button><button className={projectView==="grid"?"active":""} onClick={()=>setProjectView("grid")} aria-label={t("app:hub.gridView")}><Grid2X2/></button></div></div></div>
      {filteredProjects.length ? (
        <div className={`project-table ${projectView}`}>
          <div className="project-table-head"><span>{t("app:hub.nameColumn")}</span><span>{t("app:hub.modelColumn")}</span><span>{t("app:hub.lastOpenedColumn")}</span><span>{t("app:hub.actionsColumn")}</span></div>
          <div className="project-table-scroll">{filteredProjects.map(p => (
            <div className="project-row-wrap" key={p.id}>
              <button className="project-row" onClick={() => onOpen(p)}>
                <span className="project-icon"><Box/></span>
                <span><strong>{p.name}</strong><small>{p.description || t("app:hub.projectFallbackDescription")}</small></span>
                <span className="pill">{p.architecture?.toUpperCase() || t("app:hub.unconfigured")}</span>
                <time>{new Date(p.updatedAt).toLocaleDateString()}</time>
              </button>
              <button className="project-delete-btn" onClick={(e) => { e.stopPropagation(); setProjectPendingDeletion(p); }} title={t("app:hub.deleteProjectTitle")} aria-label={t("app:hub.deleteProjectTitle")}>
                <Trash2 size={16} />
              </button>
            </div>
          ))}</div>
        </div>
      ) : (
        <div className="empty-state"><Sparkles/><h3>{t("app:hub.emptyTitle")}</h3><p>{query||modelFilter!=="all"?t("app:hub.noSearchResults"):t("app:hub.emptyHint")}</p></div>
      )}
      </section>
    </div>
    {creating && <div className="modal-backdrop"><form className="modal" onSubmit={e => { e.preventDefault(); if(name.trim()) { onCreate(name.trim(), description.trim()); setCreating(false); } }}><button type="button" className="modal-close" onClick={() => setCreating(false)}><X/></button><span className="eyebrow">{t("app:hub.createProjectEyebrow")}</span><h2>{t("app:hub.createProjectTitle")}</h2><label>{t("app:hub.projectNameLabel")}<input autoFocus value={name} onChange={e=>setName(e.target.value)} placeholder={t("app:hub.projectNamePlaceholder")}/></label><label>{t("app:hub.projectDescriptionLabel")}<textarea value={description} onChange={e=>setDescription(e.target.value)} placeholder={t("app:hub.projectDescriptionPlaceholder")}/></label><div className="modal-actions"><button type="button" className="secondary" onClick={()=>setCreating(false)}>{t("common:cancel")}</button><button className="primary" disabled={!name.trim()}>{t("app:hub.createProjectButton")} <ChevronRight size={16}/></button></div></form></div>}
    {projectPendingDeletion&&<div className="modal-backdrop" onMouseDown={()=>setProjectPendingDeletion(null)}><section className="modal delete-project-modal" role="dialog" aria-modal="true" aria-labelledby="delete-project-title" onMouseDown={event=>event.stopPropagation()}><button type="button" className="modal-close" onClick={() => setProjectPendingDeletion(null)} aria-label={t("common:close")}><X/></button><span className="eyebrow">{t("app:deleteProject.eyebrow")}</span><h2 id="delete-project-title">{t("app:deleteProject.title")}</h2><p>{t("app:deleteProject.confirm", { name: projectPendingDeletion.name })}</p><div className="modal-actions"><button type="button" className="secondary" onClick={()=>setProjectPendingDeletion(null)}>{t("common:cancel")}</button><button type="button" className="danger-button" onClick={()=>{const target=projectPendingDeletion;setProjectPendingDeletion(null);void onDelete(target);}}><Trash2 size={15}/>{t("common:delete")}</button></div></section></div>}
  </div>;
}

function computeVramEstimate(nodes: ModelNodeT[], dataset: DatasetSummary | null, totalVramBytes: number, t: (key: string, opts?: Record<string, string | number>) => string) {
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
    text: t("app:sidebar.vramEstimate", { estimated: estimatedMB, total: totalGB, percent }),
    percent: Math.min(100, Number(percent))
  };
}

function Sidebar({ step, setStep, completed, onHome, gpuInfo, nodes, dataset, theme, setTheme, accent, setAccent }: { step:Step; setStep:(s:Step)=>void; completed:Record<Step,boolean>; onHome:()=>void; gpuInfo: GpuInfo | null; nodes: ModelNodeT[]; dataset: DatasetSummary | null; theme:Theme; setTheme:(value:Theme)=>void; accent:Accent; setAccent:(value:Accent)=>void }) {
  const { t } = useTranslation(["app", "common"]);
  const steps = useSteps();
  const totalVram = gpuInfo?.total_vram || 12487661158;
  const gpuName = gpuInfo?.name || "NVIDIA GeForce RTX 3060";
  const vramEst = computeVramEstimate(nodes, dataset, totalVram, t);

  return <aside className="sidebar">
    <div className="sidebar-top">
      <button className="brand sidebar-brand" onClick={onHome} title={t("app:sidebar.homeTitle")} aria-label={t("app:sidebar.homeTitle")}><span className="brand-mark"><BrainCircuit/></span></button>
      <AppearanceMenu theme={theme} setTheme={setTheme} accent={accent} setAccent={setAccent}/>
    </div>
    <nav><small>{t("app:sidebar.flowLabel")}</small>{steps.map((s,i) => { const Icon=s.icon; return <button key={s.id} className={step===s.id?"active":""} onClick={()=>setStep(s.id)}><span className="step-index">{completed[s.id]?<Check size={13}/>:i+1}</span><Icon size={16}/><span>{s.label}</span></button>; })}</nav>
    <div className="sidebar-bottom">
      <div><Cpu size={16}/><span><strong>{t("app:sidebar.engineLabel")}</strong><small>{t("app:sidebar.engineSublabel")}</small></span></div>
      <div className="vram-widget">
        <span className="vram-header">{t("app:sidebar.memoryLabel")}</span>
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

function ModelTask({ architecture, task, onChoose, onContinue }: { architecture:Architecture|null; task:TaskId|null; onChoose:(a:Architecture,t:TaskId)=>void; onContinue:()=>void }) {
  const { t } = useTranslation(["app", "catalog"]);
  const [family,setFamily]=useState("all");
  const [query,setQuery]=useState("");
  const [preview,setPreview]=useState<Architecture>(architecture||"mlp");
  const families:Array<{id:string;label:string;hint:string;icon:typeof BrainCircuit;architectures:Architecture[]}>=useMemo(()=>[
    {id:"all",label:t("app:modelTask.familyAll"),hint:t("app:modelTask.familyAllHint"),icon:SlidersHorizontal,architectures:Object.keys(ARCHITECTURES) as Architecture[]},
    {id:"tabular",label:t("app:modelTask.familyTabular"),hint:t("app:modelTask.familyTabularHint"),icon:Table2,architectures:["mlp","autoencoder"]},
    {id:"series",label:t("app:modelTask.familySeries"),hint:t("app:modelTask.familySeriesHint"),icon:Activity,architectures:["cnn1d","lstm","transformer"]},
    {id:"vision",label:t("app:modelTask.familyVision"),hint:t("app:modelTask.familyVisionHint"),icon:Image,architectures:["cnn","vit","unet"]},
    {id:"text",label:t("app:modelTask.familyText"),hint:t("app:modelTask.familyTextHint"),icon:FileText,architectures:["transformer","transformer_causal"]},
    {id:"generative",label:t("app:modelTask.familyGenerative"),hint:t("app:modelTask.familyGenerativeHint"),icon:Sparkles,architectures:["autoencoder"]},
    {id:"causal",label:t("app:modelTask.familyCausal"),hint:t("app:modelTask.familyCausalHint"),icon:Network,architectures:["transformer_causal"]}
  ],[t]);
  const allowed=families.find(item=>item.id===family)?.architectures||families[0].architectures;
  const visibleArchitectures=(Object.entries(ARCHITECTURES) as [Architecture,typeof ARCHITECTURES[Architecture]][]).filter(([id])=>allowed.includes(id)).filter(([id])=>`${t(`catalog:architectures.${id}.name`)} ${t(`catalog:architectures.${id}.description`)}`.toLowerCase().includes(query.toLowerCase()));
  const detail=ARCHITECTURES[preview];
  const DetailIcon=ARCHITECTURE_ICONS[preview];
  const activeTask=architecture===preview?task:null;
  return <div className="page model-page model-catalog-page">
    <div className="model-catalog-header"><div><span className="eyebrow">{t("app:modelTask.stepEyebrow")}</span><h2>{t("app:modelTask.catalogTitle")}</h2><p>{t("app:modelTask.catalogSubtitle")}</p></div><div className="catalog-tools"><label><Search/><input value={query} onChange={event=>setQuery(event.target.value)} placeholder={t("app:modelTask.searchPlaceholder")}/><kbd>⌘ K</kbd></label><button><ArrowUpDown/>{t("app:modelTask.popularFirst")}<ChevronRight/></button></div></div>
    <div className="model-catalog-layout">
      <section className="family-panel"><header><h3>{t("app:modelTask.familiesTitle")}</h3><p>{t("app:modelTask.familiesSubtitle")}</p></header><div className="family-list">{families.map(item=>{const Icon=item.icon;return <button key={item.id} className={family===item.id?"active":""} onClick={()=>{setFamily(item.id);if(!item.architectures.includes(preview))setPreview(item.architectures[0])}}><span><Icon/></span><div><strong>{item.label}</strong><small>{item.hint}</small></div><b>{item.architectures.length}</b></button>})}</div></section>
      <section className="architecture-list-panel"><header><h3>{t("app:modelTask.architecturesTitle")}</h3><p>{t("app:modelTask.architectureCount",{count:visibleArchitectures.length})}</p></header><div className="catalog-architecture-list">{visibleArchitectures.map(([id,a],index)=>{const Icon=ARCHITECTURE_ICONS[id];return <button key={id} className={preview===id?"active":""} onClick={()=>setPreview(id)}><span><Icon/></span><div><strong>{t(`catalog:architectures.${id}.name`)}</strong><small>{t(`catalog:architectures.${id}.description`)}</small><em>{t(`catalog:tasks.${a.tasks[0]}.modality`)}</em></div>{index===0&&<b>◆ {t("app:modelTask.popular")}</b>}<ChevronRight/></button>})}</div></section>
      <section className="architecture-detail-panel"><div className="detail-scroll"><header><span className="detail-arch-icon"><DetailIcon/></span><div><h2>{t(`catalog:architectures.${preview}.name`)}</h2><span className="eyebrow">{t("app:modelTask.architectureEyebrow")}</span><p>{t(`catalog:architectures.${preview}.description`)}</p></div><b>◆ {t("app:modelTask.popular")}</b></header><nav><button className="active">{t("app:modelTask.overview")}</button><button>{t("app:modelTask.tasks")}</button><button>{t("app:modelTask.details")}</button></nav><div className="architecture-overview"><div><p>{t("app:modelTask.overviewText",{architecture:t(`catalog:architectures.${preview}.name`)})}</p><div className="detail-tags"><span>{t(`catalog:tasks.${detail.tasks[0]}.modality`)}</span><span>{t("app:modelTask.scalable")}</span><span>{t("app:modelTask.localReady")}</span></div></div><div className="mini-architecture-diagram"><small>{t("app:modelTask.inputSequence")}</small><i/><b>{t(`catalog:architectures.${preview}.name`)}</b><i/><small>{t("app:modelTask.representation")}</small><div>{[1,2,3,4,5,6].map(n=><span key={n}/>)}</div></div></div><div className="available-tasks"><h3>{t("app:modelTask.availableTasks")}</h3><p>{t("app:modelTask.availableTasksHint")}</p><div>{detail.tasks.map(taskId=><button key={taskId} className={activeTask===taskId?"selected":""} onClick={()=>onChoose(preview,taskId)}><span><FileText/></span><div><strong>{t(`catalog:tasks.${taskId}.name`)}</strong><small>{t(`catalog:tasks.${taskId}.input`)} → {t(`catalog:tasks.${taskId}.output`)}</small></div><ChevronRight/></button>)}</div></div>{activeTask&&<div className="detail-contract"><div><small>{t("app:contract.compatibleInput")}</small><strong>{t(`catalog:tasks.${activeTask}.format`)}</strong></div><div><small>{t("app:contract.mainMetric")}</small><strong>{t(`catalog:tasks.${activeTask}.metric`)}</strong></div><div><small>{t("app:contract.modality")}</small><strong>{t(`catalog:tasks.${activeTask}.modality`)}</strong></div></div>}</div><footer><span>{activeTask?t(`catalog:tasks.${activeTask}.name`):t("app:modelTask.chooseTaskHint")}</span><button className="primary" disabled={!activeTask} onClick={onContinue}>{t("app:modelTask.continueWith",{task:activeTask?t(`catalog:tasks.${activeTask}.name`):t("app:modelTask.taskFallback")})}<ChevronRight/></button></footer></section>
    </div>
  </div>;
}

function DataStep({ project, task, dataset, onDownload, onImport, onChooseAnother, onApplyPipeline, onSplitsChange }: {
  project: Project | null;
  task: TaskId | null;
  dataset: DatasetSummary | null;
  onDownload: (datasetId: string, opts?: DatasetOptions) => Promise<void>;
  onImport: (path: string, opts?: DatasetOptions) => Promise<void>;
  onChooseAnother: () => void;
  onApplyPipeline: (pipeline: DataPipeline) => Promise<void>;
  onSplitsChange: (splits: DatasetSummary["splits"]) => void;
}) {
  const { t } = useTranslation(["app", "common", "catalog"]);
  const [path, setPath] = useState(dataset?.sourcePath || "");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [loadingLabel, setLoadingLabel] = useState("");
  const [dataView,setDataView]=useState<"preview"|"charts">("preview");
  const [analytics,setAnalytics]=useState<DatasetAnalytics|undefined>(dataset?.analytics);
  const [catalog,setCatalog]=useState<HuggingFaceDataset[]>([]);
  const [catalogError,setCatalogError]=useState("");
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
  useEffect(()=>{
    if(dataset?.source!=="huggingface")return;
    backend<{exists:boolean}>("data.exists",{dataset}).then(result=>{if(!result.exists)onChooseAnother()}).catch(()=>onChooseAnother());
  },[dataset?.id,dataset?.revision]);
  useEffect(()=>{
    if(!task||!project){setCatalog([]);return}
    backend<HuggingFaceDataset[]>("data.catalog",{project,task_id:task}).then(items=>{setCatalog(items);setCatalogError("")}).catch(error=>setCatalogError(translateBackendError(String(error))));
  },[task,project?.id,dataset?.id]);

  if (!task) return <Blocked message={t("app:blocked.selectTaskArchitecture")} />;

  const trainCount = dataset ? (dataset.splitCounts?.train ?? Math.round(dataset.samples * dataset.splits.train / 100)) : 0;
  const validationCount = dataset ? (dataset.splitCounts?.validation ?? Math.round(dataset.samples * dataset.splits.validation / 100)) : 0;
  const testCount = dataset ? (dataset.splitCounts?.test ?? dataset.samples - trainCount - validationCount) : 0;

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
    else if(!isTauri()) window.alert(t("app:data.browseNotTauri"));
  };

  return (
    <div className="page data-page">
      {loading&&<div className="loading-banner"><RefreshCw className="spin"/><span>{loadingLabel}</span><div><i style={{width:`${progress}%`}}/></div><strong>{progress}%</strong></div>}
      <div className="page-intro">
        <div>
          <span className="eyebrow">{t("app:data.stepEyebrow")}</span>
          <h2>{t("app:data.title")}</h2>
        </div>
      </div>

      {!dataset ? (
        <div className="dataset-source-layout">
          <section className="hf-catalog">
            <div className="hf-catalog-heading"><div><Cloud/><span><span className="eyebrow">{t("app:data.hfEyebrow")}</span><h3>{t("app:data.hfTitle")}</h3><p>{t("app:data.hfSubtitle")}</p></span></div></div>
            {catalogError&&<div className="catalog-error"><AlertCircle/>{catalogError}</div>}
            <div className="hf-dataset-list">
              {catalog.map(item=><article className="hf-dataset-card" key={item.id}>
                <div className="hf-dataset-copy"><div className="hf-dataset-title"><Database/><div><h4>{item.name}</h4><a href={`https://huggingface.co/datasets/${item.repoId}`} target="_blank" rel="noreferrer">{item.repoId}</a></div></div><p>{item.description}</p><div className="hf-dataset-meta"><span>{formatBytes(item.sizeBytes)}</span><span>{item.license}</span>{item.installed?<span className="installed"><CheckCircle2/>{t("app:data.installed")}</span>:item.cached?<span className="cached"><Save/>{t("app:data.cached")}</span>:<span><Cloud/>{t("app:data.remote")}</span>}</div></div>
                <button className="primary" disabled={loading} onClick={()=>runLoading(item.installed?t("app:data.openingDataset"):item.cached?t("app:data.preparingCached"):t("app:data.downloadingDataset",{name:item.name}),()=>onDownload(item.id,item.defaultOptions))}>{item.installed?<><Check/> {t("app:data.useDataset")}</>:item.cached?<><Database/> {t("app:data.prepareDataset")}</>:<><Download/> {t("app:data.downloadDataset")}</>}</button>
              </article>)}
            </div>
          </section>

          <article className="source-card">
            <Upload />
            <h3>{t("app:data.importTitle")}</h3>
            <p>{t("app:data.importSubtitle")}</p>
            <label className="app-field">
              <span>{t("app:data.localPathLabel")}</span>
              <div className="path-picker"><input value={path} onChange={e => setPath(e.target.value)} placeholder={t("app:data.pathPlaceholder")} /><button type="button" className="secondary" onClick={browseFolder}><FolderOpen size={15}/> {t("app:data.browse")}</button></div>
            </label>
            <button className="secondary" disabled={!path||loading} onClick={() => runLoading(t("app:data.importLoading"),()=>onImport(path, currentOpts))}>
              {t("app:data.inspectImport")}
            </button>
          </article>
        </div>
      ) : (
        <div className="data-step-layout">
          <div className="data-main">
            <DataPipelineEditor task={task} dataset={dataset} onApply={onApplyPipeline}/>
            <article className="panel">
              <div className="data-view-header"><span className="eyebrow">{t("app:data.explorerEyebrow")}</span><div className="data-view-switch" role="tablist" aria-label={t("app:data.viewAria")}><button role="tab" aria-selected={dataView==="preview"} className={dataView==="preview"?"active":""} onClick={()=>setDataView("preview")}>{t("app:data.samplesTab")}</button><button role="tab" aria-selected={dataView==="charts"} className={dataView==="charts"?"active":""} onClick={()=>setDataView("charts")}>{t("app:data.chartsTab")}</button></div></div>

              {dataView==="charts"?<DataCharts analytics={analytics} task={task}/>:dataset.preview?.type === "image" && dataset.preview.items?.length ? (
                <div className="preview-image-grid">
                  {dataset.preview.items.slice(0,4).map((item, i) => (
                    <div className="preview-image-card" key={i}>
                      <div className={`dataset-image-pair ${item.targetUrl ? "with-target" : ""}`}><figure><img src={item.url} alt={item.label || t("app:data.sampleLabel", { number: i + 1 })} /><figcaption>{t("app:data.inputCaption")}</figcaption></figure>{item.targetUrl&&<figure><img src={item.targetUrl} alt={item.targetLabel || t("app:data.targetCaption")} /><figcaption>{item.targetLabel || t("app:data.outputCaption")}</figcaption></figure>}</div>
                      <span className="preview-image-label">{t("app:data.sampleLabel", { number: i + 1 })}</span>
                    </div>
                  ))}
                </div>
              ) : dataset.preview?.type === "text" && dataset.preview.items?.length ? (
                <div className="preview-table-wrap preview-text-wrap">
                  <table className="preview-table preview-text-table">
                    <thead><tr><th>{t("app:data.samplesTab")}</th></tr></thead>
                    <tbody>{dataset.preview.items.map((item, i) => <tr key={i}><td>{item.text}</td></tr>)}</tbody>
                  </table>
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
              <span className="eyebrow">{t("app:data.partitionsEyebrow")}</span>
              <SplitControl splits={dataset.splits} onChange={onSplitsChange} disabled={dataset.splitSource==="official"||dataset.splitSource==="temporal"} t={t} />
              {dataset.splitSource==="official"&&<small className="locked-node-note"><GitBranch/> {t("app:split.officialLocked")}</small>}
              {dataset.splitSource==="temporal"&&<small className="locked-node-note"><Clock3/> {t("app:split.temporalLocked")}</small>}
              <div className="split-legend">
                <span><i className="cyan" /> {t("common:train")} <strong>{trainCount}</strong></span>
                <span><i className="amber" /> {t("common:validation")} <strong>{validationCount}</strong></span>
                <span><i className="green" /> {t("common:test")} <strong>{testCount}</strong></span>
              </div>
            </article>
          </div>

          <aside className="data-sidebar"><div className="panel data-config-panel"><span className="eyebrow">{t("app:data.nodePropertiesEyebrow")}</span>{dataset.source==="huggingface"&&<div className="active-hf-source"><Database/><span><strong>{catalog.find(item=>item.id===dataset.catalogId)?.name||dataset.catalogId}</strong><small>{dataset.repoId}</small><small>{dataset.cacheStatus==="project"?t("app:data.loadedFromProject"):dataset.cacheStatus==="disk"?t("app:data.loadedFromCache"):t("app:data.downloadComplete")}</small></span></div>}<button className="secondary dataset-change-button" onClick={onChooseAnother}><Database/> {t("app:data.changeDataset")}</button><div id="data-node-inspector"/>{dataset.classes&&<div className="data-classes-section"><div className="data-classes-heading"><span className="eyebrow">{t("app:data.classesEyebrow", { count: dataset.classes.length })}</span><small>{t("common:samples")}: <strong>{dataset.samples.toLocaleString()}</strong></small></div><div className="class-tags">{dataset.classes.slice(0,20).map(c=><span key={c}>{c}</span>)}{dataset.classes.length>20&&<span className="class-tags-more" aria-label={`${dataset.classes.length-20} more classes`}>...</span>}</div></div>}</div></aside>
        </div>
      )}
    </div>
  );
}

function SplitControl({ splits, onChange, disabled=false, t }: { splits: DatasetSummary["splits"]; onChange: (value: DatasetSummary["splits"]) => void; disabled?: boolean; t: (key: string, opts?: Record<string, string | number>) => string }) {
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
        aria-label={t("app:split.trainValidationAria")}
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
        aria-label={t("app:split.validationTestAria")}
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
  onInfer: (payload: { mode: string; values?: string; index?: number; max_new_tokens?: number }) => Promise<Record<string, unknown>>;
}) {
  const { t } = useTranslation(["app", "common", "catalog"]);
  const taskName = t(`catalog:tasks.${task}.name`);
  const isImageTask = task.startsWith("image") || task.includes("segmentation");
  const isSegmentation = task.includes("segmentation");
  const isTabularTask = task.startsWith("tabular");
  const isTextTask = task.startsWith("text.");
  const isCausalText = task === "text.language_model";

  const [mode, setMode] = useState<"test" | "manual">("test");
  const [manualValues, setManualValues] = useState(isTabularTask ? Array.from({ length: dataset.inputShape[0] }, () => "0.5").join(", ") : "");
  const [manualImageB64, setManualImageB64] = useState<string>("");
  const [inputPreview, setInputPreview] = useState<string>("");
  const [trueLabel, setTrueLabel] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [maxNewTokens,setMaxNewTokens]=useState(16);

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
        values: mode === "manual" ? (isImageTask ? manualImageB64 : manualValues) : "",
        ...(isCausalText ? {max_new_tokens:maxNewTokens} : {})
      };
      const res = await onInfer(payload);
      setResult(res);
      if (res.inputPreview) setInputPreview(res.inputPreview as string);
      if (res.trueLabel !== undefined && res.trueLabel !== null) setTrueLabel(String(res.trueLabel));
      else if (mode === "manual") setTrueLabel(null);
    } catch (e) {
      setResult({ error: translateBackendError(String(e)) });
    } finally {
      setBusy(false);
    }
  };

  if (!run) return <Blocked message={t("app:inference.blockedMessage")} />;
  const prediction=result?.prediction;
  const comparisonAvailable=trueLabel!==null&&prediction!==undefined;
  const predictionCorrect=comparisonAvailable&&String(prediction).trim().toLowerCase()===String(trueLabel).trim().toLowerCase();
  const manualType=isImageTask ? "image" : isTextTask ? "text" : isTabularTask ? "tabular" : "sequence";
  const manualModeLabel=t(`app:inference.manualType.${manualType}`);
  const manualModeHelp=t(`app:inference.manualHelp.${manualType}`, { shape: dataset.inputShape.slice(manualType==="image"?1:0).join(" × "), length: dataset.inputShape[0], count: dataset.inputShape[0] });
  const changeMode=(next:"test"|"manual")=>{setMode(next);setResult(null);setTrueLabel(null);setInputPreview(next==="manual"?manualImageB64:"")};
  const outputCaption=isSegmentation ? t("app:inference.outputImageCaption.segmentation") : t("app:inference.outputImageCaption.default");

  return (
    <div className="page inference-page">
      <div className="page-intro">
        <div>
          <span className="eyebrow">{t("app:inference.stepEyebrow")}</span>
          <h2>{t("app:inference.title")}</h2>
          <p>{taskName} · {t("app:inference.checkpointOf", { runName })}</p>
        </div>
      </div>

      <article className={`panel inference-card ${isSegmentation?"segmentation-inference":""} ${comparisonAvailable?(predictionCorrect?"correct":"incorrect"):""}`}>
        <div className="inference-workbench-header">
          <div><span className="eyebrow">{t("common:inference")}</span><h3>{t("app:inference.chooseInput")}</h3></div>
          <span className="input-shape">{t("app:inference.inputShape", { shape: dataset.inputShape.join(" × ") })}</span>
        </div>

        <div className="inference-mode-selector" role="tablist" aria-label={t("app:inference.dataSourceAria")}>
          <button type="button" role="tab" aria-selected={mode === "test"} className={mode === "test" ? "active" : ""} onClick={() => changeMode("test")}>
            <Database size={15}/><span>{t("app:inference.mode.test")}</span><small>{t("app:inference.mode.testSmall")}</small>
          </button>
          <button type="button" role="tab" aria-selected={mode === "manual"} className={mode === "manual" ? "active" : ""} onClick={() => changeMode("manual")}>
            {isImageTask ? <Upload size={15}/> : <Activity size={15}/>}<span>{manualModeLabel}</span><small>{t("app:inference.mode.manualSmall")}</small>
          </button>
        </div>

        <div className={`inference-work-area ${isImageTask && inputPreview ? "has-image-preview" : ""}`}>
          <div className="inference-middle">
          {result ? <div className="inference-results-inline">
          <div className="inference-section-heading result-heading"><div><span className="eyebrow">{t("app:inference.resultEyebrow")}</span><h3>{result?.source ? String(result.source) : t("app:inference.predictionHeading")}</h3></div>{result&&<span className={`result-status ${comparisonAvailable?(predictionCorrect?"correct":"incorrect"):""}`}>{comparisonAvailable?(predictionCorrect?t("app:inference.status.correct"):t("app:inference.status.check")):t("app:inference.status.calculated")}</span>}</div>
          {result.error ? <div className="inference-error-state"><AlertCircle/><span><strong>{t("app:inference.cannotCalculate")}</strong><small>{String(result.error)}</small></span></div> : <>
              {comparisonAvailable&&<div className={`prediction-feedback ${predictionCorrect?"correct":"incorrect"}`}>{predictionCorrect?<CheckCircle2/>:<AlertCircle/>}<span><strong>{predictionCorrect?t("app:inference.predictionCorrect"):t("app:inference.predictionIncorrect")}</strong><small>{predictionCorrect?t("app:inference.matchesExpected"):t("app:inference.expectedLabel", { label: trueLabel })}</small></span></div>}
              <div className="prediction">
                <small>{t("app:inference.predictionLabel")}</small>
                <strong>{String(result.prediction ?? t("app:inference.noResult"))}</strong>
                {trueLabel && <span className="true-val-subtitle">{t("app:inference.trueLabel", { label: trueLabel })}</span>}
              </div>

              {isImageTask && inputPreview && (
                result.outputPreview ? (
                  <div className="inference-visual-pair">
                    <figure><figcaption>{t("app:inference.inputImageCaption")}</figcaption><img src={inputPreview} alt={t("app:inference.inputImageCaption")} /></figure>
                    <figure><figcaption>{outputCaption}</figcaption><img src={String(result.outputPreview)} alt={outputCaption} /></figure>
                  </div>
                ) : (
                  <figure className="inference-input-visual">
                    <figcaption>{t("app:inference.inputImageCaption")}</figcaption>
                    <img src={inputPreview} alt={t("app:inference.inputImageCaption")} />
                  </figure>
                )
              )}

              {Array.isArray(result.probabilities) && (
                <div className="probabilities">
                  {result.probabilities.map((p, i) => (
                    <div key={i}>
                      <span>{dataset.classes?.[i] || t("app:inference.classFallback", { index: i })}</span>
                      <div>
                        <i style={{ width: `${Number(p) * 100}%` }} />
                      </div>
                      <strong>{(Number(p) * 100).toFixed(1)}%</strong>
                    </div>
                  ))}
                </div>
              )}
            </>}
        </div> : mode === "test" ? <div className="inference-preparation"><div className="source-copy"><Database/><span><strong>{t("app:inference.testSourceTitle")}</strong><small>{t("app:inference.testSourceHint")}</small></span></div><div className="awaiting-result"><BarChart3/><span>{t("app:inference.awaitingTitle")}</span></div></div> : <div className="inference-preparation"><div className="source-copy"><Activity/><span><strong>{manualModeLabel}</strong><small>{manualModeHelp}</small></span></div>{isImageTask?<div className="image-upload-box"><label className="input-dropzone"><Upload size={24}/><span>{t("app:inference.uploadHint")}</span><small>{t("app:inference.uploadFormats")}</small><input type="file" accept="image/*" onChange={handleFileUpload}/></label>{inputPreview&&<div className="image-preview-container"><img src={inputPreview} alt={manualModeLabel}/></div>}</div>:<><label className="manual-values-field"><span>{manualModeHelp}</span><textarea value={manualValues} onChange={e=>setManualValues(e.target.value)} placeholder={isTextTask?t("app:inference.textPlaceholder"):t("app:inference.tabularPlaceholder")}/></label>{isCausalText&&<label className="app-field"><span>{t("app:inference.maxNewTokens")}</span><input type="number" min="1" max="128" value={maxNewTokens} onChange={event=>setMaxNewTokens(Math.max(1,Math.min(128,Number(event.target.value)||1)))}/></label>}</>}</div>}
          </div>
          <button className="primary inference-run-button" onClick={execute} disabled={busy || (mode === "manual" && isImageTask && !manualImageB64)}><Play size={16}/> {busy ? t("app:inference.runButton.calculating") : mode === "test" ? t("app:inference.runButton.test") : t("app:inference.runButton.manual")}</button>
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
  const text = task.startsWith("text.");
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
  } else if (text) {
    const samples = [
      "El modelo aprende patrones a partir de ejemplos de texto.",
      "Los datos bien preparados mejoran cada predicción.",
      "Una secuencia conserva el contexto de sus palabras.",
      "La evaluación confirma si el modelo generaliza.",
      "Los tokens representan fragmentos de lenguaje natural.",
      "El entrenamiento ajusta los parámetros gradualmente."
    ];
    preview = { type: "text", items: samples.map(value => ({ text: value })) };
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
