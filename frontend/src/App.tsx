import { KaggleImport, KaggleReport } from "./KaggleImport";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  ReactFlow, Background, Controls, MiniMap, ViewportPortal, addEdge, useEdgesState, useNodesState,
  type Connection, type NodeMouseHandler, type ReactFlowInstance
} from "@xyflow/react";
import { Copy, Group, Ungroup, Activity, AlertCircle, ArrowDown, ArrowLeft, ArrowRight, ArrowUpDown, BarChart3, Binary, Box, BrainCircuit, Check, CheckCircle2, ChevronRight, Clock3, Cloud, Cpu, Database, Dice5, Download, FileText, FolderOpen, GitBranch, GitFork, Grid2X2, Image, LayoutGrid, Layers3, List, Network, Info, Moon, Palette, Pause, Play, Plus, RefreshCw, Repeat2, ScanLine, Search, Shield, Shrink, Square, SlidersHorizontal, Save, Sparkles, Split, Sun, Table2, Upload, WandSparkles, Waves, Waypoints, X, Zap } from "lucide-react";
import { ARCHITECTURES, BLOCK_INFO, BLOCKS, TASKS, pretrainedTemplate, presetsFor, syncOutputContract, templateFor } from "./catalog";
import { getPresetBenefit, getPresetDescription, getPresetName, getPresetTradeoff } from "./catalog-i18n";
import { backend, isTauri, onTrainingEvent, pickDatasetDirectory, pickDatasetFile, startTraining } from "./bridge";
import ModelNode from "./ModelNode";
import { operationIcon } from "./operation-icons";
import { TensorShapeSummary } from "./tensor-shape";
import { previewOperationOutput } from "./operation-preview";
import { TrainingChart } from "./Chart";
import DataPipelineEditor, { defaultDataPipeline } from "./DataPipelineEditor";
import DataCharts from "./DataCharts";
import { extractCnnBackbone, featurePorts } from "./cnn-reuse";
import { candidateEdgesForConnection, createCnnRecipe, duplicateGraphSelection, isValidGraphConnection, layoutConnectedGraph, type GraphLayoutDirection } from "./builder-graph";
import "./canvas-actions.css";
import { translateBackendError } from "./i18n/errors";
import type { Architecture, DataPipeline, DatasetAnalytics, DatasetOptions, DatasetPreview, DatasetSummary, GpuInfo, GraphValidation, HuggingFaceDataset, MetricPoint, ModelEdge, ModelNode as ModelNodeT, PretrainedModelSpec, Project, ProjectModel, RunResult, RuntimeCatalog, Step, TaskId, TrainingRun } from "./types";

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
const initialGraphNodes = (nodes: ModelNodeT[]) => [...nodes].sort((a,b)=>a.position.x-b.position.x).slice(0,3).map(({id})=>({id}));
const initialGraphFit = (nodes: ModelNodeT[]) => ({nodes:initialGraphNodes(nodes),...(window.innerWidth<=1280?{padding:0,maxZoom:1}:{padding:.3,maxZoom:.85})});
const inferLayoutDirection = (nodes:ModelNodeT[],edges:ModelEdge[]):GraphLayoutDirection => {
  const byId=new Map(nodes.map(node=>[node.id,node]));let horizontal=0,vertical=0;
  for(const edge of edges){const source=byId.get(edge.source),target=byId.get(edge.target);if(!source||!target)continue;const dx=Math.abs(target.position.x-source.position.x),dy=Math.abs(target.position.y-source.position.y);if(dx+dy<1)continue;if(dx>=dy)horizontal++;else vertical++;}
  return vertical>horizontal?"vertical":"horizontal";
};

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
function parseCsvHeader(header: string): string[] {
  const candidates = [",", "\t", ";", "|"];
  const delimiter = candidates.reduce((best, value) => (header.split(value).length > header.split(best).length ? value : best), ",");
  const columns: string[] = [];
  let value = "", quoted = false;
  for (let index = 0; index < header.length; index++) {
    const char = header[index];
    if (char === '"') {
      if (quoted && header[index + 1] === '"') { value += '"'; index++; }
      else quoted = !quoted;
    } else if (char === delimiter && !quoted) {
      columns.push(value.trim()); value = "";
    } else value += char;
  }
  columns.push(value.trim());
  return columns.filter(Boolean);
}
const newTrainingRun = (index: number, name: string): TrainingRun => ({ id: uid(), name, status: "draft", config: {}, history: [], result: null, createdAt: now() });
const newProjectModel = (index: number, architecture: Architecture, task: TaskId, name: string): ProjectModel => {
  const graph = templateFor(architecture, task);
  return { id: uid(), name, architecture, nodes: graph.nodes, edges: graph.edges, graphValid: false };
};
const loadProjects = (): Project[] => {
  try {
    const value = JSON.parse(localStorage.getItem("mb-projects") || "[]");
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
};
type Theme = "dark" | "light";
type Accent = "blue" | "orange" | "green" | "violet";
type AppDialogState = {
  kind: "alert" | "confirm" | "prompt";
  title: string;
  message?: string;
  inputLabel?: string;
  initialValue?: string;
  confirmLabel?: string;
  onConfirm?: (value?: string) => void;
};
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
  if(!outputs.length) inputs.forEach(node=>errors.set(node.id,t("app:graphError.singleOutput")));
  const outputNames=outputs.map(node=>String(node.data.properties.output_name||node.data.label||node.id).trim());
  if(new Set(outputNames).size!==outputNames.length)outputs.forEach(node=>errors.set(node.id,t("app:graphError.duplicateOutputName")));
  const seen=new Set<string>();
  edges.forEach(edge=>{const key=`${edge.source}:${edge.sourceHandle||""}->${edge.target}:${edge.targetHandle||""}`;if(seen.has(key)){errors.set(edge.source,t("app:graphError.duplicateConnection"));errors.set(edge.target,t("app:graphError.duplicateConnection"))}seen.add(key);if(!byId.has(edge.source)||!byId.has(edge.target))return;if(byId.get(edge.source)?.data.blockType==="output")errors.set(edge.source,t("app:graphError.outputCannotSource"));if(byId.get(edge.target)?.data.blockType==="input")errors.set(edge.target,t("app:graphError.inputCannotTarget"))});
  nodes.forEach(node=>{const incoming=edges.filter(edge=>edge.target===node.id);const outgoing=edges.filter(edge=>edge.source===node.id);if(node.data.blockType!=="input"&&!incoming.length)errors.set(node.id,t("app:graphError.blockNoInput"));if(node.data.blockType!=="output"&&!outgoing.length)errors.set(node.id,t("app:graphError.blockNoOutput"));if(!["concat","add"].includes(node.data.blockType)&&incoming.length>1)errors.set(node.id,t("app:graphError.singleInputOnly"));if(["concat","add"].includes(node.data.blockType)&&incoming.length<2)errors.set(node.id,t("app:graphError.needsTwoInputs"))});
  const indegree=new Map(nodes.map(node=>[node.id,0]));edges.forEach(edge=>indegree.set(edge.target,(indegree.get(edge.target)||0)+1));const queue=[...indegree].filter(([,degree])=>degree===0).map(([id])=>id);let visited=0;while(queue.length){const id=queue.shift()!;visited++;edges.filter(edge=>edge.source===id).forEach(edge=>{const degree=(indegree.get(edge.target)||1)-1;indegree.set(edge.target,degree);if(degree===0)queue.push(edge.target)})}if(visited!==nodes.length)[...indegree].filter(([,degree])=>degree>0).forEach(([id])=>errors.set(id,t("app:graphError.cycle")));
  return errors;
}

function graphValidationKey(project:Project|null,dataset:DatasetSummary|null,task:TaskId|null,architecture:Architecture|null,nodes:ModelNodeT[],edges:ModelEdge[]):string{
  return JSON.stringify({
    project:project?{id:project.id,path:project.path}:null,
    task,
    architecture,
    dataset:dataset?{id:dataset.id,revision:dataset.revision,inputShape:dataset.inputShape,outputShape:dataset.outputShape,classes:dataset.classes,options:dataset.options}:null,
    // Preserve array order because merge inputs and graph evaluation order are semantic. Layout, selection, and group metadata are not.
    nodes:nodes.map(node=>({id:node.id,type:node.type,blockType:node.data.blockType,label:node.data.label,properties:node.data.properties,inputHandles:node.data.inputHandles,outputHandles:node.data.outputHandles})),
    edges:edges.map(edge=>({source:edge.source,target:edge.target,sourceHandle:edge.sourceHandle,targetHandle:edge.targetHandle}))
  });
}

export default function App() {
  const [projects, setProjects] = useState<Project[]>(loadProjects);
  const [project, setProject] = useState<Project | null>(null);
  const [step, setStep] = useState<Step>("model");
  useEffect(()=>{document.querySelector(".content")?.scrollTo(0,0)},[step]);
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
  const [validationStats, setValidationStats] = useState<GraphValidation | null>(null);
  const [validationMessage,setValidationMessage]=useState("");
  const [training, setTraining] = useState(false);
  const runningRunId = useRef<string | null>(null);
  const backendRunId = useRef<string | null>(null);
  const pendingStop=useRef(false);
  const stoppingRef=useRef(false);
  const projectRef=useRef(project);
  const [stopping,setStopping]=useState(false);
  const [theme, setTheme] = useState<Theme>(loadTheme);
  const [accent, setAccent] = useState<Accent>(loadAccent);
  const [notice, setNotice] = useState<string>("");
  const [dialog, setDialog] = useState<AppDialogState | null>(null);
  const [gpuInfo, setGpuInfo] = useState<GpuInfo | null>(null);
  const [runtimeCatalog,setRuntimeCatalog]=useState<RuntimeCatalog|null>(null);
  const dataContinue = useRef<() => Promise<boolean>>(async () => true);

  useEffect(() => {
    backend<GpuInfo>("system.gpu").then(setGpuInfo).catch(() => setGpuInfo({cuda:false,name:"CPU",total_vram:0,allocated_vram:0}));
    backend<RuntimeCatalog>("catalog.get").then(setRuntimeCatalog).catch(()=>setRuntimeCatalog(null));
    backend<Project[]>("project.list").then(saved => {
      setProjects(current => {
        const merged = new Map(current.map(item => [item.id, item]));
        saved.forEach(item => merged.set(item.id, { ...merged.get(item.id), ...item }));
        return [...merged.values()];
      });
    }).catch(()=>{});
  }, []);


  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem("mb-theme", theme); }, [theme]);
  useEffect(() => { document.documentElement.dataset.accent = accent; localStorage.setItem("mb-accent", accent); }, [accent]);
  useEffect(()=>{if(!notice)return;const timer=window.setTimeout(()=>setNotice(""),5000);return()=>window.clearTimeout(timer)},[notice]);
  useEffect(() => { localStorage.setItem("mb-projects", JSON.stringify(projects)); }, [projects]);
  useEffect(() => {
    if (!project) return;
    const state={ step, task, architecture, dataset, nodes, edges, models, activeModelId, trainingRuns, activeRunId, graphValid };
    localStorage.setItem(`mb-state-${project.id}`, JSON.stringify(state));
    const timer=window.setTimeout(()=>backend("project.state.save",{project,state}).catch(()=>{}),600);
    return ()=>window.clearTimeout(timer);
  }, [project, step, task, architecture, dataset, nodes, edges, models, activeModelId, trainingRuns, activeRunId, graphValid]);
  useEffect(() => {
    if (!activeModelId || !architecture) return;
    setModels(items => items.map(model => model.id === activeModelId ? { ...model, architecture, nodes, edges, graphValid, semanticHash: graphValid ? validationStats?.semantic_hash : undefined } : model));
  }, [activeModelId, architecture, nodes, edges, graphValid, validationStats]);
  useEffect(() => { if (!graphValid) setValidationStats(null); }, [graphValid]);
  useEffect(() => { let off = () => {}; onTrainingEvent(event => {
    const targetId = runningRunId.current;
    if (!targetId) return;
    if (event.type === "started") { backendRunId.current=String(event.run_id);setNotice("");if(pendingStop.current && projectRef.current){pendingStop.current=false;backend("training.control",{project:projectRef.current,run_id:backendRunId.current,state:"cancelled"}).catch(error=>{stoppingRef.current=false;setStopping(false);setNotice(translateBackendError(String(error)))})} }
    if(event.run_id&&backendRunId.current&&String(event.run_id)!==backendRunId.current)return;
    if (event.type === "paused") setTrainingRuns(runs=>runs.map(item=>item.id===targetId?{...item,status:"paused"}:item));
    if (event.type === "resumed") setTrainingRuns(runs=>runs.map(item=>item.id===targetId?{...item,status:"running"}:item));
    if (event.type === "metric") setTrainingRuns(runs => runs.map(item => item.id === targetId ? { ...item, history: [...item.history.filter(p => p.epoch !== Number(event.epoch)), { epoch: Number(event.epoch), trainLoss: Number(event.train_loss), valLoss: Number(event.val_loss), metric: Number(event.metric), trainMetric:event.train_metric == null ? undefined : Number(event.train_metric), valMetric:event.val_metric == null ? undefined : Number(event.val_metric), learningRate:Number(event.learning_rate), secondaryMetrics:(event.secondary_metrics||{}) as Record<string,number> }].sort((a,b) => a.epoch-b.epoch) } : item));
    if (event.type === "complete") { const result=event.result as unknown as RunResult; setTraining(false); stoppingRef.current=false;setStopping(false);pendingStop.current=false; setTrainingRuns(runs => runs.map(item => item.id === targetId ? { ...item, status:result.status==="cancelled"?"cancelled":"completed", result, history:result.history || item.history, completedAt:now() } : item)); runningRunId.current=null;backendRunId.current=null; setNotice(result.status==="cancelled"?t("app:training.cancelled"):t("app:notice.trainingComplete")); }
    if (event.type === "error") { setTraining(false); stoppingRef.current=false;setStopping(false);pendingStop.current=false; setTrainingRuns(runs => runs.map(item => item.id === targetId ? { ...item, status:"error", error:String(event.message), completedAt:now() } : item)); runningRunId.current=null;backendRunId.current=null; setNotice(translateBackendError(String(event.message))); }
  }).then(fn => off = fn); return () => off(); }, []);

  const activeTrainingRun = trainingRuns.find(item => item.id === activeRunId) || trainingRuns[0];
  const history = activeTrainingRun?.history || [];
  const run = activeTrainingRun?.result || null;
  const completed = useMemo(() => ({ model: !!task && !!architecture, data: !!dataset, builder: graphValid, training: trainingRuns.some(item=>!!item.result), inference: false }), [task, architecture, dataset, graphValid, trainingRuns]);
  const selected = nodes.find(n => n.id === selectedId) || null;
  const validationKey = graphValidationKey(project,dataset,task,architecture,nodes,edges);
  const validationKeyRef = useRef(validationKey);
  validationKeyRef.current = validationKey;
  const previousValidationKey = useRef(validationKey);
  const validationRequest = useRef(0);
  const autoValidationTimer = useRef<number | null>(null);

  const createProject = async (name: string, description: string) => {
    const p: Project = { id: uid(), name, description, path: `projects/${name.toLowerCase().replace(/[^a-z0-9áéíóúñ]+/gi, "-")}`, createdAt: now(), updatedAt: now() };
    try { await backend("project.create", { project: p }); } catch (e) { setNotice(translateBackendError(String(e))); return; }
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
    if (!isTauri()) { setDialog({kind:"alert",title:t("app:openFolder.title"),message:t("app:openFolder.notTauri")}); return; }
    setDialog({kind:"prompt",title:t("app:openFolder.title"),inputLabel:t("app:openFolder.prompt"),onConfirm:value=>{
      const path=value?.trim();if(!path)return;
      void backend<Project>("project.open",{path}).then(p=>{setProjects(v=>[p,...v.filter(item=>item.id!==p.id)]);openProject(p)}).catch(e=>setDialog({kind:"alert",title:t("app:dialog.errorTitle"),message:translateBackendError(String(e))}));
    }});
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

  const applyTaskChoice = (a: Architecture, taskId: TaskId) => {
    const taskChanged = Boolean(task && task !== taskId);
    const template = templateFor(a, taskId);
    const nextNodes = !taskChanged && dataset ? syncOutputContract(template.nodes, template.edges, dataset) : template.nodes;
    const freshRun=newTrainingRun(taskChanged ? 1 : trainingRuns.length + 1, t("app:trainingRun.defaultName", { index:taskChanged ? 1 : trainingRuns.length + 1 }));
    const model=newProjectModel(taskChanged ? 1 : models.length + 1,a,taskId,t("app:projectModel.defaultName", { index:taskChanged ? 1 : models.length + 1 }));
    model.nodes=nextNodes; model.edges=template.edges;
    setArchitecture(a); setTask(taskId); setDataset(taskChanged ? null : dataset); setTrainingRuns(taskChanged ? [freshRun] : [...trainingRuns, freshRun]); setActiveRunId(freshRun.id); setGraphValid(false); setNodes(nextNodes); setEdges(template.edges); setSelectedId(null); setActiveModelId(model.id);
    if (taskChanged) setModels([model]);
    else setModels(items=>[...items.map(item=>item.id===activeModelId?{...item,architecture:architecture || item.architecture,nodes,edges,graphValid}:item),model]);
    if (project) { const updated = { ...project, architecture: a, taskId, updatedAt: now() }; setProject(updated); setProjects(v => v.map(p => p.id === updated.id ? updated : p)); }
  };

  const choose = (a: Architecture, taskId: TaskId) => {
    if (architecture === a && task === taskId) return;
    const taskChanged = Boolean(task && task !== taskId);
    if (taskChanged && (dataset || nodes.length > 0 || trainingRuns.some(item => item.result))) {
      setDialog({kind:"confirm",title:t("app:notice.changeTaskTitle"),message:t("app:notice.changeTaskWarning"),onConfirm:()=>applyTaskChoice(a,taskId)});
      return;
    }
    applyTaskChoice(a,taskId);
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

  const importKaggleData = async (handle: string, options: DatasetOptions) => {
    if (!task || !architecture || !project) throw new Error(t("app:data.noActiveProject"));
    const result = await backend<DatasetSummary>("data.kaggle.import", { project, task_id: task, handle, options });
    setDataset(result); setNodes(current => syncOutputContract(current, edges, result)); setGraphValid(false);
    setNotice(t("app:data.datasetReady", { samples: result.samples }));
  };

  const importData = async (path: string, options?: DatasetOptions) => {
    if (!task || !architecture || !project || !path) return;
    try { const result = await backend<DatasetSummary>("data.import", { project, task_id: task, architecture, path, options }); setDataset({...result,sourcePath:path}); setNodes(current => syncOutputContract(current, edges, result)); setGraphValid(false); setNotice(t("app:data.importedNotice")); }
    catch (e) { setNotice(translateBackendError(String(e))); }
  };

  const importDataFiles = async (files: File[], options?: DatasetOptions) => {
    if (!task || !architecture || !project || !files.length) return;
    try {
      const encoded = await Promise.all(files.map(async file => {
        const bytes = new Uint8Array(await file.arrayBuffer());
        let binary = "";
        for (let offset = 0; offset < bytes.length; offset += 0x8000) {
          binary += String.fromCharCode(...bytes.subarray(offset, Math.min(offset + 0x8000, bytes.length)));
        }
        return { name: file.name, relative_path: (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name, content_base64: btoa(binary) };
      }));
      const result = await backend<DatasetSummary>("data.import.files", { project, task_id: task, architecture, files: encoded, options });
      setDataset(result); setNodes(current => syncOutputContract(current, edges, result)); setGraphValid(false); setNotice(t("app:data.importedNotice"));
    } catch (error) {
      setNotice(translateBackendError(String(error)));
    }
  };

  const applyDataPipeline = async (pipeline: DataPipeline) => {
    if (!project || !dataset || !task) throw new Error(t("app:data.noActiveDataset"));
    try {
      const result=await backend<DatasetSummary>("data.pipeline.apply",{project,dataset,task_id:task,pipeline});
      setDataset({...result,splitPending:false});setNodes(current=>syncOutputContract(current,edges,result));setGraphValid(false);setNotice(t("app:data.pipelineApplied", { revision: result.pipeline?.revision ?? result.revision }));
    } catch(e) { setNotice(translateBackendError(String(e))); throw e; }
  };

  const validateGraph = async (source:"manual"|"automatic"="manual", expectedKey?:string) => {
    if(source==="manual"&&autoValidationTimer.current!==null){window.clearTimeout(autoValidationTimer.current);autoValidationTimer.current=null}
    const requestKey=expectedKey||validationKey;
    if(requestKey!==validationKeyRef.current)return false;
    const requestId=++validationRequest.current;
    const isCurrent=()=>requestId===validationRequest.current&&requestKey===validationKeyRef.current;
    if(isCurrent())setValidationMessage("");
    if (!project || !dataset || !task || !architecture) {
      if(isCurrent()){setGraphValid(false);setValidationStats(null);if(source==="manual")setNotice(t("app:notice.completeProjectData"))}
      return false;
    }
    const contractNodes = syncOutputContract(nodes, edges, dataset);
    if (contractNodes !== nodes) setNodes(contractNodes);
    const graphNodes = contractNodes;
    const errors=graphErrorMap(graphNodes,edges,t);
    if(!isCurrent())return false;
    setNodes(items=>items.map(node=>({...node,data:{...node.data,error:errors.get(node.id)}})));
    if (errors.size) {
      const [nodeId,message]=errors.entries().next().value as [string,string];
      setValidationStats(null);setGraphValid(false);setValidationMessage(message);
      if(source==="manual"){setSelectedId(nodeId);setNotice(`${nodes.find(node=>node.id===nodeId)?.data.label || t("app:builder.graph")}: ${message}`)}
      return false;
    }
    {
      try {
        const r = await backend<GraphValidation>("graph.validate", { project, dataset, task_id: task, architecture, graph: { nodes: graphNodes, edges } });
        if(!isCurrent())return false;
        if (r.node_shapes) setNodes(ns => ns.map(n => {const value=r.node_shapes?.[n.id];const shape=Array.isArray(value)?`[B, ${value.join(", ")}]`:value?Object.entries(value).map(([port,dims])=>`${port} [B, ${dims.join(", ")}]`).join(" · "):"?";return {...n,data:{...n.data,shape}}}));
        setValidationStats(r.valid ? r : null); setGraphValid(r.valid);setValidationMessage(r.valid?"":r.message); if(source==="manual")setNotice(r.message); return r.valid;
      } catch (e) {
        if(!isCurrent())return false;
        const message=translateBackendError(String(e));setValidationMessage(message);const failed=nodes.find(node=>message.includes(node.data.label)||message.includes(node.id));if(failed){setNodes(items=>items.map(node=>node.id===failed.id?{...node,data:{...node.data,error:message}}:node));if(source==="manual")setSelectedId(failed.id)}setValidationStats(null);setGraphValid(false);if(source==="manual")setNotice(message);return false;
      }
    }
    setGraphValid(false); return false;
  };

  const validateGraphRef=useRef(validateGraph);
  validateGraphRef.current=validateGraph;
  useEffect(()=>{
    if(previousValidationKey.current===validationKey)return;
    previousValidationKey.current=validationKey;
    validationRequest.current++;
    setGraphValid(false);setValidationStats(null);setValidationMessage("");
    setNodes(items=>items.map(node=>({...node,data:{...node.data,shape:undefined,error:undefined}})));
  },[validationKey,setNodes]);
  useEffect(()=>{
    if(autoValidationTimer.current!==null){window.clearTimeout(autoValidationTimer.current);autoValidationTimer.current=null}
    if(step!=="builder"||!project||!dataset||!task||!architecture)return;
    const scheduledKey=validationKey;
    autoValidationTimer.current=window.setTimeout(()=>{
      autoValidationTimer.current=null;
      if(validationKeyRef.current===scheduledKey)void validateGraphRef.current("automatic",scheduledKey);
    },450);
    return()=>{if(autoValidationTimer.current!==null){window.clearTimeout(autoValidationTimer.current);autoValidationTimer.current=null}};
  },[validationKey,step]);

  const saveModelRevision = async () => {
    if (!(await validateGraph()) || !project || !activeModelId) return;
    const model = models.find(item => item.id === activeModelId);
    if (!model) return;
    try {
      const saved = await backend<{ revision: number }>("model.revision.save", {
        project,
        model: { ...model, architecture, nodes, edges, graphValid: true, semanticHash: validationStats?.semantic_hash }
      });
      setNotice(t("app:builder.revisionSaved", { revision: saved.revision }));
    } catch (error) {
      setNotice(translateBackendError(String(error)));
    }
  };

  const train = async (config: Record<string, number | string | boolean>) => {
    if (!project || !dataset || !task || !architecture || !(await validateGraph())) return;
    const makeNewRun=activeTrainingRun?.status==="completed"||!activeTrainingRun;
    const targetRun=makeNewRun
      ? newTrainingRun(trainingRuns.length+1, t("app:trainingRun.defaultName", { index: trainingRuns.length + 1 }))
      : activeTrainingRun;
    const targetId=targetRun.id;
    runningRunId.current=targetId;backendRunId.current=null;pendingStop.current=false;stoppingRef.current=false;setStopping(false);projectRef.current=project;
    const trainedGraph={nodes:syncOutputContract(nodes,edges,dataset),edges};
    const nextRun={...targetRun,modelId:activeModelId || undefined,modelTask:task,modelArchitecture:architecture,modelInputShape:[...(dataset.inputShape||[])],modelGraph:trainedGraph,status:"running" as const,config,history:[],result:null,error:undefined,startedAt:now(),completedAt:undefined};
    setTraining(true);setActiveRunId(targetId);setTrainingRuns(runs=>runs.some(item=>item.id===targetId)?runs.map(item=>item.id===targetId?nextRun:item):[...runs,nextRun]);setNotice(t("app:training.preparing"));
    try {
      await startTraining({ project, dataset, task_id: task, architecture, model_spec:models.find(model=>model.id===activeModelId), graph: { nodes: syncOutputContract(nodes, edges, dataset), edges }, config });
    } catch (e) {
      setTraining(false); runningRunId.current=null;backendRunId.current=null;pendingStop.current=false;stoppingRef.current=false;setStopping(false); setTrainingRuns(runs=>runs.map(item=>item.id===targetId?{...item,status:"error",error:translateBackendError(String(e)),completedAt:now()}:item));setNotice(translateBackendError(String(e)));
    }
  };
  const controlTraining=async(state:"running"|"paused"|"cancelled")=>{if(!projectRef.current)return;if(state==="cancelled"){if(stoppingRef.current)return;stoppingRef.current=true;setStopping(true);if(!backendRunId.current){pendingStop.current=true;return}}if(!backendRunId.current)return;try{await backend("training.control",{project:projectRef.current,run_id:backendRunId.current,state})}catch(error){if(state==="cancelled")stoppingRef.current=false;setStopping(false);setNotice(translateBackendError(String(error)))}};

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
  const closeModel=(id:string)=>{const i=models.findIndex(m=>m.id===id);if(models.length<=1||i<0||trainingRuns.some(r=>r.id===runningRunId.current&&r.modelId===id))return;if(id===activeModelId)switchModel(models[i===models.length-1?i-1:i+1].id);setModels(items=>items.filter(m=>m.id!==id))};
  const closeRun=(id:string)=>{const i=trainingRuns.findIndex(r=>r.id===id);if(trainingRuns.length<=1||i<0||id===runningRunId.current)return;if(id===activeRunId)setActiveRunId(trainingRuns[i===trainingRuns.length-1?i-1:i+1].id);setTrainingRuns(items=>items.filter(r=>r.id!==id))};
  const addModel = () => {
    if(!architecture || !task)return; const defaultName=t("app:projectModel.defaultName", { index: models.length + 1 });
    setDialog({kind:"prompt",title:t("app:model.createTitle"),inputLabel:t("app:model.namePrompt"),initialValue:defaultName,onConfirm:value=>{
      const name=value?.trim();if(!name||!architecture||!task)return;
      const model=newProjectModel(models.length+1,architecture,task,defaultName);model.name=name;model.nodes=syncOutputContract(model.nodes,model.edges,dataset);setModels(items=>[...items,model]);switchModelAfterCreate(model);
    }});
  };
  const switchModelAfterCreate=(model:ProjectModel)=>{setActiveModelId(model.id);setArchitecture(model.architecture);setNodes(model.nodes);setEdges(model.edges);setGraphValid(false);setSelectedId(null)};
  const createCheckpointBackbone=(payload:{name:string;nodes:ModelNodeT[];edges:ModelEdge[];checkpoint:string;sourceRunId:string;sourceModelId:string;sourceNodeId:string;sourceHandle?:string})=>{
    if(!architecture||!task)return;
    const model:ProjectModel={id:uid(),name:payload.name,architecture,nodes:payload.nodes,edges:payload.edges,graphValid:false,source:"checkpoint",checkpointLineage:{checkpoint:payload.checkpoint,parentRunId:payload.sourceRunId,parentModelId:payload.sourceModelId,sourceNodeId:payload.sourceNodeId,sourceHandle:payload.sourceHandle}};
    const fresh=newTrainingRun(trainingRuns.length+1,t("app:trainingRun.defaultName", { index: trainingRuns.length+1 }));fresh.modelId=model.id;fresh.config={initial_checkpoint:payload.checkpoint};
    setModels(items=>[...items,model]);setTrainingRuns(items=>[...items,fresh]);setActiveRunId(fresh.id);switchModelAfterCreate(model);setStep("builder");
  };

  const stepIndex=steps.findIndex(item=>item.id===step);
  const goPrevious=()=>{if(stepIndex>0)setStep(steps[stepIndex-1].id)};
  const goNext=async()=>{
    if(step==="model"){if(!task||!architecture){setNotice(t("app:notice.selectTaskAndArchitecture"));return}setStep("data");return}
    if(step==="data"){if(!dataset){setNotice(t("app:notice.loadDataset"));return}if(!(await dataContinue.current()))return;setStep("builder");return}
    if(step==="builder"){if(await validateGraph())setStep("training");return}
    if(step==="training"){if(!run){setNotice(t("app:notice.completeRun"));return}setStep("inference")}
  };

  if (!project) return <ProjectHub projects={projects} onCreate={createProject} onOpen={openProject} onDelete={deleteProject} theme={theme} setTheme={setTheme} accent={accent} setAccent={setAccent} />;

  return <div className="app-shell">
    <Sidebar step={step} setStep={setStep} completed={completed} onHome={() => setProject(null)} gpuInfo={gpuInfo} nodes={nodes} dataset={dataset} theme={theme} setTheme={setTheme} accent={accent} setAccent={setAccent} />
    <main className="main-area">
      {notice && <div className="notice"><span>{notice}</span><button onClick={() => setNotice("")}><X size={14}/></button></div>}
      <section className={`content ${step === "training" ? "no-scroll" : ""}`}>

        {step === "model" && <ModelTask architecture={architecture} task={task} runtimeCatalog={runtimeCatalog} onChoose={choose} onContinue={goNext} />}
        {step === "data" && <DataStep project={project} task={task} dataset={dataset} dataContinue={dataContinue} setNotice={setNotice} onDownload={downloadData} onKaggle={importKaggleData} onImport={importData} onImportFiles={importDataFiles} onChooseAnother={()=>setDataset(null)} onApplyPipeline={applyDataPipeline} onSplitsChange={splits => setDataset(current => current ? { ...current, splits, splitCounts:undefined, splitPending:true } : current)} />}
        {step === "builder" && architecture && task && <Builder architecture={architecture} task={task} dataset={dataset} validationStats={validationStats} validationMessage={validationMessage} models={models} trainingRuns={trainingRuns} activeModelId={activeModelId} onSelectModel={switchModel} onCloseModel={closeModel} onAddModel={addModel} onReuseCheckpoint={createCheckpointBackbone} onApplyPretrained={spec=>setModels(items=>items.map(model=>model.id===activeModelId?{...model,source:"huggingface",pretrained:spec,schemaVersion:2,graphValid:false}:model))} theme={theme} nodes={nodes} edges={edges} setNodes={setNodes} setEdges={setEdges} onNodesChange={(changes: Parameters<typeof onNodesChange>[0]) => { if(changes.some(change=>["remove","add","replace"].includes(change.type))){setGraphValid(false);setNodes(items=>items.map(node=>({...node,data:{...node.data,shape:undefined,error:undefined}})))} onNodesChange(changes); }} onEdgesChange={(changes: Parameters<typeof onEdgesChange>[0]) => { if(changes.some(change=>["remove","add","replace"].includes(change.type))){setGraphValid(false);setNodes(items=>items.map(node=>({...node,data:{...node.data,shape:undefined,error:undefined}})))} onEdgesChange(changes); }} onDirty={()=>{setGraphValid(false);setValidationStats(null);setNodes(items=>items.map(node=>({...node,data:{...node.data,shape:undefined,error:undefined}})))}} selected={selected} selectedId={selectedId} setSelectedId={setSelectedId} onValidate={validateGraph} onSaveRevision={saveModelRevision} />}
        {step === "training" && task && <Training task={task} dataset={dataset} models={models} activeModelId={activeModelId} onSelectModel={switchModel} runs={trainingRuns} activeRunId={activeRunId} training={training} stopping={stopping} accent={accent} onSelectRun={setActiveRunId} onCloseRun={closeRun} onAddRun={addTrainingRun} onTrain={train} onControl={controlTraining} />}
        {step === "inference" && task && dataset && <Inference task={task} dataset={dataset} run={run} runName={activeTrainingRun?.name || t("app:inference.activeRun")} onInfer={infer} />}
      </section>
    </main>
    <AppDialog dialog={dialog} onClose={()=>setDialog(null)}/>
  </div>;
}

function AppDialog({dialog,onClose}:{dialog:AppDialogState|null;onClose:()=>void}){
  const {t}=useTranslation(["app","common"]);const [value,setValue]=useState(dialog?.initialValue||"");
  useEffect(()=>{setValue(dialog?.initialValue||"")},[dialog]);
  if(!dialog)return null;
  const submit=(inputValue?:string)=>{onClose();dialog.onConfirm?.(inputValue)};
  return <div className="modal-backdrop app-dialog-backdrop" onMouseDown={event=>{if(event.target===event.currentTarget)onClose()}}><section className="modal app-dialog" role="dialog" aria-modal="true" aria-labelledby="app-dialog-title" aria-describedby={dialog.message?"app-dialog-message":undefined} onMouseDown={event=>event.stopPropagation()} onKeyDown={event=>{if(event.key==="Escape"){event.preventDefault();onClose();return}if(event.key==="Tab"){const controls=Array.from(event.currentTarget.querySelectorAll<HTMLElement>("button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled])"));const first=controls[0],last=controls.at(-1);if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus()}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus()}}}}>
    <span className="eyebrow">{dialog.kind==="confirm"?t("common:warning"):dialog.kind==="alert"?t("app:dialog.noticeEyebrow"):t("app:dialog.inputEyebrow")}</span>
    <h2 id="app-dialog-title">{dialog.title}</h2>
    {dialog.message&&<p id="app-dialog-message">{dialog.message}</p>}
    {dialog.kind==="prompt"&&<label>{dialog.inputLabel||t("common:name")}<input autoFocus value={value} onChange={event=>setValue(event.target.value)} onKeyDown={event=>{if(event.key==="Enter"){event.preventDefault();submit(value)}}}/></label>}
    <div className="modal-actions">
      {dialog.kind!=="alert"&&<button type="button" className="secondary" onClick={onClose}>{t("common:cancel")}</button>}
      {dialog.kind==="alert"?<button type="button" className="primary" autoFocus onClick={onClose}>{t("common:close")}</button>:dialog.kind==="confirm"?<button type="button" className="primary" autoFocus onClick={()=>submit()}>{dialog.confirmLabel||t("common:continue")}</button>:<button type="button" className="primary" onClick={()=>submit(value)}>{dialog.confirmLabel||t("common:continue")}</button>}
    </div>
  </section></div>;
}

function Builder({ architecture, task, dataset, validationStats, validationMessage, models, trainingRuns, activeModelId, onSelectModel, onCloseModel, onAddModel, onReuseCheckpoint, onApplyPretrained, theme, nodes, edges, setNodes, setEdges, onNodesChange, onEdgesChange, onDirty, selected, selectedId, setSelectedId, onValidate, onSaveRevision }: { architecture: Architecture; task:TaskId; dataset:DatasetSummary|null; validationStats:GraphValidation|null; validationMessage:string; models:ProjectModel[]; trainingRuns:TrainingRun[]; activeModelId:string|null; onSelectModel:(id:string)=>void; onCloseModel:(id:string)=>void; onAddModel:()=>void; onReuseCheckpoint:(payload:{name:string;nodes:ModelNodeT[];edges:ModelEdge[];checkpoint:string;sourceRunId:string;sourceModelId:string;sourceNodeId:string;sourceHandle?:string})=>void; onApplyPretrained:(spec:PretrainedModelSpec)=>void; theme: Theme; nodes: ModelNodeT[]; edges: ModelEdge[]; setNodes: React.Dispatch<React.SetStateAction<ModelNodeT[]>>; setEdges: React.Dispatch<React.SetStateAction<ModelEdge[]>>; onNodesChange: any; onEdgesChange: any; onDirty: () => void; selected: ModelNodeT | null; selectedId:string|null; setSelectedId: (id: string | null) => void; onValidate: () => void; onSaveRevision: () => void }) {
  const { t } = useTranslation(["app", "common", "catalog"]);
  const [libraryOpen, setLibraryOpen] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [libraryTab,setLibraryTab]=useState<"layers"|"blocks">("layers");
  const canvasRef=useRef<HTMLDivElement>(null);
  const insertionCount=useRef(0);
  const [flow, setFlow] = useState<ReactFlowInstance<ModelNodeT, ModelEdge> | null>(null);
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null);
  const [layoutDirection, setLayoutDirection] = useState<GraphLayoutDirection>("horizontal");
  const [hoveredBlock,setHoveredBlock]=useState<(typeof BLOCKS)[Architecture][number]|null>(null);
  const [pinnedBlock,setPinnedBlock]=useState<(typeof BLOCKS)[Architecture][number]|null>(null);
  const [hoveredPreset,setHoveredPreset]=useState<{name:string;description:string;nodes:ModelNodeT[];edges:ModelEdge[];benefit?:string;tradeoff?:string}|null>(null);
  const infoCard=useRef<HTMLDivElement>(null);
  const [presetsOpen,setPresetsOpen]=useState(false);
  const [presetTab,setPresetTab]=useState<"scratch"|"pretrained">("scratch");
  const [pretrainedModels,setPretrainedModels]=useState<PretrainedModelSpec[]>([]);
  const [pretrainedLoading,setPretrainedLoading]=useState(false);
  const [scratchGuidance,setScratchGuidance]=useState("");
  const [presetName,setPresetName]=useState("");
  const [customPresets,setCustomPresets]=useState<Array<{id:string;name:string;description:string;nodes:ModelNodeT[];edges:ModelEdge[]}>>(()=>JSON.parse(localStorage.getItem(`mb-presets-${architecture}`)||"[]"));
  const [reuseBackboneOpen,setReuseBackboneOpen]=useState(false);
  const [builderDialog,setBuilderDialog]=useState<AppDialogState|null>(null);
  const [reuseRunId,setReuseRunId]=useState("");
  const [reuseFeatureKey,setReuseFeatureKey]=useState("");
  const [reuseBackboneError,setReuseBackboneError]=useState("");
  const reusableRuns=useMemo(()=>trainingRuns.filter(run=>run.status==="completed"&&!!run.result?.checkpoint&&!!run.modelGraph&&run.modelTask?.startsWith("image.")&&run.modelArchitecture===architecture&&run.modelInputShape?.[0]===dataset?.inputShape?.[0]),[trainingRuns,architecture,dataset?.inputShape]);
  const selectedReuseRun=reusableRuns.find(run=>run.id===reuseRunId)||reusableRuns.at(-1);
  const reuseFeatures=useMemo(()=>selectedReuseRun?.modelGraph?featurePorts(selectedReuseRun.modelGraph.nodes):[],[selectedReuseRun]);
  const selectedReuseFeature=reuseFeatures.find(feature=>`${feature.nodeId}::${feature.handle||""}`===reuseFeatureKey)||reuseFeatures[0];
  useEffect(()=>{if(reuseBackboneOpen){setReuseRunId(reusableRuns.at(-1)?.id||"");setReuseFeatureKey("");setReuseBackboneError("")}},[reuseBackboneOpen]);
  useEffect(()=>{if(!reuseFeatures.some(feature=>`${feature.nodeId}::${feature.handle||""}`===reuseFeatureKey))setReuseFeatureKey(reuseFeatures[0]?`${reuseFeatures[0].nodeId}::${reuseFeatures[0].handle||""}`:"")},[reuseFeatures,reuseFeatureKey]);
  useEffect(()=>localStorage.setItem(`mb-presets-${architecture}`,JSON.stringify(customPresets)),[architecture,customPresets]);
  useEffect(()=>setLayoutDirection(inferLayoutDirection(nodes,edges)),[activeModelId]);
  useEffect(()=>{if(!presetsOpen)return;setPretrainedLoading(true);Promise.all([backend<PretrainedModelSpec[]>("model.catalog",{task_id:task,architecture}),backend<{reason?:string}>("model.catalog.guidance",{architecture})]).then(([items,guidance])=>{setPretrainedModels(items);setScratchGuidance(guidance.reason||"")}).catch(()=>{setPretrainedModels([]);setScratchGuidance("")}).finally(()=>setPretrainedLoading(false))},[presetsOpen,architecture,task]);
  useEffect(()=>{
    if(!pinnedBlock)return;
    const close=(event:PointerEvent)=>{if(infoCard.current&&!infoCard.current.contains(event.target as Node))setPinnedBlock(null)};
    document.addEventListener("pointerdown",close);
    return()=>document.removeEventListener("pointerdown",close);
  },[pinnedBlock]);

  const removeEdge = useCallback((edgeId: string) => {
    onDirty();
    setEdges(es => es.filter(e => e.id !== edgeId));
  }, [setEdges, onDirty]);
  const toggleExpanded=useCallback((id:string)=>setNodes(items=>items.map(item=>item.id===id?{...item,data:{...item.data,expanded:!item.data.expanded}}:item)),[setNodes]);

  const liveErrors=useMemo(()=>graphErrorMap(nodes,edges,t),[nodes,edges,t]);
  const selectedNodes=useMemo(()=>nodes.filter(node=>node.selected),[nodes]);
  const selectedEdges=useMemo(()=>edges.filter(edge=>edge.selected),[edges]);
  const validationState=validationStats?"valid":validationMessage||liveErrors.size?"errors":"pending";
  const validationStateLabel=validationState==="valid"?t("common:success"):validationState==="errors"?t("common:error"):t("common:loading");
  const validationIssueCount=Math.max(liveErrors.size,validationMessage?1:0);
  const selectionToolbarRef=useRef<HTMLDivElement>(null);
  const updateSelectionToolbarPosition=useCallback(()=>{
    const toolbar=selectionToolbarRef.current,canvas=canvasRef.current;
    if(!toolbar||!canvas||!flow||(!selectedNodes.length&&!selectedEdges.length))return;
    const rect=canvas.getBoundingClientRect();
    let anchor:{x:number;y:number};
    if(selectedNodes.length){
      const left=Math.min(...selectedNodes.map(node=>node.position.x));
      const right=Math.max(...selectedNodes.map(node=>node.position.x+(node.measured?.width||190)));
      const top=Math.min(...selectedNodes.map(node=>node.position.y));
      anchor={x:(left+right)/2,y:top};
    }else anchor=flow.screenToFlowPosition({x:rect.left+rect.width/2,y:rect.top+rect.height/2});
    const screen=flow.flowToScreenPosition(anchor);
    const inset=Math.min(126,Math.max(8,rect.width/2-8));
    toolbar.style.left=`${Math.max(inset,Math.min(rect.width-inset,screen.x-rect.left))}px`;
    toolbar.style.top=`${Math.max(72,screen.y-rect.top-12)}px`;
  },[flow,selectedNodes,selectedEdges]);
  useEffect(()=>{updateSelectionToolbarPosition()},[updateSelectionToolbarPosition,libraryOpen,inspectorOpen]);
  const selectedError=selected?(liveErrors.get(selected.id)||selected.data.error):undefined;
  const nodesWithHandler = useMemo(() => {
    return nodes.map(n => ({
      ...n,
      data: {
        ...n.data,
        layoutDirection,
        tensorTask:task,
        tensorArchitecture:architecture,
        error: liveErrors.get(n.id)||n.data.error,
        inputHandles:["concat","add"].includes(n.data.blockType)?Array.from({length:Math.max(2,edges.filter(edge=>edge.target===n.id).length+1,...edges.filter(edge=>edge.target===n.id).map(edge=>Number(edge.targetHandle?.match(/^input_(\d+)$/)?.[1]??-1)+2))},(_,index)=>`input_${index}`):undefined,
        onToggleExpanded: toggleExpanded
      }
    }));
  }, [nodes,edges,task,architecture,layoutDirection,toggleExpanded,liveErrors]);

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

  const candidateEdges = useCallback((connection: Connection | ModelEdge, removedId?: string) => candidateEdgesForConnection(connection,nodes,edges,removedId),[edges,nodes]);
  const validCandidate = useCallback((connection: Connection | ModelEdge, removedId?: string) => isValidGraphConnection(connection,nodes,edges,removedId),[nodes,edges]);
  const isValidConnection = useCallback((connection: Connection | ModelEdge) => validCandidate(connection,(connection as ModelEdge).id),[validCandidate]);
  const connect = useCallback((connection: Connection) => {
    if(!validCandidate(connection))return;
    onDirty();setEdges(addEdge({...connection,type:"smoothstep"},candidateEdges(connection)));
  },[validCandidate,candidateEdges,onDirty,setEdges]);
  const reconnect = useCallback((old:ModelEdge, connection:Connection)=>{
    if(!validCandidate(connection,old.id))return;
    onDirty();setEdges(addEdge({...connection,id:old.id,type:"smoothstep"},candidateEdges(connection,old.id)));
  },[validCandidate,candidateEdges,onDirty,setEdges]);

  const insertMergeBranch = (kind:"add"|"concat") => {
    const source=nodes.find(node=>node.id===selectedId);
    const outgoing=edges.filter(edge=>edge.source===selectedId);
    if(!source||source.data.blockType==="output"||outgoing.length!==1)return;
    const match=source.data.shape?.match(/^\[B,\s*(\d+),\s*\d+,\s*\d+\]$/);
    if(!match)return;
    const channels=Number(match[1]);
    const downstream=outgoing[0];
    const sourceIndex=nodes.findIndex(node=>node.id===source.id);
    const branchX=source.position.x+230;
    const mergeX=source.position.x+(kind==="add"?920:460);
    const mergeId=`${kind}-${uid().slice(0,8)}`;
    const branchId=`${kind}-branch-${uid().slice(0,8)}`;
    const nextId=downstream.target;
    const merge:ModelNodeT={id:mergeId,type:"modelNode",position:{x:mergeX,y:source.position.y},deletable:true,data:{blockType:kind,label:kind==="add"?t("app:builder.residualAddLabel"):t("app:builder.concatLabel"),category:"Merge",inputHandles:["input_0","input_1"],properties:{axis:1}}};
    const branchNodes:ModelNodeT[]=kind==="add"?[
      {id:branchId,type:"modelNode",position:{x:branchX,y:source.position.y-220},deletable:true,data:{blockType:"conv2d",label:t("app:builder.residualConvLabel"),category:"Convolution",properties:{out_channels:channels,kernel_size:3,stride:1,padding:1,bias:false}}},
      {id:`${branchId}-relu`,type:"modelNode",position:{x:branchX+230,y:source.position.y-220},deletable:true,data:{blockType:"relu",label:"ReLU",category:"Activation",properties:{}}},
      {id:`${branchId}-conv2`,type:"modelNode",position:{x:branchX+460,y:source.position.y-220},deletable:true,data:{blockType:"conv2d",label:t("app:builder.residualConvLabel"),category:"Convolution",properties:{out_channels:channels,kernel_size:3,stride:1,padding:1,bias:false}}}
    ]:[{id:branchId,type:"modelNode",position:{x:branchX,y:source.position.y-220},deletable:true,data:{blockType:"identity",label:t("app:builder.concatBranchLabel"),category:"Shape",properties:{}}}];
    const branchEnd=kind==="add"?`${branchId}-conv2`:branchId;
    const branchEdges:ModelEdge[]=kind==="add"?[
      {id:`${source.id}-${branchId}`,source:source.id,target:branchId,type:"smoothstep"},
      {id:`${branchId}-${branchId}-relu`,source:branchId,target:`${branchId}-relu`,type:"smoothstep"},
      {id:`${branchId}-relu-${branchId}-conv2`,source:`${branchId}-relu`,target:`${branchId}-conv2`,type:"smoothstep"},
      {id:`${branchId}-conv2-${mergeId}`,source:branchEnd,target:mergeId,targetHandle:"input_1",type:"smoothstep"}
    ]:[
      {id:`${source.id}-${branchId}`,source:source.id,target:branchId,type:"smoothstep"},
      {id:`${branchId}-${mergeId}`,source:branchId,target:mergeId,targetHandle:"input_1",type:"smoothstep"}
    ];
    onDirty();
    const descendants=new Set<string>();
    const pending=[nextId];
    while(pending.length){const id=pending.pop()!;if(descendants.has(id))continue;descendants.add(id);pending.push(...edges.filter(edge=>edge.source===id).map(edge=>edge.target));}
    const next=nodes.find(node=>node.id===nextId);
    const shift=Math.max(0,mergeX+230-(next?.position.x??mergeX+230));
    setNodes(items=>{const moved=items.map(item=>descendants.has(item.id)?{...item,position:{...item.position,x:item.position.x+shift}}:item);return [...moved.slice(0,sourceIndex+1),...branchNodes,merge,...moved.slice(sourceIndex+1)]});
    setEdges(items=>[...items.filter(edge=>edge.id!==downstream.id),{id:`${source.id}-${mergeId}`,source:source.id,target:mergeId,targetHandle:"input_0",type:"smoothstep"},...branchEdges,{id:`${mergeId}-${nextId}`,source:mergeId,target:nextId,type:"smoothstep"}]);
    setSelectedId(mergeId);
    window.setTimeout(()=>flow?.fitView({nodes:[{id:source.id},{id:mergeId},{id:nextId}],duration:250,padding:.5}),20);
  };

  const select: NodeMouseHandler<ModelNodeT> = (_, n) => { setSelectedId(n.id); setSelectedGroupId(null); };
  const centerFlowPoint=()=>{const rect=canvasRef.current?.getBoundingClientRect();return rect&&flow?flow.screenToFlowPosition({x:rect.left+rect.width/2,y:rect.top+rect.height/2}):{x:400,y:250}};
  const nextInsertionCenter=()=>{const point=centerFlowPoint();const index=insertionCount.current++;const radius=index?10+Math.sqrt(index)*5:0;const angle=index*2.399963229728653;return {x:point.x+Math.cos(angle)*radius,y:point.y+Math.sin(angle)*radius}};
  const centerPosition=()=>{const point=nextInsertionCenter();return {x:point.x-95,y:point.y-55}};
  const addBlock = (b: typeof BLOCKS[Architecture][number], position = centerPosition()) => {
    onDirty();
    const id = `${b.type}-${uid().slice(0, 6)}`;
    setNodes(ns => [...ns.map(node=>({...node,selected:false})), { id, type: "modelNode", position, selected:true, deletable: true, data: { blockType: b.type, label: b.label, category: b.category, properties: { ...b.defaults } } }]);
    setSelectedId(id);
  };

  const duplicateSelection=useCallback(()=>{
    const duplicate=duplicateGraphSelection(nodes,edges,type=>`${type}-${uid()}`);
    if(!duplicate.selectedId)return;
    onDirty();setNodes(duplicate.nodes);setEdges(duplicate.edges);setSelectedId(duplicate.selectedId);const groupId=duplicate.nodes.find(node=>node.id===duplicate.selectedId)?.data.groupId;setSelectedGroupId(typeof groupId==="string"?groupId:null);
  },[nodes,edges,onDirty,setNodes,setEdges,setSelectedId]);

  const deleteNodeIds=useCallback((requested:Set<string>)=>{
    if(!requested.size)return;
    const groupedIds=new Set(nodes.filter(node=>requested.has(node.id)&&typeof node.data.groupId==="string").map(node=>node.data.groupId as string));
    const outputs=nodes.filter(node=>node.data.blockType==="output");
    const keepOutputId=outputs.find(node=>!requested.has(node.id))?.id||outputs[0]?.id;
    const deleting=new Set([...requested].filter(id=>{
      const node=nodes.find(item=>item.id===id);
      return node&&node.data.blockType!=="input"&&!(node.data.blockType==="output"&&id===keepOutputId);
    }));
    const remaining=nodes.filter(node=>!deleting.has(node.id));
    const protectedGroupMembers=remaining.filter(node=>groupedIds.has(String(node.data.groupId)));
    if(!deleting.size&&protectedGroupMembers.length===0)return;
    onDirty();
    setNodes(items=>items.filter(node=>!deleting.has(node.id)).map(node=>({
      ...node,
      selected:false,
      ...(groupedIds.has(String(node.data.groupId))?{data:{...node.data,groupId:undefined,groupName:undefined}}:{}),
    })));
    setEdges(items=>items.filter(edge=>!deleting.has(edge.source)&&!deleting.has(edge.target)));
    setSelectedId(selectedId&&deleting.has(selectedId)?null:selectedId);
    setSelectedGroupId(null);
  },[nodes,selectedId,onDirty,setNodes,setEdges,setSelectedId]);
  const deleteSelection=useCallback(()=>{
    const selectedGroups=new Set(nodes.filter(node=>node.selected&&typeof node.data.groupId==="string").map(node=>node.data.groupId as string));
    const requested=new Set(nodes.filter(node=>node.selected||selectedGroups.has(String(node.data.groupId))).map(node=>node.id));
    if(requested.size){deleteNodeIds(requested);return;}
    const selectedEdges=edges.filter(edge=>edge.selected);
    if(selectedEdges.length){onDirty();const ids=new Set(selectedEdges.map(edge=>edge.id));setEdges(items=>items.filter(edge=>!ids.has(edge.id)));}
  },[nodes,edges,deleteNodeIds,onDirty,setEdges]);
  useEffect(()=>{
    const key=(event:KeyboardEvent)=>{
      const target=event.target as HTMLElement|null;
      if(target?.closest("input,textarea,select,[contenteditable]:not([contenteditable='false'])"))return;
      if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="d"){event.preventDefault();duplicateSelection();return;}
      if(event.key==="Delete"||event.key==="Backspace"){
        if(!nodes.some(node=>node.selected)&&!edges.some(edge=>edge.selected))return;
        event.preventDefault();deleteSelection();
      }
    };
    window.addEventListener("keydown",key);
    return()=>window.removeEventListener("keydown",key);
  },[duplicateSelection,deleteSelection,nodes,edges]);

  const visualGroups=useMemo(()=>{
    const grouped=new Map<string,ModelNodeT[]>();for(const node of nodes){const id=node.data.groupId;if(typeof id==="string"){const members=grouped.get(id)||[];members.push(node);grouped.set(id,members)}}
    return [...grouped].map(([id,members])=>({id,members,name:String(members[0].data.groupName||"Group"),selected:selectedGroupId===id,x:Math.min(...members.map(n=>n.position.x))-24,y:Math.min(...members.map(n=>n.position.y))-48,right:Math.max(...members.map(n=>n.position.x+(n.measured?.width||190)))+24,bottom:Math.max(...members.map(n=>n.position.y+(n.measured?.height||110)))+24}));
  },[nodes,selectedGroupId]);
  const groupSelection=()=>{const members=nodes.filter(n=>n.selected);if(members.length<2)return;const memberIds=new Set(members.map(node=>node.id));setBuilderDialog({kind:"prompt",title:t("app:builder.groupCreateTitle"),inputLabel:t("app:builder.groupName"),initialValue:t("app:builder.groupDefault"),onConfirm:name=>{if(!name?.trim())return;const id=uid();onDirty();setNodes(items=>items.map(n=>memberIds.has(n.id)?{...n,data:{...n.data,groupId:id,groupName:name.trim()}}:n));setSelectedGroupId(id);setSelectedId(members[0]?.id||null)}})};
  const ungroupSelection=()=>{const ids=new Set(nodes.filter(n=>n.selected&&n.data.groupId).map(n=>n.data.groupId));if(!ids.size)return;onDirty();setNodes(items=>items.map(n=>ids.has(n.data.groupId)?{...n,data:{...n.data,groupId:undefined,groupName:undefined}}:n));setSelectedGroupId(null)};
  const selectGroup=(id:string)=>{const first=nodes.find(node=>node.data.groupId===id);setNodes(items=>items.map(n=>({...n,selected:n.data.groupId===id})));setSelectedId(first?.id||null);setSelectedGroupId(id)};
  const renameGroup=(id:string,name:string)=>setBuilderDialog({kind:"prompt",title:t("app:builder.groupRenameTitle"),inputLabel:t("app:builder.groupName"),initialValue:name,onConfirm:value=>{const next=value?.trim();if(next)setNodes(items=>items.map(n=>n.data.groupId===id?{...n,data:{...n.data,groupName:next}}:n))}});
  const dragGroup=(event:React.PointerEvent,id:string)=>{if(event.button!==0||!flow)return;event.preventDefault();event.stopPropagation();selectGroup(id);let previous=flow.screenToFlowPosition({x:event.clientX,y:event.clientY});const move=(e:PointerEvent)=>{const current=flow.screenToFlowPosition({x:e.clientX,y:e.clientY});const dx=current.x-previous.x,dy=current.y-previous.y;previous=current;setNodes(items=>items.map(n=>n.data.groupId===id?{...n,position:{x:n.position.x+dx,y:n.position.y+dy}}:n))};const up=()=>{window.removeEventListener("pointermove",move);window.removeEventListener("pointerup",up);window.removeEventListener("pointercancel",up)};window.addEventListener("pointermove",move);window.addEventListener("pointerup",up);window.addEventListener("pointercancel",up)};
  const applyLayout=(direction:GraphLayoutDirection)=>{
    setLayoutDirection(direction);
    const laidOut=layoutConnectedGraph(nodes,edges,direction);
    setNodes(laidOut);
    const ordered=[...laidOut].sort((a,b)=>direction==="vertical"?a.position.y-b.position.y:a.position.x-b.position.x);
    const focus=ordered.slice(0,2).map(({id})=>({id}));
    window.requestAnimationFrame(()=>{if(focus.length)void flow?.fitView({nodes:focus,duration:250,padding:.15,minZoom:1,maxZoom:1})});
  };

  const addCnnRecipe=(kind:0|1|2|3)=>{
    const name=kind===0?t("app:builder.recipeResidual"):t("app:builder.recipeConv",{count:kind});
    const recipe=createCnnRecipe(kind,nextInsertionCenter(),name,type=>`${type}-${uid().slice(0,8)}`,type=>t(`catalog:blocks.${type}.label`,{defaultValue:type}));
    onDirty();setNodes(items=>[...items.map(node=>({...node,selected:false})),...recipe.nodes]);setEdges(items=>[...items,...recipe.edges]);setSelectedId(recipe.nodes[0]?.id||null);
  };

  const addOutput=()=>{
    onDirty();
    const outputs=nodes.filter(node=>node.data.blockType==="output");
    const name=`Output ${outputs.length+1}`;
    const source=nodes.find(node=>node.id===selectedId&&node.data.blockType!=="output");
    const id=`output-${uid().slice(0,8)}`;
    const position={x:Math.max(0,...nodes.map(node=>node.position.x))+190,y:(source?.position.y??180)+outputs.length*125};
    const output:ModelNodeT={id,type:"modelNode",position,deletable:true,data:{blockType:"output",label:name,category:"Output",properties:{output_name:name}}};
    setNodes(items=>[...items,output]);
    if(source)setEdges(items=>addEdge({id:`${source.id}-${id}`,source:source.id,target:id,type:"smoothstep"},items));
    setSelectedId(id);
  };

  const createFromCheckpointBackbone=()=>{
    if(!selectedReuseRun?.modelGraph||!selectedReuseRun.result?.checkpoint||!selectedReuseFeature)return;
    try{
      const extracted=extractCnnBackbone({nodes:selectedReuseRun.modelGraph.nodes,edges:selectedReuseRun.modelGraph.edges,feature:selectedReuseFeature,task,dataset,inputShape:dataset?.inputShape||[],sourceInputShape:selectedReuseRun.modelInputShape});
      const sourceModel=models.find(model=>model.id===selectedReuseRun.modelId);
      const taskLabel=t(`catalog:tasks.${task}.name`,{defaultValue:task});
      const name=`${sourceModel?.name||selectedReuseRun.name} · ${selectedReuseFeature.label} · ${taskLabel}`;
      onReuseCheckpoint({name,nodes:extracted.nodes,edges:extracted.edges,checkpoint:selectedReuseRun.result.checkpoint,sourceRunId:selectedReuseRun.id,sourceModelId:selectedReuseRun.modelId||"",sourceNodeId:selectedReuseFeature.nodeId,sourceHandle:selectedReuseFeature.handle});
      setReuseBackboneOpen(false);
    }catch(error){const message=error instanceof Error?error.message:String(error);setReuseBackboneError(message==="CNN_REUSE_DENSE_SCALE_INVALID"?t("app:builder.reuseScaleInvalid"):message)}
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

  const applyPreset=(preset:{nodes:ModelNodeT[];edges:ModelEdge[]})=>{onDirty();setNodes(syncOutputContract(preset.nodes,preset.edges,dataset).map(node=>({...node,data:{...node.data,error:undefined}})));setEdges(preset.edges);setSelectedId(null);setSelectedGroupId(null);setLayoutDirection(inferLayoutDirection(preset.nodes,preset.edges));setPresetsOpen(false);window.setTimeout(()=>flow?.fitView(initialGraphFit(preset.nodes)),20)};
  const applyPretrained=async(spec:PretrainedModelSpec)=>{await backend("model.license.accept",{model_id:spec.id});onApplyPretrained(spec);applyPreset(pretrainedTemplate(spec,task))};
  const savePreset=(requestedName=presetName)=>{const name=requestedName.trim();if(!name)return;const preset={id:uid(),name,description:t("app:customPreset.description"),nodes:nodes.map(node=>({...node,selected:false})),edges:edges.map(edge=>({...edge,selected:false}))};setCustomPresets(items=>[...items,preset]);setPresetName("")};
  const requestSaveRevision=()=>setBuilderDialog({kind:"prompt",title:t("app:savePreset.label"),inputLabel:t("common:name"),initialValue:presetName,onConfirm:name=>{if(!name?.trim())return;savePreset(name);void onSaveRevision()}});
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
      <div className="model-tabs" role="tablist" aria-label={t("app:model.modelsAria")}>{models.map(model=>{const modelRunning=trainingRuns.some(run=>run.modelId===model.id&&(run.status==="running"||run.status==="paused"));const closeTitle=modelRunning?t("app:builder.closeModelTraining"):models.length<=1?t("app:builder.closeModelLast"):t("app:builder.closeModel");return <div className="closable-tab model-tab" role="presentation" key={model.id}><button role="tab" aria-selected={model.id===activeModelId} className={model.id===activeModelId?"active":""} onClick={()=>onSelectModel(model.id)}><BrainCircuit size={13}/>{model.name}<small>{t(`catalog:architectures.${model.architecture}.name`)}</small></button><button className="tab-close" disabled={models.length<=1||modelRunning} onClick={event=>{event.stopPropagation();onCloseModel(model.id)}} title={closeTitle} aria-label={closeTitle}><X size={12}/></button></div>})}<button className="add-model" onClick={onAddModel} title={t("app:model.createTitle")} aria-label={t("app:model.createTitle")}><Plus size={15}/></button></div>
      <div className="builder-toolbar">
        <button className={presetsOpen ? "active" : ""} onClick={()=>{setPresetTab("scratch");setPresetsOpen(true)}}><Layers3/> {t("common:presets")}</button>
        {reusableRuns.length>0&&<button onClick={()=>setReuseBackboneOpen(true)}><Network/> {t("app:builder.reuseBackbone")}</button>}
        <button onClick={addOutput} title={t("app:builder.addOutputHelp")}><Plus/> {t("app:builder.addOutput")}</button>
        <button disabled={!selected?.data.shape?.match(/^\[B,\s*\d+,\s*\d+,\s*\d+\]$/)||edges.filter(edge=>edge.source===selectedId).length!==1} onClick={()=>insertMergeBranch("add")} title={t("app:builder.residualHelp")}><GitFork/> {t("app:builder.addResidual")}</button>
        <button disabled={!selected?.data.shape?.match(/^\[B,\s*\d+,\s*\d+,\s*\d+\]$/)||edges.filter(edge=>edge.source===selectedId).length!==1} onClick={()=>insertMergeBranch("concat")} title={t("app:builder.concatHelp")}><Split/> {t("app:builder.addConcat")}</button>
        <span className="toolbar-separator" />
        <div className="builder-layout-toggle" role="group" aria-label={t("app:builder.layout") }><span>{t("app:builder.layout")}</span><button type="button" className={layoutDirection==="horizontal"?"active":""} aria-pressed={layoutDirection==="horizontal"} title={t("app:builder.layoutHorizontalHelp")} onClick={()=>applyLayout("horizontal")}><ArrowRight/>{t("app:builder.layoutHorizontal")}</button><button type="button" className={layoutDirection==="vertical"?"active":""} aria-pressed={layoutDirection==="vertical"} title={t("app:builder.layoutVerticalHelp")} onClick={()=>applyLayout("vertical")}><ArrowDown/>{t("app:builder.layoutVertical")}</button></div>
        <span className="toolbar-separator" />
        <button onClick={requestSaveRevision}><Save /> {t("app:builder.saveRevision")}</button>
        <div className="toolbar-spacer" />
        <span>{t("app:builder.graphSummary", { blocks: nodes.length, connections: edges.length })}</span>
      </div>

      <div className="builder-workspace">
        {libraryOpen && (
          <aside className="block-library">
            <div><span className="eyebrow">{t("app:builder.libraryEyebrow")}</span><h3>{t(`catalog:architectures.${architecture}.name`)}</h3></div>
            <div className="library-tabs" role="tablist" aria-label={t("app:builder.libraryEyebrow")}>
              <button role="tab" aria-selected={libraryTab==="layers"} className={libraryTab==="layers"?"active":""} onClick={()=>{setLibraryTab("layers");setHoveredPreset(null)}}>{t("app:builder.libraryLayersTab")}</button>
              <button role="tab" aria-selected={libraryTab==="blocks"} className={libraryTab==="blocks"?"active":""} onClick={()=>{setLibraryTab("blocks");setHoveredBlock(null)}}>{t("app:builder.libraryBlocksTab")}</button>
            </div>
            {libraryTab==="layers"&&Object.entries(groups).map(([cat, blocks]) => {
              const layers=blocks.filter(block=>block.type!=="multi_scale_cnn");
              return layers.length?<section key={cat}>
                <small>{t(`catalog:blockCategories.${cat}`)}</small>
                {layers.map(b => { const Icon=operationIcon(b.type);return <div className="library-item" key={b.type} onMouseEnter={()=>{if(!pinnedBlock){setHoveredPreset(null);setHoveredBlock(b)}}} onMouseLeave={()=>setHoveredBlock(null)}><button draggable onClick={() => addBlock(b)} onDragStart={e => { e.dataTransfer.setData("application/modelbuilder", b.type); e.dataTransfer.effectAllowed = "move"; }}><Icon size={13} />{t(`catalog:blocks.${b.type}.label`)}</button><button className={`block-info-button ${pinnedBlock?.type===b.type?"active":""}`} onClick={event=>{event.stopPropagation();setPinnedBlock(b);setHoveredBlock(null)}} title={t("app:blockInfo.pinTitle", { block: t(`catalog:blocks.${b.type}.label`) })} aria-label={t("app:blockInfo.pinTitle", { block: t(`catalog:blocks.${b.type}.label`) })}><Info size={13}/></button></div>})}
              </section>:null;
            })}
            {libraryTab==="blocks"&&<>
              {architecture==="cnn"&&<section><small>{t("app:builder.libraryRecipes")}</small>{([1,2,3,0] as const).map(kind=>{const name=kind===0?t("app:builder.recipeResidual"):t("app:builder.recipeConv",{count:kind});return <div className="library-item library-preset-item" key={kind} onMouseEnter={()=>{if(!pinnedBlock)setHoveredPreset({name,description:t("app:builder.recipePreviewDescription"),nodes:[],edges:[]})}} onMouseLeave={()=>setHoveredPreset(null)}><button onClick={()=>addCnnRecipe(kind)}><Layers3 size={13}/>{kind===0?t("app:builder.recipeResidualShort"):t("app:builder.recipeConvShort",{count:kind})}</button></div>})}</section>}
              {architecture==="cnn"&&BLOCKS[architecture].filter(block=>block.type==="multi_scale_cnn").map(block=>{const Icon=operationIcon(block.type);return <section key={block.type}><small>{t("app:builder.libraryComposite")}</small><div className="library-item" onMouseEnter={()=>{if(!pinnedBlock){setHoveredPreset(null);setHoveredBlock(block)}}} onMouseLeave={()=>setHoveredBlock(null)}><button draggable onClick={()=>addBlock(block)} onDragStart={event=>{event.dataTransfer.setData("application/modelbuilder",block.type);event.dataTransfer.effectAllowed="move"}}><Icon size={13}/>{t(`catalog:blocks.${block.type}.shortLabel`,{defaultValue:t(`catalog:blocks.${block.type}.label`)})}</button><button className={`block-info-button ${pinnedBlock?.type===block.type?"active":""}`} onClick={event=>{event.stopPropagation();setPinnedBlock(block);setHoveredBlock(null)}} title={t("app:blockInfo.pinTitle",{block:t(`catalog:blocks.${block.type}.label`)})} aria-label={t("app:blockInfo.pinTitle",{block:t(`catalog:blocks.${block.type}.label`)})}><Info size={13}/></button></div></section>})}
              <section><small>{t("app:builder.libraryPresets")}</small>{builtinPresets.map(preset=>{const info=presetI18n(preset);return <div className="library-item library-preset-item" key={preset.id} onMouseEnter={()=>setHoveredPreset({name:info.name,description:info.description,nodes:preset.nodes,edges:preset.edges,benefit:info.benefit,tradeoff:info.tradeoff})} onMouseLeave={()=>setHoveredPreset(null)}><button onClick={()=>applyPreset(preset)}><Sparkles size={13}/>{info.name.split(" · ").at(-1)||info.name}</button></div>})}{customPresets.map(preset=><div className="library-item library-preset-item" key={preset.id} onMouseEnter={()=>setHoveredPreset(preset)} onMouseLeave={()=>setHoveredPreset(null)}><button onClick={()=>applyPreset(preset)}><Save size={13}/>{preset.name}</button></div>)}
                <div className="library-save-preset"><input value={presetName} onChange={event=>setPresetName(event.target.value)} placeholder={t("app:savePreset.placeholder")} aria-label={t("app:savePreset.label")}/><button disabled={!presetName.trim()} onClick={()=>savePreset()}><Save size={12}/>{t("common:save")}</button></div>
              </section>
            </>}
          </aside>
        )}

        <div className="flow-wrap" ref={canvasRef}>
          <button type="button" className={`builder-panel-toggle builder-panel-toggle-library ${libraryOpen?"panel-open":"panel-closed"}`} onClick={()=>setLibraryOpen(value=>!value)} title={`${libraryOpen?t("common:close"):t("common:open")} ${t("common:blocks")}`} aria-label={`${libraryOpen?t("common:close"):t("common:open")} ${t("common:blocks")}`}><ChevronRight size={15}/></button>
          <button type="button" className={`builder-panel-toggle builder-panel-toggle-inspector ${inspectorOpen?"panel-open":"panel-closed"}`} onClick={()=>setInspectorOpen(value=>!value)} title={`${inspectorOpen?t("common:close"):t("common:open")} ${t("app:inspector.properties")}`} aria-label={`${inspectorOpen?t("common:close"):t("common:open")} ${t("app:inspector.properties")}`}><ChevronRight size={15}/></button>
          <div className="canvas-navigation-hint">{t("app:builder.canvasNavigation")}</div>
          <div className="canvas-param-badge">
            <Cpu size={13} />
            <span>{validationStats ? `${validationStats.parameters.toLocaleString()} total · ${validationStats.trainable_parameters.toLocaleString()} entrenables · ~${Math.ceil(validationStats.parameters*4/1048576)} MB` : t("app:builder.paramEstimate", { count: paramEstimate > 0 ? paramEstimate.toLocaleString() : "---" })}</span>
          </div>
          <div className={`builder-validation-status ${validationState}`} role="status" aria-live="polite" title={validationMessage||`${t("common:validation")}: ${validationStateLabel}`}>
            {validationState==="valid"?<CheckCircle2 size={13}/>:validationState==="errors"?<AlertCircle size={13}/>:<Clock3 size={13}/>}
            <span>{t("common:validation")}</span><strong>{validationState==="errors"?`${validationIssueCount} ${validationStateLabel}`:validationStateLabel}</strong>
          </div>
          {selectedNodes.length>0||selectedEdges.length>0?<div className="canvas-selection-toolbar nodrag nopan" ref={selectionToolbarRef} role="toolbar" aria-label={t("app:builder.deleteSelection")}>
            {selectedNodes.length>0&&<button type="button" onClick={duplicateSelection} title={t("app:builder.duplicateShortcut")}><Copy size={14}/><span>{t("app:builder.duplicateSelection")}</span></button>}
            {selectedNodes.length>=2&&<button type="button" onClick={groupSelection} title={t("app:builder.groupSelection")}><Group size={14}/><span>{t("app:builder.groupSelection")}</span></button>}
            {selectedNodes.length>=2&&selectedNodes.some(node=>typeof node.data.groupId==="string")&&<button type="button" onClick={ungroupSelection} title={t("app:builder.ungroup")}><Ungroup size={14}/><span>{t("app:builder.ungroup")}</span></button>}
            <button type="button" className="danger" onClick={deleteSelection} title={t("app:builder.deleteSelection")}><Trash2 size={14}/><span>{t("app:builder.deleteSelection")}</span></button>
          </div>:null}

          <ReactFlow
            nodes={nodesWithHandler}
            edges={edgesWithHandler}
            onInit={setFlow}
            onMove={updateSelectionToolbarPosition}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={connect}
            onReconnect={reconnect}
            isValidConnection={isValidConnection}
            multiSelectionKeyCode="Shift"
            onNodeClick={select}
            onPaneClick={() => { setSelectedId(null); setSelectedGroupId(null); }}
            onDragOver={e => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; }}
            onDrop={onDrop}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
            fitViewOptions={initialGraphFit(nodes)}
            colorMode={theme === "dark" ? "dark" : "light"}
            deleteKeyCode={null}
          >
            <ViewportPortal>{visualGroups.map(group=><div className={`visual-group${group.selected?" selected":""}`} role="group" aria-label={group.name} key={group.id} style={{left:group.x,top:group.y,width:group.right-group.x,height:group.bottom-group.y}}>
              <div className="visual-group-hit visual-group-hit-top" onPointerDown={event=>dragGroup(event,group.id)} />
              <div className="visual-group-hit visual-group-hit-left" onPointerDown={event=>dragGroup(event,group.id)} />
              <div className="visual-group-hit visual-group-hit-right" onPointerDown={event=>dragGroup(event,group.id)} />
              <div className="visual-group-hit visual-group-hit-bottom" onPointerDown={event=>dragGroup(event,group.id)} />
              <div className="visual-group-header" onPointerDown={event=>{if(!(event.target as HTMLElement).closest("button"))dragGroup(event,group.id)}}>
                <button type="button" className="visual-group-title" aria-pressed={group.selected} title={t("app:builder.groupDragHint")} onPointerDown={event=>dragGroup(event,group.id)} onClick={()=>selectGroup(group.id)} onDoubleClick={()=>renameGroup(group.id,group.name)}>{group.name}</button>
              </div>
            </div>)}</ViewportPortal>
            <Background color={theme === "dark" ? "#35373b" : "#d1d2d5"} gap={22} />
            <Controls />
            <MiniMap nodeColor="var(--accent)" maskColor={theme === "dark" ? "rgba(16,17,18,.75)" : "rgba(240,240,242,.75)"} pannable zoomable />
          </ReactFlow>
        </div>

        {inspectorOpen&&<aside className="inspector">
          {selected ? (
            <>
              <div className="inspector-head">
                <span className="eyebrow">{t("app:inspector.properties")}</span>
                <h3>{t(`catalog:blocks.${selected.data.blockType}.label`, { defaultValue: selected.data.label })}</h3>
                <code>{selected.data.blockType}</code>
              </div>
      {Object.entries(selected.data.properties).length ? Object.entries(selected.data.properties).map(([k, v]) => (
                <Property key={k} name={k} label={["out_channels","in_channels","base_channels"].includes(k)?t(`app:inspector.${k}`):undefined} value={v} disabled={(selected.data.blockType.startsWith("hf_")&&!["frozen","trainable_stages","max_length","offline"].includes(k)) || (!!dataset && ((selected.data.blockType==="linear"&&k==="out_features")||(selected.data.blockType==="conv2d"&&k==="out_channels")) && edges.some(edge=>edge.source===selected.id&&nodes.find(node=>node.id===edge.target)?.data.blockType==="output"))} onChange={nv => updateProp(k, nv)} />
              )) : <p className="muted">{t("app:inspector.noParams")}</p>}
              {selected.data.blockType!=="input"&&selected.data.blockType!=="output"&&estimateNodeParameters(selected)>0&&<label className="property trainable-property"><span>{t("app:inspector.frozen")}</span><button type="button" role="switch" aria-checked={Boolean(selected.data.properties.frozen)} className={selected.data.properties.frozen?"toggle on":"toggle"} onClick={()=>updateProp("frozen",!selected.data.properties.frozen)}><i/></button></label>}
              {["batchnorm1d","batchnorm2d"].includes(selected.data.blockType)&&<label className="property trainable-property"><span>{t("app:inspector.freezeBatchNormStats")}</span><button type="button" role="switch" aria-checked={Boolean(selected.data.properties.freeze_batchnorm_stats)} className={selected.data.properties.freeze_batchnorm_stats?"toggle on":"toggle"} onClick={()=>updateProp("freeze_batchnorm_stats",!selected.data.properties.freeze_batchnorm_stats)}><i/></button></label>}
              <div className="inspector-section">
                <span className="eyebrow">{t("app:inspector.tensor")}</span>
                <TensorShapeSummary shape={selected.data.shape} blockType={selected.data.blockType} task={task} architecture={architecture} properties={selected.data.properties} label={selected.data.label} variant="inspector" />
              </div>
              <div className="inspector-section parameter-summary"><span className="eyebrow">{t("app:inspector.parameters")}</span><strong>{estimateNodeParameters(selected).toLocaleString()}</strong><small>{t("app:inspector.parametersPerBlock")}</small><div><span>{t("app:inspector.totalModel")}</span><b>{paramEstimate.toLocaleString()}</b></div></div>
              {selectedError&&<div className="inspector-error"><AlertCircle/>{translateBackendError(selectedError)}</div>}
            </>
          ) : (
            <div className="inspector-empty">
              <GitBranch />
              <h3>{t("app:inspector.emptyTitle")}</h3>
              <p>{t("app:inspector.emptyHint")}</p>
            </div>
          )}
          <details className="model-shape-summary">
            <summary>{t("app:builder.shapeSummary",{count:nodes.length})}</summary>
            <div className="shape-summary-list">{nodes.map(node=><button key={node.id} className={node.id===selectedId?"active":""} onClick={()=>{setSelectedId(node.id);flow?.fitView({nodes:[{id:node.id}],duration:250,padding:.7,maxZoom:1})}}><span>{node.data.label}</span><code>{node.data.shape||"?"}</code></button>)}</div>
          </details>
        </aside>}
        <AppDialog dialog={builderDialog} onClose={()=>setBuilderDialog(null)}/>
      </div>
      <LibraryHoverInfo block={pinnedBlock||hoveredBlock} pinned={Boolean(pinnedBlock)} preset={!pinnedBlock&&!hoveredBlock?hoveredPreset:null} selected={selected} architecture={architecture} task={task} dataset={dataset} onClose={()=>setPinnedBlock(null)} />
      {reuseBackboneOpen&&<div className="modal-backdrop" onMouseDown={()=>setReuseBackboneOpen(false)}><section className="modal reuse-backbone-modal" role="dialog" aria-modal="true" aria-labelledby="reuse-backbone-title" onMouseDown={event=>event.stopPropagation()}><button className="modal-close" onClick={()=>setReuseBackboneOpen(false)} aria-label={t("common:close")}><X/></button><span className="eyebrow">{t("app:builder.reuseBackboneEyebrow")}</span><h2 id="reuse-backbone-title">{t("app:builder.reuseBackboneTitle")}</h2><p>{t("app:builder.reuseBackboneDescription")}</p>{reusableRuns.length>0?<><label>{t("app:builder.reuseRunLabel")}<select value={selectedReuseRun?.id||""} onChange={event=>{setReuseRunId(event.target.value);setReuseFeatureKey("");setReuseBackboneError("")}}>{reusableRuns.map(run=><option key={run.id} value={run.id}>{run.name} · {run.modelGraph?.nodes.length||0} bloques · {new Date(run.createdAt).toLocaleDateString()}</option>)}</select></label>{reuseFeatures.length>0?<label>{t("app:builder.reuseFeatureLabel")}<select value={selectedReuseFeature?`${selectedReuseFeature.nodeId}::${selectedReuseFeature.handle||""}`:""} onChange={event=>setReuseFeatureKey(event.target.value)}>{reuseFeatures.map(feature=><option key={`${feature.nodeId}::${feature.handle||""}`} value={`${feature.nodeId}::${feature.handle||""}`}>{feature.label} · [B,{feature.channels},{feature.height},{feature.width}]</option>)}</select></label>:<p className="inspector-error">{t("app:builder.reuseNoFeatures")}</p>}{selectedReuseFeature&&<div className="reuse-backbone-summary"><Network/><span><strong>{selectedReuseFeature.label}</strong><small>[B, {selectedReuseFeature.channels}, {selectedReuseFeature.height}, {selectedReuseFeature.width}] · {t("app:builder.reuseFrozenHint")}</small></span></div>}{reuseBackboneError&&<p className="inspector-error">{reuseBackboneError}</p>}<div className="modal-actions"><button className="secondary" onClick={()=>setReuseBackboneOpen(false)}>{t("common:cancel")}</button><button className="primary" disabled={!selectedReuseFeature} onClick={createFromCheckpointBackbone}>{t("app:builder.reuseCreateButton")} <ChevronRight size={16}/></button></div></>:<p>{t("app:builder.reuseNoRuns")}</p>}</section></div>}
      {presetsOpen&&<div className="modal-backdrop" onMouseDown={event=>{if(event.target===event.currentTarget)setPresetsOpen(false)}}><section className="modal presets-modal" role="dialog" aria-modal="true" aria-labelledby="presets-modal-title"><button className="modal-close" onClick={()=>setPresetsOpen(false)} aria-label={t("common:close")}><X/></button><span className="eyebrow">{t("app:presets.modalEyebrow")}</span><h2 id="presets-modal-title">{t("app:presets.modalTitle")}</h2><div className="preset-tabs" role="tablist" aria-label={t("app:presets.modalTitle")}><button type="button" role="tab" aria-selected={presetTab==="scratch"} className={presetTab==="scratch"?"active":""} onClick={()=>setPresetTab("scratch")}><Sparkles/>{t("app:presets.scratchTab")}</button><button type="button" role="tab" aria-selected={presetTab==="pretrained"} className={presetTab==="pretrained"?"active":""} onClick={()=>setPresetTab("pretrained")}><Cloud/>{t("app:presets.pretrainedTab")}</button></div>{presetTab==="pretrained"?<div className="preset-view" role="tabpanel"><h3>{t("app:presets.pretrainedHeading")}</h3>{pretrainedLoading?<p className="muted">{t("app:presets.pretrainedLoading")}</p>:pretrainedModels.length>0?<div className="preset-list">{pretrainedModels.map(spec=><article key={spec.id}><div><Cloud/><span><strong>{spec.name}</strong><small>{spec.repoId} · {spec.size} · {spec.cached?t("app:presets.cached"):t("app:presets.requiresDownload")}</small></span></div><p>{spec.description}<br/><b>{t("app:presets.licenseLabel")}</b> {spec.license} · <b>{t("app:presets.revisionLabel")}</b> {spec.revision.slice(0,8)} · <b>{t("app:presets.libraryLabel")}</b> {spec.library} / {spec.processor}{spec.parameterCount?" · "+(spec.parameterCount/1e6).toFixed(1)+" M "+t("app:presets.parameters"):""}{spec.inputContract?" · "+spec.inputContract:""}.<br/>{spec.limitations||t("app:presets.defaultLimitations")}</p><button className="secondary" onClick={()=>applyPretrained(spec)}>{t("app:presets.acceptLicenseUse")}</button></article>)}</div>:<p className="muted">{scratchGuidance||t("app:presets.pretrainedEmpty")}</p>}</div>:<div className="preset-view" role="tabpanel"><h3>{t("app:presets.scratchHeading")}</h3>{scratchGuidance&&<p className="muted">{scratchGuidance}</p>}<div className="preset-list">{builtinPresets.map(preset=>{const info=presetI18n(preset);return <article key={preset.id}><div><Sparkles/><span><strong>{info.name}</strong><small>{info.description}</small></span></div><p><b>{t("app:presets.benefitLabel")}</b> {info.benefit}. <b>{t("app:presets.tradeoffLabel")}</b> {info.tradeoff}.</p><button className="secondary" onClick={()=>applyPreset(preset)}>{t("app:presets.apply")}</button></article>;})}{customPresets.map(preset=><article key={preset.id}><div><Save/><span><strong>{preset.name}</strong><small>{preset.description}</small></span></div><p><b>{t("app:presets.benefitLabel")}</b> {t("app:customPreset.benefit")}. <b>{t("app:presets.tradeoffLabel")}</b> {t("app:customPreset.tradeoff")}.</p><button className="secondary" onClick={()=>applyPreset(preset)}>{t("app:presets.apply")}</button></article>)}</div><div className="save-preset"><label>{t("app:savePreset.label")}<input value={presetName} onChange={event=>setPresetName(event.target.value)} placeholder={t("app:savePreset.placeholder")}/></label><button className="primary" disabled={!presetName.trim()} onClick={()=>savePreset()}><Save/> {t("common:save")}</button></div></div>}</section></div>}
    </div>
  );
}

type LibraryPresetPreview={name:string;description:string;nodes:ModelNodeT[];edges:ModelEdge[];benefit?:string;tradeoff?:string};

function LibraryHoverInfo({block,pinned,preset,selected,architecture,task,dataset,onClose}:{block:(typeof BLOCKS)[Architecture][number]|null;pinned:boolean;preset:LibraryPresetPreview|null;selected:ModelNodeT|null;architecture:Architecture;task:TaskId;dataset:DatasetSummary|null;onClose:()=>void}){
  const {t}=useTranslation(["app","common","catalog"]);
  if(block){
    const label=t(`catalog:blocks.${block.type}.label`,{defaultValue:block.label});
    const description=t(`catalog:blocks.${block.type}.description`,{defaultValue:BLOCK_INFO[block.type]?.description||t("app:blockInfo.fallbackDescription")});
    const usage=t(`catalog:blocks.${block.type}.usage`,{defaultValue:BLOCK_INFO[block.type]?.usage||t("app:blockInfo.fallbackUsage")});
    const diagram=t(`catalog:blocks.${block.type}.diagram`,{returnObjects:true,defaultValue:BLOCK_INFO[block.type]?.diagram||[t("common:input"),label,t("common:output")]}) as string[];
    const calculated=previewOperationOutput(block.type,block.defaults,selected?.data.shape);
    return <div className={`block-info-popover ${pinned?"pinned":"preview"}`}><div className="block-info-head"><span className="eyebrow">{pinned?t("app:blockInfo.pinnedEyebrow"):t("app:blockInfo.previewEyebrow")}</span>{pinned&&<button className="modal-close" onClick={onClose} aria-label={t("app:blockInfo.closeInfo")}><X/></button>}</div><h2>{label}</h2><p>{description}</p><div className="node-diagram">{diagram.map((part,index)=><div key={`${part}-${index}`}>{index>0&&<ChevronRight/>}<span className={index===1?"operation":""}>{part}</span></div>)}</div><h3>{t("app:blockInfo.usageTitle")}</h3><p>{usage}</p><div className="shape-preview"><span>{t("app:blockInfo.outputPreview")}</span><code>{calculated||diagram.at(-1)||"?"}</code><small>{t(calculated?"app:blockInfo.calculatedPreview":"app:blockInfo.examplePreview")}</small></div><div className="default-properties"><span>{t("app:blockInfo.defaults")}</span><code>{Object.keys(block.defaults).length?JSON.stringify(block.defaults):t("app:blockInfo.noEditableDefaults")}</code></div>{!pinned&&<small className="pin-hint">{t("app:blockInfo.pinHint")}</small>}</div>;
  }
  if(!preset)return null;
  const outputs=preset.nodes.filter(node=>node.data.blockType==="output");
  const outputNames=outputs.map(node=>String(node.data.properties.output_name||node.data.label)).filter(Boolean);
  const targetShape=dataset?.outputShape?.length?`[B, ${dataset.outputShape.join(", ")}]`:"?";
  const outputExample=outputNames.length>1?outputNames.join(" · "):targetShape;
  return <div className="block-info-popover preview"><div className="block-info-head"><span className="eyebrow">{t("app:blockInfo.previewEyebrow")}</span></div><h2>{preset.name}</h2><p>{preset.description}</p><div className="node-diagram"><div><span>{t("app:builder.graph")}</span><ChevronRight/><span className="operation">{preset.nodes.length?t("app:builder.libraryPresets"):t("app:builder.libraryRecipes")}</span><ChevronRight/><span>{outputExample}</span></div></div><div className="shape-preview"><span>{t("app:blockInfo.outputPreview")}</span><code>{outputExample}</code><small>{t("app:blockInfo.examplePreview")}</small></div>{preset.benefit&&<p className="library-preset-benefit"><strong>{t("app:presets.benefitLabel")}</strong> {preset.benefit}{preset.tradeoff?<> · <strong>{t("app:presets.tradeoffLabel")}</strong> {preset.tradeoff}</>:null}</p>}</div>;
}

function Property({ name, label, value, disabled=false, onChange }: { name: string; label?:string; value: string | number | boolean; disabled?:boolean; onChange: (v: string | number | boolean) => void }) {
  const { t } = useTranslation(["app", "common"]);
  return (
    <label className="property">
      <span>{label||name.replaceAll("_", " ")}</span>
      {typeof value === "boolean" ? (
        <button disabled={disabled} className={`toggle ${value ? "on" : ""}`} onClick={() => onChange(!value)} type="button"><i /></button>
      ) : (
        <input disabled={disabled} title={disabled ? t("app:property.autoAdjusted") : undefined} type={typeof value === "number" ? "number" : "text"} step="any" value={String(value)} onChange={e => onChange(typeof value === "number" ? Number(e.target.value) : e.target.value)} />
      )}
    </label>
  );
}

function Training({ task, dataset, models, activeModelId, onSelectModel, runs, activeRunId, training, stopping, accent, onSelectRun, onCloseRun, onAddRun, onTrain, onControl }: { task: TaskId; dataset:DatasetSummary|null; models:ProjectModel[]; activeModelId:string|null; onSelectModel:(id:string)=>void; runs: TrainingRun[]; activeRunId: string; training: boolean; stopping:boolean; accent: Accent; onSelectRun:(id:string)=>void; onCloseRun:(id:string)=>void; onAddRun:()=>void; onTrain: (c: Record<string, number | string | boolean>) => void; onControl:(state:"running"|"paused"|"cancelled")=>void }) {
  const { t } = useTranslation(["app", "common", "catalog"]);
  const activeRun=runs.find(item=>item.id===activeRunId) || runs[0];
  const runningRun=runs.find(item=>item.status==="running"||item.status==="paused");
  const history=activeRun?.history || [];
  const run=activeRun?.result || null;
  const defaultEpochs=task==="text.language_model"?10:task.startsWith("image.segmentation")?40:20;
  const defaultBestCriterion="val_loss";
  const [epochs, setEpochs] = useState(defaultEpochs);
  const [lr, setLr] = useState(0.001);
  const [batch, setBatch] = useState(Number(dataset?.options?.batchSize||32));
  const [optimizer, setOptimizer] = useState("adamw");
  const [bestModelCriterion, setBestModelCriterion] = useState(defaultBestCriterion);
  const [scheduler,setScheduler]=useState("none");
  const [warmup,setWarmup]=useState(0);
  const [earlyStoppingEnabled,setEarlyStoppingEnabled]=useState(false);
  const [patience,setPatience]=useState(5);
  const [accumulation,setAccumulation]=useState(1);
  const [precision,setPrecision]=useState("fp32");
  const [clipNorm,setClipNorm]=useState(5);
  const [lossName,setLossName]=useState("auto");
  const [finetuningRecipe,setFinetuningRecipe]=useState("auto");
  const [backboneLr,setBackboneLr]=useState(0.0001);
  const [headLr,setHeadLr]=useState(0.001);
  const [classWeights,setClassWeights]=useState(false);
  const [balancedSampler,setBalancedSampler]=useState(false);
  const [resumeCheckpoint,setResumeCheckpoint]=useState("");
  const [initialCheckpoint,setInitialCheckpoint]=useState("");
  useEffect(()=>{
    const config=activeRun?.config || {};
    const savedPatience=Math.max(0,Math.floor(Number(config.early_stopping_patience)||0));
    setEpochs(Number(config.epochs || defaultEpochs));setLr(Number(config.learning_rate || .001));setBatch(Number(config.batch_size || dataset?.options?.batchSize || 32));setOptimizer(String(config.optimizer || "adamw"));setBestModelCriterion(String(config.best_model_criterion || defaultBestCriterion));setScheduler(String(config.scheduler||"none"));setWarmup(Number(config.warmup_epochs||0));setEarlyStoppingEnabled(savedPatience>0);setPatience(savedPatience||5);setAccumulation(Number(config.gradient_accumulation||1));setPrecision(String(config.precision||"fp32"));setClipNorm(Number(config.gradient_clip_norm||5));setLossName(String(config.loss||"auto"));setFinetuningRecipe(String(config.finetuning_recipe||"auto"));setBackboneLr(Number(config.backbone_learning_rate||.0001));setHeadLr(Number(config.head_learning_rate||.001));setClassWeights(Boolean(config.class_weights));setBalancedSampler(Boolean(config.balanced_sampler));setResumeCheckpoint(String(config.resume_checkpoint||""));setInitialCheckpoint(String(config.initial_checkpoint||""));
  },[activeRunId,activeRun?.config?.early_stopping_patience,dataset?.options?.batchSize,defaultEpochs,defaultBestCriterion]);
  const currentEpoch=history.at(-1)?.epoch || 0;
  const targetEpochs=Number(activeRun?.config.epochs || epochs || 1);
  const progress=Math.min(100,Math.round(currentEpoch/targetEpochs*100));
  const elapsedSeconds=activeRun?.startedAt ? Math.max(0,Math.round(((activeRun.completedAt ? new Date(activeRun.completedAt).getTime() : Date.now())-new Date(activeRun.startedAt).getTime())/1000)) : 0;
  const etaSeconds=currentEpoch ? Math.max(0,Math.round((elapsedSeconds/currentEpoch)*(targetEpochs-currentEpoch))) : 0;
  const formatTime=(seconds:number)=>seconds<60?t("app:time.seconds", { count: seconds }):t("app:time.minutes", { minutes: Math.floor(seconds/60), seconds: seconds%60 });
  const bestLoss=history.length ? history.reduce((best, point)=>point.valLoss<best.valLoss?point:best,history[0]) : null;
  const metricHigherIsBetter=!new Set<TaskId>(["tabular.regression","tabular.reconstruction","image.regression","image.reconstruction","image.translation","sequence.regression","sequence.forecast"]).has(task);
  const bestMetric=history.length ? history.reduce((best, point)=>(metricHigherIsBetter ? point.metric>best.metric : point.metric<best.metric)?point:best,history[0]) : null;

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
        {runs.map(item=>{const runActive=item.status==="running"||item.status==="paused";const closeTitle=runActive?t("app:training.closeRunActive"):runs.length<=1?t("app:training.closeRunLast"):t("app:training.closeRun");return <div className="closable-tab run-tab" role="presentation" key={item.id}><button role="tab" aria-selected={item.id===activeRunId} className={item.id===activeRunId?"active":""} onClick={()=>onSelectRun(item.id)}><i className={`run-dot ${item.status}`}/><span>{item.name}</span>{item.status==="completed"&&<Check size={12}/>}</button><button className="tab-close" disabled={runs.length<=1||runActive} onClick={event=>{event.stopPropagation();onCloseRun(item.id)}} title={closeTitle} aria-label={closeTitle}><X size={12}/></button></div>})}
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
          </label>
          <div className="early-stopping-setting">
            <div className="early-stopping-toggle-row">
              <span id="training-early-stopping-label">{t("app:training.advanced.earlyStopping")}</span>
              <button type="button" role="switch" aria-checked={earlyStoppingEnabled} aria-labelledby="training-early-stopping-label" aria-describedby="training-early-stopping-help" className={`toggle ${earlyStoppingEnabled?"on":""}`} onClick={()=>setEarlyStoppingEnabled(value=>!value)}><i/></button>
            </div>
            <small id="training-early-stopping-help" className="early-stopping-help">{t("app:training.advanced.earlyStoppingHelp")}</small>
            {earlyStoppingEnabled&&<label htmlFor="training-early-stopping-patience">{t("app:training.advanced.earlyStoppingPatience")}<input id="training-early-stopping-patience" aria-describedby="training-early-stopping-help" type="number" min="1" step="1" value={patience} onChange={event=>setPatience(Math.max(1,Math.floor(Number(event.target.value)||1)))}/></label>}
          </div>
          <details className="advanced-training"><summary>{t("app:training.advanced.title")}</summary>
            <label>{t("app:training.advanced.recipe")}<select value={finetuningRecipe} onChange={e=>setFinetuningRecipe(e.target.value)}><option value="auto">{t("app:training.advanced.recipeAuto")}</option><option value="head">{t("app:training.advanced.recipeHead")}</option><option value="gradual">{t("app:training.advanced.recipeGradual")}</option><option value="full">{t("app:training.advanced.recipeFull")}</option><option value="lora">LoRA</option><option value="smoke">{t("app:training.advanced.recipeSmoke")}</option></select></label>
            <label>{t("app:training.advanced.backboneLr")}<input type="number" step="0.00001" value={backboneLr} onChange={e=>setBackboneLr(Number(e.target.value))}/></label>
            <label>{t("app:training.advanced.headLr")}<input type="number" step="0.0001" value={headLr} onChange={e=>setHeadLr(Number(e.target.value))}/></label>
            {task.includes("classification")&&<><label>{t("app:training.advanced.classWeights")}<button type="button" className={`toggle ${classWeights?"on":""}`} onClick={()=>setClassWeights(value=>!value)}><i/></button></label><label>{t("app:training.advanced.balancedSampler")}<button type="button" className={`toggle ${balancedSampler?"on":""}`} onClick={()=>setBalancedSampler(value=>!value)}><i/></button></label></>}
            <label>{t("app:training.advanced.resume")}<input value={resumeCheckpoint} onChange={e=>setResumeCheckpoint(e.target.value)} placeholder={t("app:training.advanced.resumePlaceholder")}/></label>
            <label>{t("app:training.advanced.initialWeights")}<input value={initialCheckpoint} onChange={e=>setInitialCheckpoint(e.target.value)} placeholder={t("app:training.advanced.initialWeightsPlaceholder")}/></label>
            <label>{t("app:training.advanced.loss")}<select value={lossName} onChange={e=>setLossName(e.target.value)}><option value="auto">{t("app:training.advanced.automatic")}</option>{!task.includes("classification")&&<><option value="mse">MSE</option><option value="mae">MAE</option><option value="huber">Huber</option></>}{task.includes("classification")&&<option value="focal">Focal</option>}</select></label>
            <label>{t("app:training.advanced.scheduler")}<select value={scheduler} onChange={e=>setScheduler(e.target.value)}><option value="none">{t("app:training.advanced.none")}</option><option value="cosine">{t("app:training.advanced.cosineWarmup")}</option><option value="step">{t("app:training.advanced.stepSchedule")}</option><option value="plateau">{t("app:training.advanced.reducePlateau")}</option></select></label>
            <label>{t("app:training.advanced.warmup")}<input type="number" min="0" value={warmup} onChange={e=>setWarmup(Number(e.target.value))}/></label>
            <label>{t("app:training.advanced.accumulation")}<input type="number" min="1" value={accumulation} onChange={e=>setAccumulation(Number(e.target.value))}/></label>
            <label>{t("app:training.advanced.gradientClip")}<input type="number" min="0" step="0.5" value={clipNorm} onChange={e=>setClipNorm(Number(e.target.value))}/></label>
            <label>{t("app:training.advanced.precision")}<select value={precision} onChange={e=>setPrecision(e.target.value)}><option value="fp32">FP32</option><option value="fp16">FP16 (CUDA)</option><option value="bf16">BF16 (CUDA)</option></select></label>
          </details>
          <label>{t("app:training.bestModelCriterionLabel")}
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
          <div className="run-config-footer">{training?<div className="training-control-actions"><button className="stop-training" disabled={stopping} onClick={()=>onControl("cancelled")}><Square size={15} fill="currentColor"/> {stopping?t("app:training.stopping"):t("app:training.stop")}</button><button className="secondary pause-training" disabled={stopping||!runningRun} onClick={()=>onControl(runningRun?.status==="paused"?"running":"paused")}>{runningRun?.status==="paused"?<Play size={15}/>:<Pause size={15}/>} {runningRun?.status==="paused"?t("app:training.resume"):t("app:training.pause")}</button></div>:<button className="primary full" onClick={() => onTrain({ epochs, learning_rate: lr, backbone_learning_rate:backboneLr, head_learning_rate:headLr, batch_size: batch, optimizer, best_model_criterion: bestModelCriterion, scheduler, warmup_epochs:warmup, early_stopping_patience:earlyStoppingEnabled?patience:0, gradient_accumulation:accumulation, gradient_clip_norm:clipNorm, precision, loss:lossName, finetuning_recipe:finetuningRecipe, class_weights:classWeights, balanced_sampler:balancedSampler, resume_checkpoint:resumeCheckpoint, initial_checkpoint:initialCheckpoint })}>{activeRun?.status==="completed" ? <><Play/> {t("app:training.retrainButton")}</> : <><Play /> {t("app:training.startButton")}</>}</button>}{activeRun?.status==="completed"&&!training&&<small>{t("app:training.newRunHint")}</small>}</div>
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

const mainLogoUrl = new URL("../../assets/Logo_Icon.png", import.meta.url).href;

function AppLogo({className}:{className:string}){return <span className={`app-logo ${className}`} role="img" aria-label="ModelBuilder" style={{maskImage:`url(${mainLogoUrl})`,WebkitMaskImage:`url(${mainLogoUrl})`}}/>}

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
  return <div className="hub"><header className="hub-header"><div className="brand" title={t("common:appName")} aria-label={t("common:appName")}><AppLogo className="main-screen-logo"/><strong>{t("common:appName")}</strong></div><button className="primary" onClick={()=>setCreating(true)}><Plus size={17}/>{t("app:hub.newProjectButton")}</button><AppearanceMenu theme={theme} setTheme={setTheme} accent={accent} setAccent={setAccent}/></header>
    <div className="hub-content">
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
  if(totalVramBytes<=0)return{text:`${estimatedMB} MB estimados · memoria del sistema`,percent:0};
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
  const totalVram = gpuInfo?.total_vram ?? 0;
  const gpuName = gpuInfo?.name || "CPU";
  const vramEst = computeVramEstimate(nodes, dataset, totalVram, t);

  return <aside className="sidebar">
    <div className="sidebar-top">
      <button className="brand sidebar-brand" onClick={onHome} title={t("app:sidebar.homeTitle")} aria-label={t("app:sidebar.homeTitle")}><AppLogo className="sidebar-app-logo"/></button>
      <AppearanceMenu theme={theme} setTheme={setTheme} accent={accent} setAccent={setAccent}/>
    </div>
    <nav>{steps.map((s,i) => { const Icon=s.icon; return <button key={s.id} className={step===s.id?"active":""} onClick={()=>setStep(s.id)}><span className="step-index">{completed[s.id]?<Check size={13}/>:i+1}</span><Icon size={16}/><span>{s.label}</span></button>; })}</nav>
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

function ModelTask({ architecture, task, runtimeCatalog, onChoose, onContinue }: { architecture:Architecture|null; task:TaskId|null; runtimeCatalog:RuntimeCatalog|null; onChoose:(a:Architecture,t:TaskId)=>void; onContinue:()=>void }) {
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
  const detail={...ARCHITECTURES[preview],tasks:runtimeCatalog?.architectures[preview]?.tasks||ARCHITECTURES[preview].tasks};
  const DetailIcon=ARCHITECTURE_ICONS[preview];
  const activeTask=architecture===preview?task:null;
  return <div className="page model-page model-catalog-page">
    <div className="model-catalog-header"><div><span className="eyebrow">{t("app:modelTask.stepEyebrow")}</span><h2>{t("app:modelTask.catalogTitle")}</h2><p>{t("app:modelTask.catalogSubtitle")}</p></div><div className="catalog-tools"><label><Search/><input value={query} onChange={event=>setQuery(event.target.value)} placeholder={t("app:modelTask.searchPlaceholder")}/><kbd>⌘ K</kbd></label><button><ArrowUpDown/>{t("app:modelTask.popularFirst")}<ChevronRight/></button></div></div>
    <div className="model-catalog-layout">
      <section className="family-panel"><header><h3>{t("app:modelTask.familiesTitle")}</h3><p>{t("app:modelTask.familiesSubtitle")}</p></header><div className="family-list">{families.map(item=>{const Icon=item.icon;return <button key={item.id} className={family===item.id?"active":""} onClick={()=>{setFamily(item.id);if(!item.architectures.includes(preview))setPreview(item.architectures[0])}}><span><Icon/></span><div><strong>{item.label}</strong><small>{item.hint}</small></div><b>{item.architectures.length}</b></button>})}</div></section>
      <section className="architecture-list-panel"><header><h3>{t("app:modelTask.architecturesTitle")}</h3><p>{t("app:modelTask.architectureCount",{count:visibleArchitectures.length})}</p></header><div className="catalog-architecture-list">{visibleArchitectures.map(([id,a],index)=>{const Icon=ARCHITECTURE_ICONS[id];return <button key={id} className={preview===id?"active":""} onClick={()=>setPreview(id)}><span><Icon/></span><div><strong>{t(`catalog:architectures.${id}.name`)}</strong><small>{t(`catalog:architectures.${id}.description`)}</small><em>{t(`catalog:tasks.${a.tasks[0]}.modality`)}</em></div><ChevronRight/></button>})}</div></section>
      <section className="architecture-detail-panel"><div className="detail-scroll"><header><span className="detail-arch-icon"><DetailIcon/></span><div><h2>{t(`catalog:architectures.${preview}.name`)}</h2><span className="eyebrow">{t("app:modelTask.architectureEyebrow")}</span><p>{t(`catalog:architectures.${preview}.description`)}</p></div></header><div className="architecture-overview"><div><p>{t("app:modelTask.overviewText",{architecture:t(`catalog:architectures.${preview}.name`)})}</p><div className="detail-tags"><span>{t(`catalog:tasks.${detail.tasks[0]}.modality`)}</span><span>{t("app:modelTask.scalable")}</span><span>{t("app:modelTask.localReady")}</span></div></div><div className="mini-architecture-diagram"><small>{t("app:modelTask.inputSequence")}</small><i/><b>{t(`catalog:architectures.${preview}.name`)}</b><i/><small>{t("app:modelTask.representation")}</small><div>{[1,2,3,4,5,6].map(n=><span key={n}/>)}</div></div></div><div className="available-tasks"><h3>{t("app:modelTask.availableTasks")}</h3><p>{t("app:modelTask.availableTasksHint")}</p><div>{detail.tasks.map(taskId=><button key={taskId} className={activeTask===taskId?"selected":""} onClick={()=>onChoose(preview,taskId)}><span><FileText/></span><div><strong>{t(`catalog:tasks.${taskId}.name`)}</strong><small>{t(`catalog:tasks.${taskId}.input`)} → {t(`catalog:tasks.${taskId}.output`)}</small></div><ChevronRight/></button>)}</div></div>{activeTask&&<div className="detail-contract"><div><small>{t("app:contract.compatibleInput")}</small><strong>{t(`catalog:tasks.${activeTask}.format`)}</strong></div><div><small>{t("app:contract.mainMetric")}</small><strong>{t(`catalog:tasks.${activeTask}.metric`)}</strong></div><div><small>{t("app:contract.modality")}</small><strong>{t(`catalog:tasks.${activeTask}.modality`)}</strong></div></div>}</div><footer><span>{activeTask?t(`catalog:tasks.${activeTask}.name`):t("app:modelTask.chooseTaskHint")}</span><button className="primary" disabled={!activeTask} onClick={onContinue}>{t("app:modelTask.continueWith",{task:activeTask?t(`catalog:tasks.${activeTask}.name`):t("app:modelTask.taskFallback")})}<ChevronRight/></button></footer></section>
    </div>
  </div>;
}

function DataStep({ project, task, dataset, dataContinue, setNotice, onDownload, onKaggle, onImport, onImportFiles, onChooseAnother, onApplyPipeline, onSplitsChange }: {
  project: Project | null;
  task: TaskId | null;
  dataset: DatasetSummary | null;
  dataContinue: React.MutableRefObject<() => Promise<boolean>>;
  setNotice: (message: string) => void;
  onDownload: (datasetId: string, opts?: DatasetOptions) => Promise<void>;
  onKaggle: (handle: string, opts: DatasetOptions) => Promise<void>;
  onImport: (path: string, opts?: DatasetOptions) => Promise<void>;
  onImportFiles: (files: File[], opts?: DatasetOptions) => Promise<void>;
  onChooseAnother: () => void;
  onApplyPipeline: (pipeline: DataPipeline) => Promise<void>;
  onSplitsChange: (splits: DatasetSummary["splits"]) => void;
}) {
  const { t } = useTranslation(["app", "common", "catalog"]);
  const [path, setPath] = useState(dataset?.sourcePath || "");
  const [sourceChoice,setSourceChoice]=useState<"kaggle"|"huggingface"|"local"|null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(false);
  const [loadingLabel, setLoadingLabel] = useState("");
  const [targetColumn, setTargetColumn] = useState("");
  const [csvColumns, setCsvColumns] = useState<string[]>([]);
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [pipelineDraft, setPipelineDraft] = useState<DataPipeline | null>(null);
  const [pipelineDirty, setPipelineDirty] = useState(false);
  const [dataView,setDataView]=useState<"preview"|"charts"|"info">("preview");
  const [analytics,setAnalytics]=useState<DatasetAnalytics|undefined>(dataset?.analytics);
  const [catalog,setCatalog]=useState<HuggingFaceDataset[]>([]);
  const [catalogError,setCatalogError]=useState("");
  const [randomPreview,setRandomPreview]=useState<(DatasetPreview & {sampleIndices?:number[]})|null>(null);
  const [previewLoading,setPreviewLoading]=useState(false);
  const [previewError,setPreviewError]=useState("");
  const previewDatasetKey=useRef("");
  previewDatasetKey.current=`${dataset?.id||""}:${dataset?.revision||0}`;
  useEffect(()=>{setRandomPreview(null);setPreviewError("");setPreviewLoading(false)},[dataset?.id,dataset?.revision]);
  const refreshSamples=async()=>{if(!dataset||!task||previewLoading)return;const key=`${dataset.id}:${dataset.revision||0}`;setPreviewLoading(true);setPreviewError("");try{const result=await backend<DatasetPreview & {sampleIndices?:number[]}>("data.preview",{dataset,task_id:task});if(previewDatasetKey.current===key)setRandomPreview(result)}catch(error){if(previewDatasetKey.current===key)setPreviewError(translateBackendError(String(error)))}finally{if(previewDatasetKey.current===key)setPreviewLoading(false)}};
  const shownPreview=randomPreview||dataset?.preview;
  const isImage = task ? (task.startsWith("image") || task.includes("segmentation")) : false;
  const isText = task?.startsWith("text.") || false;

  const [channels, setChannels] = useState<number>(dataset?.options?.channels || (dataset?.inputShape?.[0] === 1 ? 1 : 3));
  const [targetChannels, setTargetChannels] = useState<number>(dataset?.options?.targetChannels || 3);
  const [resolution, setResolution] = useState<number>(dataset?.options?.resolution || (isImage && dataset?.inputShape?.[1] ? dataset.inputShape[1] : 256));
  const [normalization, setNormalization] = useState<"none" | "standard" | "minmax" | "minmax_sym">(dataset?.options?.normalization || "none");
  const [batchSize, setBatchSize] = useState<number>(dataset?.options?.batchSize || 32);
  const [maxLength,setMaxLength]=useState<number>(dataset?.options?.max_length || 64);
  const [vocabSize,setVocabSize]=useState<number>(dataset?.options?.vocab_size || 2048);
  const resetSource=()=>{setSourceChoice(null);onChooseAnother()};
  const onPipelineDraftChange = useCallback((pipeline: DataPipeline, dirty: boolean) => {
    setPipelineDraft(pipeline);
    setPipelineDirty(dirty);
  }, []);

  useEffect(() => {
    if (!dataset || !task) { setPipelineDraft(null); setPipelineDirty(false); return; }
    setPipelineDraft(dataset.pipeline || defaultDataPipeline(task, dataset));
    setPipelineDirty(!dataset.pipeline);
  }, [dataset?.id, dataset?.revision, task]);

  useEffect(() => {
    if (dataset) {
      if (dataset.options?.channels) setChannels(dataset.options.channels);
      if (dataset.options?.targetChannels) setTargetChannels(dataset.options.targetChannels);
      else if (dataset.inputShape?.[0]) setChannels(dataset.inputShape[0] === 1 ? 1 : 3);

      if (dataset.options?.resolution) setResolution(dataset.options.resolution);
      else if (isImage && dataset.inputShape?.[1]) setResolution(dataset.inputShape[1]);

      if (dataset.options?.normalization) setNormalization(dataset.options.normalization);
      if (dataset.options?.batchSize) setBatchSize(dataset.options.batchSize);
    }
  }, [dataset, isImage]);
  useEffect(()=>{if(!dataset||!task)return;setAnalytics(dataset.analytics);if(!dataset.analytics)backend<DatasetAnalytics>("data.analytics",{dataset,task_id:task}).then(setAnalytics).catch(()=>{})},[dataset?.id,dataset?.revision,task]);
  useEffect(()=>{
    if(dataset?.source!=="huggingface" && dataset?.source!=="kaggle")return;
    backend<{exists:boolean}>("data.exists",{dataset}).then(result=>{if(!result.exists)resetSource()}).catch(()=>resetSource());
  },[dataset?.id,dataset?.revision]);
  useEffect(()=>{
    if(!task||!project){setCatalog([]);return}
    backend<HuggingFaceDataset[]>("data.catalog",{project,task_id:task}).then(items=>{setCatalog(items);setCatalogError("")}).catch(error=>setCatalogError(translateBackendError(String(error))));
  },[task,project?.id,dataset?.id]);

  if (!task) return <Blocked message={t("app:blocked.selectTaskArchitecture")} />;

  const trainCount = dataset ? (dataset.splitCounts?.train ?? Math.round(dataset.samples * dataset.splits.train / 100)) : 0;
  const validationCount = dataset ? (dataset.splitCounts?.validation ?? Math.round(dataset.samples * dataset.splits.validation / 100)) : 0;
  const testCount = dataset ? (dataset.splitCounts?.test ?? dataset.samples - trainCount - validationCount) : 0;
  const formatImageSize=(size:{width:number;height:number}|null|undefined)=>size?`${size.width.toLocaleString()} × ${size.height.toLocaleString()} px`:t("app:data.sourceResolutionUnknown");

  const currentOpts: DatasetOptions = {
    ...(isImage ? { channels, resolution, ...(task==="image.translation"?{target_channels:targetChannels}:{}) } : isText ? {max_length:maxLength,vocab_size:vocabSize} : { normalization }),
    ...(task.startsWith("tabular") && targetColumn.trim() ? { target_column: targetColumn.trim() } : {}),
    ...(csvColumns.length ? { feature_columns: selectedFeatures } : {}),
    batchSize
  };

  const runLoading = async (label:string, operation:()=>Promise<void>) => {
    setLoading(true);setLoadingLabel(label);
    try { await operation(); }
    finally { window.setTimeout(()=>setLoading(false),300); }
  };
  useEffect(() => {
    dataContinue.current = async () => {
      if (!dataset || !pipelineDraft || (!pipelineDirty && !dataset.splitPending && dataset.pipeline)) return true;
      try {
        await runLoading(t("app:data.applyingPipeline"), () => onApplyPipeline(pipelineDraft));
        return true;
      } catch {
        return false;
      }
    };
    return () => { dataContinue.current = async () => true; };
  }, [dataContinue, dataset, pipelineDraft, pipelineDirty, onApplyPipeline, t]);

  const browseFolder=async()=>{
    if(!isTauri()) { fileInput.current?.click(); return; }
    const selectedPath=isImage ? await pickDatasetDirectory() : await pickDatasetFile();
    if(selectedPath){setPath(selectedPath);setCsvColumns([]);setPendingFiles([]);setSelectedFeatures([]);setTargetColumn("");}
  };
  useEffect(() => {
    if (!fileInput.current) return;
    if (isImage) fileInput.current.setAttribute("webkitdirectory", "");
    else fileInput.current.removeAttribute("webkitdirectory");
  }, [isImage]);

  const importSelectedFiles = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    event.target.value = "";
    if (!files.length) return;
    if (task.startsWith("tabular") && files.length === 1) {
      const header = (await files[0].slice(0, 128 * 1024).text()).split(/\r?\n/, 1)[0];
      const columns = parseCsvHeader(header);
      if (columns.length < 2) { setNotice(t("app:data.csvHeaderInvalid")); return; }
      setPendingFiles(files); setCsvColumns(columns);
      const preferred = task === "tabular.reconstruction" ? "" : columns.includes("target") ? "target" : columns.at(-1) || "";
      setTargetColumn(preferred); setSelectedFeatures(columns.filter(column => column !== preferred));
      return;
    }
    await runLoading(t("app:data.importLoading"), () => onImportFiles(files, currentOpts));
  };

  const inspectOrImportPath = async () => {
    if (!path && !pendingFiles.length) return;
    if (task.startsWith("tabular") && !csvColumns.length) {
      setLoading(true); setLoadingLabel(t("app:data.inspectingColumns"));
      try {
        const response = await backend<{ columns: string[] }>("data.inspect_csv", { path });
        setCsvColumns(response.columns);
        const preferred = task === "tabular.reconstruction" ? "" : response.columns.includes("target") ? "target" : response.columns.at(-1) || "";
        setTargetColumn(preferred); setSelectedFeatures(response.columns.filter(column => column !== preferred));
      } catch (error) {
        setNotice(translateBackendError(String(error)));
      } finally {
        window.setTimeout(() => setLoading(false), 300);
      }
      return;
    }
    await runLoading(t("app:data.importLoading"), () => pendingFiles.length ? onImportFiles(pendingFiles, currentOpts) : onImport(path, currentOpts));
  };

  const chooseTargetColumn = (next: string) => {
    const automatic = csvColumns.includes("target") ? "target" : csvColumns.at(-1) || "";
    const previous = targetColumn || automatic;
    const effectiveNext = next || automatic;
    setTargetColumn(next);
    setSelectedFeatures(current => {
      const updated = current.filter(column => column !== effectiveNext);
      if (previous && previous !== effectiveNext && !current.includes(previous)) updated.push(previous);
      return csvColumns.filter(column => updated.includes(column));
    });
  };

  return (
    <div className="page data-page">
      {loading&&<div className="loading-banner"><RefreshCw className="spin"/><span>{loadingLabel}</span><div><i/></div></div>}
      <div className="page-intro">
        <div>
          <span className="eyebrow">{t("app:data.stepEyebrow")}</span>
          <h2>{t("app:data.title")}</h2>
        </div>
      </div>

      {!dataset ? (
        <div className="dataset-source-flow">
          {!sourceChoice ? <section className="source-selection"><span className="eyebrow">{t("app:data.sourceEyebrow")}</span><h3>{t("app:data.sourceTitle")}</h3><p>{t("app:data.sourceSubtitle")}</p><div className="source-choice-grid">
            <button className="source-choice-card" onClick={()=>setSourceChoice("kaggle")}><span><Cloud/></span><strong>{t("app:data.kaggleChoice")}</strong><small>{t("app:data.kaggleChoiceHint")}</small><ChevronRight/></button>
            <button className="source-choice-card" onClick={()=>setSourceChoice("huggingface")}><span><Layers3/></span><strong>{t("app:data.hfChoice")}</strong><small>{t("app:data.hfChoiceHint")}</small><ChevronRight/></button>
            <button className="source-choice-card" onClick={()=>setSourceChoice("local")}><span><Upload/></span><strong>{t("app:data.localChoice")}</strong><small>{t("app:data.localChoiceHint")}</small><ChevronRight/></button>
          </div></section> : <>
            <div className="source-flow-toolbar"><button className="secondary" onClick={()=>setSourceChoice(null)}><ArrowLeft/> {t("app:data.changeSource")}</button><span>{t(`app:data.sourceNames.${sourceChoice}`)}</span></div>
            {sourceChoice==="kaggle"&&(task==="image.classification"?<KaggleImport onImport={onKaggle}/>:<section className="source-detail-card"><Cloud/><h3>{t("app:data.kaggleChoice")}</h3><p>{t("app:data.kaggleUnsupported")}</p></section>)}
            {sourceChoice==="huggingface"&&<section className="hf-catalog"><div className="hf-catalog-heading"><div><Cloud/><span><span className="eyebrow">{t("app:data.hfEyebrow")}</span><h3>{t("app:data.hfTitle")}</h3><p>{t("app:data.hfSubtitle")}</p></span></div></div>
              {catalogError&&<div className="catalog-error"><AlertCircle/>{catalogError}</div>}
              <div className="hf-dataset-list">{catalog.map(item=><article className="hf-dataset-card" key={item.id}>
                <div className="hf-dataset-copy"><div className="hf-dataset-title"><Database/><div><h4>{item.name}</h4><a href={`https://huggingface.co/datasets/${item.repoId}`} target="_blank" rel="noreferrer">{item.repoId}</a></div></div><p>{item.description}</p><div className="hf-dataset-meta"><span>{formatBytes(item.sizeBytes)}</span><span>{item.license}</span>{item.installed?<span className="installed"><CheckCircle2/>{t("app:data.installed")}</span>:item.cached?<span className="cached"><Save/>{t("app:data.cached")}</span>:<span><Cloud/>{t("app:data.remote")}</span>}</div></div>
                <button className="primary" disabled={loading} onClick={()=>runLoading(item.installed?t("app:data.openingDataset"):item.cached?t("app:data.preparingCached"):t("app:data.downloadingDataset",{name:item.name}),()=>onDownload(item.id,item.defaultOptions))}>{item.installed?<><Check/> {t("app:data.useDataset")}</>:item.cached?<><Database/> {t("app:data.prepareDataset")}</>:<><Download/> {t("app:data.downloadDataset")}</>}</button>
              </article>)}</div>
            </section>}
            {sourceChoice==="local"&&<article className="source-card local-source-card">
              <Upload/><h3>{t("app:data.importTitle")}</h3><p>{t("app:data.importSubtitle")}</p><p className="muted dataset-format-hint">{t(TASKS[task].formatKey || "",{defaultValue:TASKS[task].format})}</p>
              <input ref={fileInput} type="file" hidden multiple={isImage} accept={isImage ? ".png,.jpg,.jpeg,.webp,.bmp,.csv,.txt,.json" : isText ? ".txt,.csv,.tsv" : ".csv,.tsv"} onChange={importSelectedFiles} />
              {task==="image.translation"&&<label className="app-field"><span>{t("app:data.targetChannels")}</span><select value={targetChannels} onChange={event=>setTargetChannels(Number(event.target.value))}><option value={1}>{t("app:data.grayscale")}</option><option value={3}>{t("app:data.rgb")}</option></select></label>}
              {task.startsWith("tabular") && <>{task !== "tabular.reconstruction" && <label className="app-field"><span>{t("app:data.targetColumnLabel")}</span>{csvColumns.length ? <select value={targetColumn || "__auto__"} onChange={event=>chooseTargetColumn(event.target.value === "__auto__" ? "" : event.target.value)}><option value="__auto__">{t("app:data.targetAuto")}</option>{csvColumns.map(column=><option key={column} value={column}>{column}</option>)}</select> : <input value={targetColumn} onChange={event=>setTargetColumn(event.target.value)} placeholder={t("app:data.targetColumnPlaceholder")} />}</label>}{csvColumns.length>0&&<fieldset className="column-selection"><legend>{t("app:data.featureColumnsLabel")}</legend>{csvColumns.filter(column=>task==="tabular.reconstruction"||column!==(targetColumn || (csvColumns.includes("target")?"target":csvColumns.at(-1)))).map(column=><label key={column}><input type="checkbox" checked={selectedFeatures.includes(column)} onChange={event=>setSelectedFeatures(current=>event.target.checked?[...current,column]:current.filter(value=>value!==column))}/><span>{column}</span></label>)}</fieldset>}</>}
              <label className="app-field"><span>{t("app:data.localPathLabel")}</span><div className="path-picker"><input value={path} onChange={e => { setPath(e.target.value); setCsvColumns([]); setPendingFiles([]); setSelectedFeatures([]); setTargetColumn(""); }} placeholder={t("app:data.pathPlaceholder")} /><button type="button" className="secondary" onClick={browseFolder}><FolderOpen size={15}/> {t("app:data.browse")}</button></div></label>
              {pendingFiles.length>0&&<p className="muted">{t("app:data.selectedFile",{name:pendingFiles[0].name})}</p>}
              <button className="secondary" disabled={(!path&&!pendingFiles.length)||loading} onClick={inspectOrImportPath}>{task.startsWith("tabular") && !csvColumns.length ? t("app:data.inspectColumns") : t("app:data.inspectImport")}</button>
            </article>}
          </>}
        </div>
      ) : (
        <div className="data-step-layout">
          <div className="data-main">
            <DataPipelineEditor task={task} dataset={dataset} externalDirty={dataset.splitPending} onApply={onApplyPipeline} onDraftChange={onPipelineDraftChange}/>
            <article className="panel data-explorer-panel">
              <div className="data-view-header"><span className="eyebrow">{t("app:data.explorerEyebrow")}</span><div className="data-view-switch" role="tablist" aria-label={t("app:data.viewAria")}><button role="tab" aria-selected={dataView==="info"} className={dataView==="info"?"active":""} onClick={()=>setDataView("info")}>{t("app:data.infoTab")}</button><button role="tab" aria-selected={dataView==="preview"} className={dataView==="preview"?"active":""} onClick={()=>setDataView("preview")}>{t("app:data.samplesTab")}</button><button role="tab" aria-selected={dataView==="charts"} className={dataView==="charts"?"active":""} onClick={()=>setDataView("charts")}>{t("app:data.chartsTab")}</button>{dataView==="preview"&&<button className="secondary" onClick={refreshSamples} disabled={previewLoading} title={t(previewLoading?"app:data.loadingSamples":"app:data.randomSamples")} aria-label={t("app:data.randomSamples")}><Dice5 size={16}/></button>}</div></div>
              {previewError&&<p role="alert" className="source-resolution-unavailable">{previewError}</p>}
              {dataView==="info"&&<div className="source-resolution-inline" aria-label={t("app:data.sourceResolutionTitle")}>
                <div><span className="eyebrow">{t("app:data.sourceResolutionTitle")}</span><p>{t("app:data.sourceResolutionDescription")}</p></div>
                {dataset.imageResolution&&dataset.imageResolution.count>0?<div className="source-resolution-values">
                  <span><small>{t("app:data.resolutionMin")}</small><strong>{formatImageSize(dataset.imageResolution.min)}</strong></span>
                  <span><small>{t("app:data.resolutionMean")}</small><strong>{formatImageSize(dataset.imageResolution.mean)}</strong></span>
                  <span><small>{t("app:data.resolutionMax")}</small><strong>{formatImageSize(dataset.imageResolution.max)}</strong></span>
                  <span className="source-resolution-note"><small>{t("app:data.nonSquareImages")}</small><strong>{dataset.imageResolution.nonSquareCount.toLocaleString()} / {dataset.imageResolution.count.toLocaleString()}</strong></span>
                  {isImage&&<span className="source-resolution-model"><small>{t("app:data.modelInputSize")}</small><strong>{dataset.inputShape.join(" × ")}</strong></span>}
                  {dataset.imageResolution.unavailable>0&&<small className="source-resolution-unavailable">{t("app:data.sourceResolutionUnavailable",{count:dataset.imageResolution.unavailable})}</small>}
                </div>:<p className="source-resolution-empty">{dataset.imageResolution?.unavailable?t("app:data.sourceResolutionUnavailable",{count:dataset.imageResolution.unavailable}):isImage?t("app:data.sourceResolutionLegacyMissing"):t("app:data.noImages")}</p>}
              </div>}

              {dataView==="info"?null:dataView==="charts"?<DataCharts analytics={analytics} task={task}/>:shownPreview?.type === "image" && shownPreview.items?.length ? (
                <div className="preview-image-grid">
                  {shownPreview.items.slice(0,4).map((item, i) => (
                    <div className="preview-image-card" key={i}>
                      <div className={`dataset-image-pair ${item.targetUrl ? "with-target" : ""}`}><figure><img src={item.url} alt={item.label || t("app:data.sampleLabel", { number: i + 1 })} /><figcaption>{t("app:data.inputCaption")}</figcaption></figure>{item.targetUrl&&<figure><img src={item.targetUrl} alt={item.targetLabel || t("app:data.targetCaption")} /><figcaption>{item.targetLabel || t("app:data.outputCaption")}</figcaption></figure>}</div>
                      <span className="preview-image-label">{t("app:data.sampleLabel", { number: (randomPreview?.sampleIndices?.[i] ?? i) + 1 })}</span>
                    </div>
                  ))}
                </div>
              ) : shownPreview?.type === "text" && shownPreview.items?.length ? (
                <div className="preview-table-wrap preview-text-wrap">
                  <table className="preview-table preview-text-table">
                    <thead><tr><th>{t("app:data.samplesTab")}</th></tr></thead>
                    <tbody>{shownPreview.items.map((item, i) => <tr key={i}><td>{item.text}</td></tr>)}</tbody>
                  </table>
                </div>
              ) : shownPreview?.type === "tabular" && shownPreview.items?.length ? (
                <div className="preview-table-wrap">
                  <table className="preview-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        {shownPreview.columns?.map((c, i) => <th key={i}>{c}</th>)}
                        <th>Target</th>
                      </tr>
                    </thead>
                    <tbody>
                      {shownPreview.items.map((item, i) => (
                        <tr key={i}>
                          <td>{(randomPreview?.sampleIndices?.[i] ?? i) + 1}</td>
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

            <article className="panel data-split-panel">
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

          <aside className="data-sidebar"><div className="panel data-config-panel"><span className="eyebrow">{t("app:data.nodePropertiesEyebrow")}</span>{dataset.source === "kaggle" && dataset.importReport && <details className="kaggle-provenance"><summary>Kaggle · {dataset.kaggleHandle}</summary><KaggleReport report={dataset.importReport}/></details>}{dataset.source==="huggingface"&&<div className="active-hf-source"><Database/><span><strong>{catalog.find(item=>item.id===dataset.catalogId)?.name||dataset.catalogId}</strong><small>{dataset.repoId}</small><small>{dataset.cacheStatus==="project"?t("app:data.loadedFromProject"):dataset.cacheStatus==="disk"?t("app:data.loadedFromCache"):t("app:data.downloadComplete")}</small></span></div>}<button className="secondary dataset-change-button" onClick={resetSource}><Database/> {t("app:data.changeDataset")}</button><div id="data-node-inspector"/>{dataset.classes&&<div className="data-classes-section"><div className="data-classes-heading"><span className="eyebrow">{t("app:data.classesEyebrow", { count: dataset.classes.length })}</span><small>{t("common:samples")}: <strong>{dataset.samples.toLocaleString()}</strong></small></div><div className="class-tags">{dataset.classes.slice(0,20).map(c=><span key={c}>{c}</span>)}{dataset.classes.length>20&&<span className="class-tags-more" aria-label={`${dataset.classes.length-20} more classes`}>...</span>}</div></div>}</div></aside>
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
  const isImageClassification = task === "image.classification";
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
  const classificationTrueLabel = isImageClassification && result?.trueLabel !== undefined && result.trueLabel !== null
    ? String(result.trueLabel)
    : null;
  const predictionClassIndex = isImageClassification && prediction !== undefined
    ? (dataset.classes ?? []).findIndex((className) => String(className).trim().toLocaleLowerCase() === String(prediction).trim().toLocaleLowerCase())
    : -1;
  const predictedProbability = predictionClassIndex >= 0 && Array.isArray(result?.probabilities)
    ? Number(result.probabilities[predictionClassIndex])
    : null;
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

      <article className={`panel inference-card ${isSegmentation?"segmentation-inference":""} ${!isImageClassification&&comparisonAvailable?(predictionCorrect?"correct":"incorrect"):""}`}>
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
          {result ? <div className={`inference-results-inline ${isImageClassification ? "image-classification-result" : ""}`}>
          {isImageClassification ? result.error ? <div className="inference-error-state"><AlertCircle/><span><strong>{t("app:inference.cannotCalculate")}</strong><small>{String(result.error)}</small></span></div> : <div className="classification-result-content">
            {inputPreview && <figure className="classification-result-image"><figcaption>{t("app:inference.inputImageCaption")}</figcaption><img src={inputPreview} alt={t("app:inference.inputImageCaption")}/></figure>}
            <div className="classification-result-details">
              <div className="classification-result-field">
                <span>{t("app:inference.predictedClassLabel")}</span>
                <strong>{String(result.prediction ?? t("app:inference.noResult"))}</strong>
                {predictedProbability !== null && Number.isFinite(predictedProbability) && <small>{t("app:inference.predictionProbability", { probability: `${(predictedProbability * 100).toFixed(1)}%` })}</small>}
              </div>
              {classificationTrueLabel !== null && <div className="classification-result-field">
                <span>{t("app:inference.trueClassLabel")}</span>
                <strong>{classificationTrueLabel}</strong>
              </div>}
            </div>
          </div> : <>
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

  const classes = task.includes("classification") || segmentation || task==="image.detection"
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
    : task==="image.translation"
      ? [options?.targetChannels||3,resolution,resolution]
      : task==="image.localization"
        ? [4]
      : task==="image.detection"
        ? [classes?.length||3]
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

function mockPoint(epoch:number,task:TaskId):MetricPoint { const train=1.4*Math.exp(-epoch/5)+.08+Math.sin(epoch)*.02; const val=1.5*Math.exp(-epoch/5.6)+.1+Math.sin(epoch*.7)*.025; const high=task.includes("classification")||task.includes("segmentation")||task==="image.localization"||task==="image.detection"; return {epoch,trainLoss:+train.toFixed(4),valLoss:+val.toFixed(4),metric:+(high?(1-val/1.8):val*.65).toFixed(4)} }
