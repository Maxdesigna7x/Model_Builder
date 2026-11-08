import { memo, useLayoutEffect } from "react";
import { useTranslation } from "react-i18next";
import { Handle, Position, useUpdateNodeInternals, type NodeProps } from "@xyflow/react";
import { ChevronDown, ChevronRight, CircleAlert, Flame, LockKeyhole, Snowflake } from "lucide-react";
import "./node-port-hit-area.css";
import { operationIcon } from "./operation-icons";
import OperationDiagram from "./OperationDiagram";
import { TensorShapeSummary } from "./tensor-shape";
import type { ModelNode as ModelNodeType } from "./types";

function ModelNodeView({ id, data, selected }: NodeProps<ModelNodeType>) {
  const { t } = useTranslation(["charts", "app"]);
  const updateNodeInternals=useUpdateNodeInternals();
  const main = Object.entries(data.properties)[0];
  const isEssential = data.blockType === "input";
  const isHuggingFace=data.blockType.startsWith("hf_");
  const OperationIcon=operationIcon(data.blockType);
  const frozen=Boolean(data.properties.frozen);
  const tensorTask=(data.tensorTask as import("./types").TaskId|undefined)||"image.classification";
  const tensorArchitecture=(data.tensorArchitecture as import("./types").Architecture|undefined)||"cnn";
  const inputHandles=data.blockType==="concat"||data.blockType==="add"?(data.inputHandles as string[]|undefined)||["input_0","input_1"]:[];
  const outputHandles=data.blockType==="multi_scale_cnn"?Array.from({length:Math.max(2,Math.min(4,Number(data.properties.stages||3)))},(_,index)=>`p${index+2}`):data.blockType==="hf_vision_backbone"&&Boolean(data.properties.expose_feature_maps)?Array.from({length:Math.max(1,Math.min(4,Number(data.properties.feature_count||3)))},(_,index)=>`map_${index}`):[];
  const layoutDirection=(data.layoutDirection as "horizontal"|"vertical"|undefined)||"horizontal";
  const isVertical=layoutDirection==="vertical";
  const inputPosition=isVertical?Position.Top:Position.Left;
  const outputPosition=isVertical?Position.Bottom:Position.Right;
  const inputHandleKey=inputHandles.join("|");
  const outputHandleKey=outputHandles.join("|");
  const stagesKey=(data.stages||[]).join("|");
  const propertiesKey=JSON.stringify(data.properties);

  useLayoutEffect(()=>{
    updateNodeInternals(id);
  },[id,updateNodeInternals,layoutDirection,data.blockType,data.label,data.shape,data.expanded,stagesKey,inputHandleKey,outputHandleKey,propertiesKey]);

  return (
    <div className={`model-node layout-${layoutDirection} ${selected ? "selected" : ""} ${data.error ? "invalid" : ""}`}>
      {data.blockType !== "input" && (inputHandles.length? <><Handle type="target" id="target" position={inputPosition} style={{opacity:0,pointerEvents:"none"}}/>{inputHandles.map((handle,index)=><span key={`${handle}-label`} className={`model-port-anchor input-port ${layoutDirection}-port`} style={isVertical?{left:`${((index+1)/(inputHandles.length+1))*100}%`,top:0}:{left:0,top:`${39+index*22}px`}}><small className="model-port-name">{handle}</small><Handle className="model-node-port" type="target" id={handle} position={inputPosition} data-port-label={handle}/></span>)}</>:<Handle className="model-node-port" type="target" position={inputPosition} />)}
      <div className="node-kicker"><OperationIcon size={11}/> {data.blockType}</div>
      <div className="node-title">{data.label}</div>
      {(isHuggingFace||frozen)&&<div className="node-weight-state">{data.blockType.includes("processor")?<><LockKeyhole size={10}/> processor fijado</>:<><Snowflake size={10}/> {isHuggingFace?"pretrained · ":""}congelado</>}</div>}
      {data.compound&&<button className="node-expand" onClick={event=>{event.stopPropagation();data.onToggleExpanded?.(id)}}>{data.expanded?<ChevronDown size={10}/>:<ChevronRight size={10}/>} {data.expanded?"Ocultar etapas":"Ver etapas"}</button>}
      {data.compound&&data.expanded&&<div className="node-stages">{data.stages?.map(stage=><span key={stage}>{stage}</span>)}</div>}
      {main && <div className="node-property">{["out_channels","in_channels","base_channels"].includes(main[0])?t(`app:inspector.${main[0]}`):main[0].replaceAll("_"," ")} <strong>{String(main[1])}</strong></div>}
      <OperationDiagram type={data.blockType} properties={data.properties} />
      <TensorShapeSummary shape={data.shape} blockType={data.blockType} task={tensorTask} architecture={tensorArchitecture} properties={data.properties} label={data.label} variant="node" />
      {data.error && <CircleAlert size={14} className="node-error" />}
      {data.blockType !== "output" && (outputHandles.length?<><Handle type="source" id="source" position={outputPosition} style={{opacity:0,pointerEvents:"none"}}/>{outputHandles.map((handle,index)=><span key={`${handle}-label`} className={`model-port-anchor output-port ${layoutDirection}-port`} style={isVertical?{left:`${((index+1)/(outputHandles.length+1))*100}%`,bottom:0}:{right:0,top:`${46+index*24}px`}}><small className="model-port-name">{handle}</small><Handle className="model-node-port" type="source" id={handle} position={outputPosition} data-port-label={handle}/></span>)}</>:<Handle className="model-node-port" type="source" position={outputPosition} />)}
    </div>
  );
}
export default memo(ModelNodeView);
