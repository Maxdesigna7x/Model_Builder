import { memo } from "react";
import { useTranslation } from "react-i18next";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Box, ChevronDown, ChevronRight, CircleAlert, Cloud, Flame, LockKeyhole, Snowflake, Trash2 } from "lucide-react";
import type { ModelNode as ModelNodeType } from "./types";

function ModelNodeView({ id, data, selected }: NodeProps<ModelNodeType>) {
  const { t } = useTranslation("charts");
  const main = Object.entries(data.properties)[0];
  const isEssential = data.blockType === "input";
  const isHuggingFace=data.blockType.startsWith("hf_");
  const frozen=Boolean(data.properties.frozen);
  const inputHandles=data.blockType==="concat"||data.blockType==="add"?(data.inputHandles as string[]|undefined)||["input_0","input_1"]:[];
  const outputHandles=data.blockType==="multi_scale_cnn"?Array.from({length:Math.max(2,Math.min(4,Number(data.properties.stages||3)))},(_,index)=>`p${index+2}`):data.blockType==="hf_vision_backbone"&&Boolean(data.properties.expose_feature_maps)?Array.from({length:Math.max(1,Math.min(4,Number(data.properties.feature_count||3)))},(_,index)=>`map_${index}`):[];

  return (
    <div className={`model-node ${selected ? "selected" : ""} ${data.error ? "invalid" : ""}`}>
      {!isEssential && data.canDelete !== false && data.onDelete && (
        <button
          className="node-delete-btn"
          onClick={(e) => {
            e.stopPropagation();
            data.onDelete?.(id);
          }}
          title={t("charts:removeBlock")}
        >
          <Trash2 size={11} />
        </button>
      )}
      {data.blockType !== "input" && (inputHandles.length? <><Handle type="target" id="target" position={Position.Left} style={{opacity:0,pointerEvents:"none"}}/>{inputHandles.map((handle,index)=><span key={`${handle}-label`} className="model-port-label input-port" style={{top:`${39+index*22}px`}}>{handle}<Handle type="target" id={handle} position={Position.Left} style={{top:"8px"}} data-port-label={handle}/></span>)}</>:<Handle type="target" position={Position.Left} />)}
      <div className="node-kicker">{isHuggingFace?<Cloud size={11}/>:<Box size={11} />} {data.blockType}</div>
      <div className="node-title">{data.label}</div>
      {(isHuggingFace||frozen)&&<div className="node-weight-state">{data.blockType.includes("processor")?<><LockKeyhole size={10}/> processor fijado</>:<><Snowflake size={10}/> {isHuggingFace?"pretrained · ":""}congelado</>}</div>}
      {data.compound&&<button className="node-expand" onClick={event=>{event.stopPropagation();data.onToggleExpanded?.(id)}}>{data.expanded?<ChevronDown size={10}/>:<ChevronRight size={10}/>} {data.expanded?"Ocultar etapas":"Ver etapas"}</button>}
      {data.compound&&data.expanded&&<div className="node-stages">{data.stages?.map(stage=><span key={stage}>{stage}</span>)}</div>}
      {main && <div className="node-property">{main[0]} <strong>{String(main[1])}</strong></div>}
      <div className="node-shape">{data.shape || t("charts:shapePlaceholder")}</div>
      {data.error && <CircleAlert size={14} className="node-error" />}
      {data.blockType !== "output" && (outputHandles.length?<><Handle type="source" id="source" position={Position.Right} style={{opacity:0,pointerEvents:"none"}}/>{outputHandles.map((handle,index)=><span key={`${handle}-label`} className="model-port-label output-port" style={{top:`${46+index*24}px`}}>{handle}<Handle type="source" id={handle} position={Position.Right} style={{top:"8px"}} data-port-label={handle}/></span>)}</>:<Handle type="source" position={Position.Right} />)}
    </div>
  );
}
export default memo(ModelNodeView);
