import { memo } from "react";
import { useTranslation } from "react-i18next";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Box, CircleAlert, Trash2 } from "lucide-react";
import type { ModelNode as ModelNodeType } from "./types";

function ModelNodeView({ id, data, selected }: NodeProps<ModelNodeType>) {
  const { t } = useTranslation("charts");
  const main = Object.entries(data.properties)[0];
  const isEssential = data.blockType === "input" || data.blockType === "output";

  return (
    <div className={`model-node ${selected ? "selected" : ""} ${data.error ? "invalid" : ""}`}>
      {!isEssential && data.onDelete && (
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
      {data.blockType !== "input" && <Handle type="target" position={Position.Left} />}
      <div className="node-kicker"><Box size={11} /> {data.blockType}</div>
      <div className="node-title">{data.label}</div>
      {main && <div className="node-property">{main[0]} <strong>{String(main[1])}</strong></div>}
      <div className="node-shape">{data.shape || t("charts:shapePlaceholder")}</div>
      {data.error && <CircleAlert size={14} className="node-error" />}
      {data.blockType !== "output" && <Handle type="source" position={Position.Right} />}
    </div>
  );
}
export default memo(ModelNodeView);
