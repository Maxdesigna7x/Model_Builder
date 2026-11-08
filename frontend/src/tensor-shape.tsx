import { useTranslation } from "react-i18next";
import type { Architecture, TaskId } from "./types";

type ShapePart = { name?: string; dimensions: string[] };
type Axis = { symbol: string; value: string; meaning: string };

const imageMapBlocks = new Set([
  "conv2d", "convtranspose2d", "maxpool2d", "avgpool2d", "adaptiveavgpool2d",
  "resize2d", "batchnorm2d", "dropout2d", "multi_scale_cnn", "hf_vision_backbone",
  "concat", "add", "identity", "input", "output", "reshape"
]);
const channelsFirst1dBlocks = new Set([
  "conv1d", "causalpad1d", "maxpool1d", "avgpool1d", "adaptiveavgpool1d", "batchnorm1d"
]);
const sequenceBlocks = new Set([
  "input", "embedding", "sequence_projection", "positional_encoding", "transformer_encoder",
  "causal_transformer", "lstm", "gru", "rnn", "sequence_pool", "temporal_select", "layernorm"
]);

function parseShapeParts(shape?: string): ShapePart[] {
  if (!shape || shape === "?") return [];
  return shape.split(" · ").flatMap((entry) => {
    const match = entry.trim().match(/^(?:(\w+)\s+)?\[([^\]]+)\]$/);
    return match ? [{ name: match[1], dimensions: match[2].split(",").map(value => value.trim()) }] : [];
  });
}

export function parseTensorShape(shape?: string): ShapePart[] {
  return parseShapeParts(shape);
}

function isImageMap(part: ShapePart, blockType: string, task: TaskId, architecture: Architecture): boolean {
  if (part.dimensions.length !== 4 || part.dimensions[0] !== "B") return false;
  if (architecture === "vit" && blockType !== "input") return false;
  return imageMapBlocks.has(blockType) && (task.startsWith("image.") || ["conv2d", "convtranspose2d", "maxpool2d", "avgpool2d", "adaptiveavgpool2d", "resize2d", "batchnorm2d", "dropout2d", "multi_scale_cnn", "hf_vision_backbone"].includes(blockType));
}

function axesFor(part: ShapePart, blockType: string, task: TaskId, architecture: Architecture, properties: Record<string, unknown>, label: string, t: (key: string, options?: Record<string, unknown>) => string): Axis[] {
  const dimensions = part.dimensions;
  const hasBatch = dimensions[0] === "B";
  const values = hasBatch ? dimensions.slice(1) : dimensions;
  const axes: Axis[] = [];
  if (hasBatch) axes.push({ symbol: "B", value: "B", meaning: t("app:tensorAxes.batch") });
  if (isImageMap(part, blockType, task, architecture)) {
    axes.push(
      { symbol: "C", value: values[0], meaning: t("app:tensorAxes.channels") },
      { symbol: "H", value: values[1], meaning: t("app:tensorAxes.height") },
      { symbol: "W", value: values[2], meaning: t("app:tensorAxes.width") }
    );
    return axes;
  }

  const isText = task.startsWith("text.");
  const isSequence = task.startsWith("sequence.") || architecture === "cnn1d" || architecture === "lstm";
  if (values.length === 1) {
    let symbol = "F";
    let meaning = t("app:tensorAxes.features");
    if (isText && ["input", "embedding"].includes(blockType)) { symbol = "T"; meaning = t("app:tensorAxes.tokens"); }
    else if (blockType === "output" && task.includes("classification")) { symbol = "K"; meaning = t("app:tensorAxes.classes"); }
    else if (blockType === "output" && (task.includes("regression") || task.includes("localization"))) { symbol = "Q"; meaning = t("app:tensorAxes.values"); }
    axes.push({ symbol, value: values[0], meaning });
    return axes;
  }
  if (values.length === 2) {
    if (isSequence && channelsFirst1dBlocks.has(blockType)) {
      axes.push({ symbol: "C", value: values[0], meaning: t("app:tensorAxes.channels") }, { symbol: "T", value: values[1], meaning: t("app:tensorAxes.timeSteps") });
      return axes;
    }
    if (isSequence && blockType === "input" && architecture === "cnn1d" && (/\[(?:C|F),\s*(?:L|T)\]/i.test(label) || values[0] === "C" || values[0] === "F")) {
      axes.push({ symbol: "C", value: values[0], meaning: t("app:tensorAxes.channels") }, { symbol: "T", value: values[1], meaning: t("app:tensorAxes.timeSteps") });
      return axes;
    }
    if (isSequence && blockType === "permute") {
      const permutation = String(properties.dims || "2,1").split(",").map(value => Number(value.trim()) - 1);
      const sourceAxes = [
        { symbol: "T", meaning: t("app:tensorAxes.timeSteps") },
        { symbol: "F", meaning: t("app:tensorAxes.features") }
      ];
      if (permutation.length === values.length && permutation.every(index => sourceAxes[index])) {
        permutation.forEach((sourceIndex, index) => {
          const sourceAxis=sourceAxes[sourceIndex];
          axes.push({symbol:sourceAxis.symbol==="F"&&permutation[0]===1?"C":sourceAxis.symbol,value:values[index],meaning:sourceAxis.symbol==="F"&&permutation[0]===1?t("app:tensorAxes.channels"):sourceAxis.meaning});
        });
        return axes;
      }
    }
    if (isText && task === "text.language_model" && blockType === "output") {
      axes.push({ symbol: "T", value: values[0], meaning: t("app:tensorAxes.tokens") }, { symbol: "V", value: values[1], meaning: t("app:tensorAxes.vocabulary") });
      return axes;
    }
    if (architecture === "vit" || ["patch_embedding", "transformer_encoder", "causal_transformer", "embedding"].includes(blockType)) {
      axes.push({ symbol: "S", value: values[0], meaning: t("app:tensorAxes.tokens") }, { symbol: "D", value: values[1], meaning: t("app:tensorAxes.embedding") });
      return axes;
    }
    if (task === "sequence.forecast" && ["reshape", "output"].includes(blockType)) {
      axes.push({ symbol: "P", value: values[0], meaning: t("app:tensorAxes.horizon") }, { symbol: "Q", value: values[1], meaning: t("app:tensorAxes.values") });
      return axes;
    }
    if (isSequence || sequenceBlocks.has(blockType)) {
      axes.push({ symbol: "T", value: values[0], meaning: t("app:tensorAxes.timeSteps") }, { symbol: "F", value: values[1], meaning: t("app:tensorAxes.features") });
      return axes;
    }
  }
  values.forEach((value, index) => axes.push({ symbol: `D${index + 1}`, value, meaning: t("app:tensorAxes.dimension", { index: index + 1 }) }));
  return axes;
}

export function TensorShapeSummary({ shape, blockType, task, architecture, variant, properties = {}, label = "" }: {
  shape?: string;
  blockType: string;
  task: TaskId;
  architecture: Architecture;
  variant: "node" | "inspector";
  properties?: Record<string, unknown>;
  label?: string;
}) {
  const { t } = useTranslation(["app"]);
  const parts = parseShapeParts(shape);
  if (!parts.length) return <div className={variant === "node" ? "node-shape" : "shape-box"}>{shape || t("app:inspector.shapePlaceholder")}</div>;
  const formatted = parts.map(part => ({ part, axes: axesFor(part, blockType, task, architecture, properties, label, t) }));
  const imageAxes = formatted.find(item => isImageMap(item.part, blockType, task, architecture));
  if (variant === "node") {
    if (!imageAxes) return <div className="node-shape">{shape}</div>;
    return <div className="node-shape node-tensor-summary">
      <div className="node-image-shape">
        <span><small>{t("app:tensorAxes.channelsShort")}</small><strong>{imageAxes.part.dimensions[1]}</strong></span>
        <span><small>{t("app:tensorAxes.resolution")}</small><strong>{imageAxes.part.dimensions[2]} × {imageAxes.part.dimensions[3]}</strong></span>
      </div>
      <code>{shape}</code>
    </div>;
  }
  return <div className="tensor-axis-plain">
    {formatted.map(({ part, axes }, index) => <div key={`${part.name || "tensor"}-${index}`}>
      {part.name && <strong>{part.name}</strong>}
      {axes.map((axis,i)=><div className="tensor-axis-row" key={`${axis.symbol}-${i}`}><code>{axis.value}</code><span>{axis.symbol==="B"?t("app:tensorAxes.batchShort"):axis.meaning}</span></div>)}
    </div>)}
  </div>;
}
