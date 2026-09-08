import i18n from "./i18n";
import { ARCHITECTURES, BLOCKS, BLOCK_INFO, TASKS } from "./catalog";
import type { Architecture, TaskId } from "./types";

function findBlock(type: string) {
  for (const arch of Object.keys(BLOCKS) as Architecture[]) {
    const block = BLOCKS[arch].find(b => b.type === type);
    if (block) return block;
  }
  return undefined;
}

export function getArchitectureName(id: Architecture): string {
  return i18n.t(`catalog:architectures.${id}.name`, { defaultValue: ARCHITECTURES[id]?.name ?? id });
}

export function getArchitectureDescription(id: Architecture): string {
  return i18n.t(`catalog:architectures.${id}.description`, { defaultValue: ARCHITECTURES[id]?.description ?? "" });
}

export function getTaskName(id: TaskId): string {
  const task = TASKS[id];
  return task?.nameKey ? i18n.t(task.nameKey, { defaultValue: task.name }) : task?.name ?? id;
}

export function getTaskCategory(id: TaskId): string {
  const task = TASKS[id];
  return task?.categoryKey ? i18n.t(task.categoryKey, { defaultValue: task.category }) : task?.category ?? "";
}

export function getTaskModality(id: TaskId): string {
  const task = TASKS[id];
  return task?.modalityKey ? i18n.t(task.modalityKey, { defaultValue: task.modality }) : task?.modality ?? "";
}

export function getTaskInput(id: TaskId): string {
  const task = TASKS[id];
  return task?.inputKey ? i18n.t(task.inputKey, { defaultValue: task.input }) : task?.input ?? "";
}

export function getTaskOutput(id: TaskId): string {
  const task = TASKS[id];
  return task?.outputKey ? i18n.t(task.outputKey, { defaultValue: task.output }) : task?.output ?? "";
}

export function getTaskMetric(id: TaskId): string {
  const task = TASKS[id];
  return task?.metricKey ? i18n.t(task.metricKey, { defaultValue: task.metric }) : task?.metric ?? "";
}

export function getTaskFormat(id: TaskId): string {
  const task = TASKS[id];
  return task?.formatKey ? i18n.t(task.formatKey, { defaultValue: task.format }) : task?.format ?? "";
}

export function getBlockCategory(category: string): string {
  return i18n.t(`catalog:blockCategories.${category}`, { defaultValue: category });
}

export function getBlockLabel(type: string): string {
  const block = findBlock(type);
  if (block?.labelKey) return i18n.t(block.labelKey, { defaultValue: block.label });
  return block?.label ?? type;
}

export function getBlockDescription(type: string): string {
  return i18n.t(`catalog:blocks.${type}.description`, { defaultValue: BLOCK_INFO[type]?.description ?? "" });
}

export function getBlockUsage(type: string): string {
  return i18n.t(`catalog:blocks.${type}.usage`, { defaultValue: BLOCK_INFO[type]?.usage ?? "" });
}

export function getBlockDiagram(type: string): string[] {
  const fallback = BLOCK_INFO[type]?.diagram ?? ["Input", type, "Output"];
  const translated = i18n.t(`catalog:blocks.${type}.diagram`, { returnObjects: true, defaultValue: fallback }) as string[];
  return Array.isArray(translated) ? translated : fallback;
}

export function getPresetName(architecture: Architecture, id: string): string {
  return i18n.t(`catalog:presets.${architecture}.${id}.name`, { defaultValue: id });
}

export function getPresetDescription(architecture: Architecture, id: string): string {
  return i18n.t(`catalog:presets.${architecture}.${id}.description`, { defaultValue: "" });
}

export function getPresetBenefit(architecture: Architecture, id: string): string {
  return i18n.t(`catalog:presets.${architecture}.${id}.benefit`, { defaultValue: "" });
}

export function getPresetTradeoff(architecture: Architecture, id: string): string {
  return i18n.t(`catalog:presets.${architecture}.${id}.tradeoff`, { defaultValue: "" });
}
