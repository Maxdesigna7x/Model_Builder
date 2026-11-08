import { memo, useLayoutEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Handle, Position, useUpdateNodeInternals, type NodeProps } from '@xyflow/react';
import { Wrench, CircleAlert, Snowflake, Image, Target } from 'lucide-react';
import { operationIcon } from './operation-icons';
import type { ModelNode as ModelNodeType } from './types';

function ModelNodeView({ id, data, selected }: NodeProps<ModelNodeType>) {
  const { t } = useTranslation(['app', 'catalog']);
  const update = useUpdateNodeInternals();
  const Icon = data.blockType === "input" && ["cnn","vit","unet"].includes(String(data.tensorArchitecture)) ? Image : data.blockType === "output" ? Target : operationIcon(data.blockType);
  const inputs = ['concat', 'add'].includes(data.blockType) ? (data.inputHandles as string[] || ['input_0', 'input_1']) : ['target'];
  const outputs = data.blockType === 'multi_scale_cnn'
    ? Array.from({length: Math.max(2, Math.min(4, Number(data.properties.stages || 3)))}, (_, i) => `p${i + 2}`)
    : data.blockType === 'hf_vision_backbone' && data.properties.expose_feature_maps
      ? Array.from({length: Math.max(1, Math.min(4, Number(data.properties.feature_count || 3)))}, (_, i) => `map_${i}`) : ['source'];
  const signature = inputs.join() + outputs.join();
  useLayoutEffect(() => update(id), [id, signature, update]);
  const priority = data.blockType.includes('conv') ? ['out_channels','kernel_size','stride'] : data.blockType === 'linear' ? ['out_features','bias'] : Object.keys(data.properties);
  const summary = priority.filter(key=>data.properties[key] !== undefined && !['output_name','frozen','checkpoint','model_id'].includes(key)).slice(0,3)
    .map(key=>`${t(`app:structured.short.${key}`, {defaultValue:key.replaceAll('_',' ')})}: ${data.properties[key]}`).join(' · ');
  const shape = data.shape?.replace(/^\[|\]$/g,'').replaceAll(', ', ' × ');

  return <div className={`model-node structured-card ${selected ? 'selected' : ''} ${data.error ? 'invalid' : ''}`}>
    {data.blockType !== 'input' && inputs.map((handle, i) => <Handle key={handle} type="target" id={handle} position={Position.Top} isConnectable={false} style={{left: `${(i + 1) / (inputs.length + 1) * 100}%`}} />)}
    <span className="structured-card-icon"><Icon size={21}/></span>
    <div className="structured-card-copy"><div className="node-title">{data.label}{Boolean(data.properties.frozen) && <Snowflake size={12}/>}</div><div className="node-property" title={summary}>{summary || t(`catalog:blocks.${data.blockType}.label`, {defaultValue: data.blockType})}</div></div>
    <span className="card-tensor" title={data.shape || t('app:structured.shapePending')}>{shape || '—'}</span>
    {data.error && <button className="node-repair nodrag" title={t('app:repair.inspect')} aria-label={t('app:repair.inspect')} onClick={e=>{e.stopPropagation();(data.onShowSolutions as (()=>void)|undefined)?.();}}><Wrench size={14}/></button>}
    {data.blockType !== 'output' && outputs.map((handle, i) => <Handle key={handle} type="source" id={handle} position={Position.Bottom} isConnectable={false} style={{left: `${(i + 1) / (outputs.length + 1) * 100}%`}}/>)}
  </div>;
}
export default memo(ModelNodeView);
