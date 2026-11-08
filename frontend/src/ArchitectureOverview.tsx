import { useTranslation } from 'react-i18next';
import type { Architecture } from './types';

const stages: Record<Architecture, string[]> = {
  mlp: ['features','dense','activation','prediction'], cnn1d: ['signal','temporalConv','pooling','prediction'],
  lstm: ['sequence','memory','state','prediction'], transformer: ['sequence','embedding','attention','prediction'],
  transformer_causal: ['tokens','embedding','maskedAttention','tokens'], cnn: ['image','convolution','pooling','prediction'],
  vit: ['image','patches','attention','prediction'], unet: ['image','encoder','decoder','mask'],
  autoencoder: ['input','encoder','latent','reconstruction'],
};
export default function ArchitectureOverview({architecture}:{architecture:Architecture}) {
  const {t}=useTranslation('app');
  return <div className="architecture-visual" aria-label={t('structured.architectureDiagram')}>
    <svg viewBox="0 0 660 176" role="img"><title>{t('structured.architectureDiagram')}</title>
      <defs><linearGradient id="architecture-fill" x1="0" y1="0" x2="1" y2="1"><stop stopColor="currentColor" stopOpacity=".23"/><stop offset="1" stopColor="currentColor" stopOpacity=".03"/></linearGradient></defs>
      {stages[architecture].map((stage,i)=><g key={`${i}-${stage}`} transform={`translate(${38+i*166},20)`}>
        {i<3&&<g className="diagram-arrow"><path d="M82 53 H119 M113 48 L119 53 L113 58"/></g>}
        {i===0?<g className="diagram-object"><rect x="4" y="20" width="64" height="64" rx="9"/>{['image','input'].includes(stage)?<><circle cx="49" cy="38" r="5"/><path d="M6 70 L26 48 L42 66 L52 55 L67 73"/></>:<>{[0,1,2].map(j=><path key={j} d={`M15 ${38+j*14} l8 -6 l9 10 l9 -8 l14 5`}/>)}</>}</g>
        :i===3?<g className="diagram-object"><rect x="21" y="10" width="34" height="88" rx="12"/>{[0,1,2].map(j=><circle key={j} cx="38" cy={30+j*24} r="6"/>)}</g>
        :<g className="diagram-object">{[3,2,1,0].map(j=><rect key={j} x={5+j*8} y={24-j*8} width={i===1?66:48} height={i===1?72:48} rx="8"/>)}</g>}
        <text x="38" y="142" textAnchor="middle">{t(`structured.stages.${stage}`)}</text>
      </g>)}
    </svg>
  </div>;
}
