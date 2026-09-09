import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import * as echarts from "echarts/core";
import { BarChart, HeatmapChart, LineChart } from "echarts/charts";
import { GridComponent, LegendComponent, TooltipComponent, VisualMapComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsOption } from "echarts";
import type { DatasetAnalytics, TaskId } from "./types";

echarts.use([BarChart,HeatmapChart,LineChart,GridComponent,LegendComponent,TooltipComponent,VisualMapComponent,CanvasRenderer]);

type ChartKind="classes"|"target"|"intensity"|"channels"|"features"|"summary"|"correlation"|"series"|"lengths"|"tokens"|"coverage"|"pixels";

export default function DataCharts({analytics,task}:{analytics?:DatasetAnalytics;task:TaskId}){
  const { t } = useTranslation(["charts", "common"]);
  const element=useRef<HTMLDivElement>(null);const choices=useMemo(()=>chartChoices(analytics),[analytics]);const [kind,setKind]=useState<ChartKind>(choices[0]?.id||"intensity");const [feature,setFeature]=useState(0);
  const names=useMemo(()=>({
    samples:t("charts:axis.samples"),
    frequency:t("charts:axis.frequency"),
    pixels:t("charts:axis.pixels"),
    mean:t("charts:series.mean"),
    stdDev:t("charts:series.stdDev"),
    step:t("charts:axis.step"),
    value:t("charts:axis.value"),
    coverage:t("charts:axis.coverage"),
    intensity:t("charts:axis.intensity"),
    texts:t("charts:axis.texts"),
    occurrences:t("charts:axis.occurrences"),
    masks:t("charts:axis.masks"),
  }),[t]);
  useEffect(()=>{if(!choices.some(choice=>choice.id===kind))setKind(choices[0]?.id||"intensity")},[choices,kind]);
  useEffect(()=>{if(!element.current||!analytics)return;const chart=echarts.init(element.current);const render=()=>chart.setOption(optionFor(kind,analytics,feature,element.current!,names),true);render();const observer=new ResizeObserver(()=>chart.resize());observer.observe(element.current);const mutation=new MutationObserver(render);mutation.observe(document.documentElement,{attributes:true,attributeFilter:["data-theme","data-accent"]});return()=>{observer.disconnect();mutation.disconnect();chart.dispose()}},[analytics,kind,feature,task,names]);
  if(!analytics)return <div className="data-chart-empty">{t("charts:loadingProfile")}</div>;
  const hasFeatures=kind==="features"&&(analytics.featureHistograms?.length||0)>1;
  const currentChoice=choices.find(choice=>choice.id===kind);
  return <div className="data-chart-view"><div className="data-chart-controls"><span className="data-chart-label">{t("charts:labels.chart")}</span><div className="data-chart-choice-list" role="group" aria-label={t("charts:labels.chart")}>{choices.map(choice=><button key={choice.id} type="button" className={choice.id===kind?"active":""} aria-pressed={choice.id===kind} onClick={()=>setKind(choice.id)}>{t(choice.label)}</button>)}</div>{hasFeatures&&<label><span>{t("charts:labels.variable")}</span><select value={feature} onChange={event=>setFeature(Number(event.target.value))}>{analytics.featureHistograms!.map((item,index)=><option key={item.name} value={index}>{item.name}</option>)}</select></label>}<small>{t(chartHelp(kind))}</small></div><div ref={element} className="data-echart" role="img" aria-label={t("charts:chartOf", { label: currentChoice ? t(currentChoice.label) : t("charts:labels.data") })}/></div>;
}

function chartChoices(a?:DatasetAnalytics):Array<{id:ChartKind;label:string}>{if(!a)return[];const result:Array<{id:ChartKind;label:string}>=[];if(a.classDistribution)result.push({id:"classes",label:"charts:choices.classes"});if(a.targetHistogram)result.push({id:"target",label:"charts:choices.target"});if(a.intensityHistogram)result.push({id:"intensity",label:"charts:choices.intensity"});if(a.channelStats)result.push({id:"channels",label:"charts:choices.channels"});if(a.featureHistograms)result.push({id:"features",label:"charts:choices.features"});if(a.featureSummary)result.push({id:"summary",label:"charts:choices.summary"});if(a.correlation)result.push({id:"correlation",label:"charts:choices.correlation"});if(a.meanSeries)result.push({id:"series",label:"charts:choices.series"});if(a.lengthHistogram)result.push({id:"lengths",label:"charts:choices.lengths"});if(a.tokenFrequency)result.push({id:"tokens",label:"charts:choices.tokens"});if(a.maskCoverage)result.push({id:"coverage",label:"charts:choices.coverage"});if(a.pixelDistribution)result.push({id:"pixels",label:"charts:choices.pixels"});return result}
function chartHelp(kind:ChartKind){const help:Record<ChartKind,string>={classes:"charts:help.classes",target:"charts:help.target",intensity:"charts:help.intensity",channels:"charts:help.channels",features:"charts:help.features",summary:"charts:help.summary",correlation:"charts:help.correlation",series:"charts:help.series",lengths:"charts:help.lengths",tokens:"charts:help.tokens",coverage:"charts:help.coverage",pixels:"charts:help.pixels"};return help[kind]}

function theme(element:HTMLElement){const css=getComputedStyle(document.documentElement);return {accent:css.getPropertyValue("--accent").trim(),text:css.getPropertyValue("--text").trim(),muted:css.getPropertyValue("--muted").trim(),border:css.getPropertyValue("--border").trim(),surface:css.getPropertyValue("--surface").trim(),amber:css.getPropertyValue("--amber").trim(),green:css.getPropertyValue("--green").trim(),font:getComputedStyle(element).fontFamily}}
function optionFor(kind:ChartKind,a:DatasetAnalytics,feature:number,element:HTMLElement,names:ReturnType<typeof chartNames>):EChartsOption{const c=theme(element);const base:EChartsOption={animationDuration:240,backgroundColor:"transparent",textStyle:{fontFamily:c.font,color:c.text,fontSize:12},tooltip:{trigger:"axis",backgroundColor:c.surface,borderColor:c.border,textStyle:{color:c.text,fontSize:12}},grid:{left:54,right:18,top:22,bottom:42,containLabel:false},xAxis:{type:"category",axisLabel:{color:c.muted,fontSize:11,hideOverlap:true},axisLine:{lineStyle:{color:c.border}},axisTick:{show:false}},yAxis:{type:"value",axisLabel:{color:c.muted,fontSize:11},axisLine:{show:false},splitLine:{lineStyle:{color:c.border,type:"dashed"}}}};
  let labels:string[]=[];let values:number[]=[];let name=names.samples;
  if(kind==="classes"&&a.classDistribution){labels=a.classDistribution.labels;values=a.classDistribution.values;name=names.samples}else if(kind==="target"&&a.targetHistogram){labels=a.targetHistogram.labels;values=a.targetHistogram.values;name=names.frequency}else if(kind==="intensity"&&a.intensityHistogram){labels=a.intensityHistogram.labels;values=a.intensityHistogram.values;name=names.pixels}else if(kind==="features"&&a.featureHistograms?.length){const selected=a.featureHistograms[Math.min(feature,a.featureHistograms.length-1)];labels=selected.labels;values=selected.values;name=names.frequency}else if(kind==="lengths"&&a.lengthHistogram){labels=a.lengthHistogram.labels;values=a.lengthHistogram.values;name=names.texts}else if(kind==="tokens"&&a.tokenFrequency){labels=a.tokenFrequency.labels;values=a.tokenFrequency.values;name=names.occurrences}else if(kind==="coverage"&&a.maskCoverage){labels=a.maskCoverage.labels;values=a.maskCoverage.values;name=names.masks}else if(kind==="pixels"&&a.pixelDistribution){labels=a.pixelDistribution.labels;values=a.pixelDistribution.values;name=names.pixels}
  if(values.length)return {...base,xAxis:{...(base.xAxis as object),data:labels,name:kind==="coverage"?names.coverage:kind==="intensity"?names.intensity:""},yAxis:{...(base.yAxis as object),name},series:[{name,type:"bar",data:values,itemStyle:{color:c.accent,borderRadius:[3,3,0,0]},barMaxWidth:38}]};
  if(kind==="channels"&&a.channelStats)return {...base,legend:{data:[names.mean,names.stdDev],textStyle:{color:c.muted,fontSize:11},top:0},grid:{...(base.grid as object),top:34},xAxis:{...(base.xAxis as object),data:a.channelStats.labels},series:[{name:names.mean,type:"bar",data:a.channelStats.mean,itemStyle:{color:c.accent},barMaxWidth:34},{name:names.stdDev,type:"bar",data:a.channelStats.std,itemStyle:{color:c.amber},barMaxWidth:34}]};
  if(kind==="summary"&&a.featureSummary)return {...base,legend:{data:[names.mean,names.stdDev],textStyle:{color:c.muted,fontSize:11},top:0},grid:{...(base.grid as object),top:34},xAxis:{...(base.xAxis as object),data:a.featureSummary.labels},series:[{name:names.mean,type:"bar",data:a.featureSummary.mean,itemStyle:{color:c.accent},barMaxWidth:28},{name:names.stdDev,type:"bar",data:a.featureSummary.std,itemStyle:{color:c.amber},barMaxWidth:28}]};
  if(kind==="series"&&a.meanSeries)return {...base,legend:{data:a.meanSeries.series.map(s=>s.name),textStyle:{color:c.muted,fontSize:11},top:0},grid:{...(base.grid as object),top:34},xAxis:{...(base.xAxis as object),data:a.meanSeries.labels,name:names.step},yAxis:{...(base.yAxis as object),name:names.value},series:a.meanSeries.series.map((series,index)=>({name:series.name,type:"line",smooth:true,showSymbol:false,data:series.values,lineStyle:{width:2,color:[c.accent,c.amber,c.green][index%3]},itemStyle:{color:[c.accent,c.amber,c.green][index%3]}}))};
  if(kind==="correlation"&&a.correlation){const size=a.correlation.labels.length;return {...base,grid:{left:60,right:62,top:12,bottom:38},xAxis:{type:"category",data:a.correlation.labels,axisLabel:{color:c.muted,fontSize:11}},yAxis:{type:"category",data:a.correlation.labels,axisLabel:{color:c.muted,fontSize:11}},visualMap:{min:-1,max:1,calculable:false,orient:"vertical",right:2,top:"middle",textStyle:{color:c.muted,fontSize:11},inRange:{color:[c.amber,c.surface,c.accent]}},tooltip:{...(base.tooltip as object),position:"top"},series:[{type:"heatmap",data:a.correlation.values,itemStyle:{borderColor:c.surface,borderWidth:2},emphasis:{itemStyle:{shadowBlur:8,shadowColor:c.accent}}}]};}
  return base;
}

function chartNames(t: ReturnType<typeof useTranslation>["t"]) {
  return {
    samples: t("charts:axis.samples"),
    frequency: t("charts:axis.frequency"),
    pixels: t("charts:axis.pixels"),
    mean: t("charts:series.mean"),
    stdDev: t("charts:series.stdDev"),
    step: t("charts:axis.step"),
    value: t("charts:axis.value"),
    coverage: t("charts:axis.coverage"),
    intensity: t("charts:axis.intensity"),
    texts: t("charts:axis.texts"),
    occurrences: t("charts:axis.occurrences"),
    masks: t("charts:axis.masks"),
  };
}
