import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import * as echarts from "echarts/core";
import { LineChart } from "echarts/charts";
import { GridComponent, LegendComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsOption } from "echarts";
import type { MetricPoint } from "./types";

echarts.use([LineChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer]);

const accentColors = { blue: "#22d3ee", orange: "#fb923c", green: "#4ade80", violet: "#a78bfa" } as const;

export function TrainingChart({ history, metric, accent, compact = false }: { history: MetricPoint[]; metric: string; accent: keyof typeof accentColors; compact?: boolean }) {
  const element = useRef<HTMLDivElement>(null);
  const isLoss = metric === "loss";
  const { t } = useTranslation(["common", "charts"]);
  useEffect(() => {
    if (!element.current) return;
    const trainLabel = t("common:train");
    const validationLabel = t("common:validation");
    const chart = echarts.init(element.current);
    const option: EChartsOption = {
      animationDuration: 260,
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "#171a1f",
        borderColor: "#303842",
        textStyle: { color: "#f1f5f9" },
        valueFormatter: value => Number(value).toFixed(4)
      },
      legend: { data: isLoss ? [trainLabel, validationLabel] : [metric], textStyle: { color: "#a8b3c2" }, top: 0 },
      grid: { left: 44, right: 16, top: 38, bottom: 30 },
      xAxis: { type: "category", name: t("charts:axis.epoch"), data: history.map(point => point.epoch), axisLabel: { color: "#768397" }, axisLine: { lineStyle: { color: "#303842" } } },
      yAxis: { type: "value", scale: true, axisLabel: { color: "#768397" }, splitLine: { lineStyle: { color: "#262c34", type: "dashed" } } },
      series: isLoss ? [
        { name: trainLabel, type: "line", smooth: true, showSymbol: history.length < 30, data: history.map(point => point.trainLoss), lineStyle: { color: accentColors[accent], width: 2 }, itemStyle: { color: accentColors[accent] } },
        { name: validationLabel, type: "line", smooth: true, showSymbol: true, data: history.map(point => point.valLoss), lineStyle: { color: "#fbbf24", width: 2, type: "dashed" }, itemStyle: { color: "#fbbf24" } }
      ] : [{ name: metric, type: "line", smooth: true, areaStyle: { color: "rgba(167,139,250,.12)" }, data: history.map(point => point.metric), lineStyle: { color: "#a78bfa", width: 2 }, itemStyle: { color: "#a78bfa" } }]
    };
    chart.setOption(option);
    const resize = () => chart.resize();
    const observer = new ResizeObserver(() => chart.resize());
    if (element.current) observer.observe(element.current);
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      observer.disconnect();
      chart.dispose();
    };
  }, [history, metric, isLoss, accent, t]);
  return <div ref={element} className="echarts-for-react" style={{ flex: 1, minHeight: 180, height: "100%", width: "100%" }} />;
}
