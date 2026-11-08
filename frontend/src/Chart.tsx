import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import * as echarts from "echarts/core";
import { LineChart } from "echarts/charts";
import { GridComponent, LegendComponent, MarkLineComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsOption } from "echarts";
import type { MetricPoint } from "./types";

echarts.use([LineChart, GridComponent, LegendComponent, MarkLineComponent, TooltipComponent, CanvasRenderer]);

const accentColors = { blue: "#22d3ee", orange: "#fb923c", green: "#4ade80", violet: "#a78bfa" } as const;
const metricColors = { train: "#a78bfa", validation: "#22d3ee" } as const;
const lossColors = { train: "#fb923c", validation: "#fbbf24" } as const;

function findBestValidation(history: MetricPoint[], metric: string, isLoss: boolean) {
  const lowerIsBetter = isLoss || /^(mse|mae)$/i.test(metric.trim());
  return history.reduce<{ index: number; point: MetricPoint; value: number } | null>((best, point, index) => {
    const value = isLoss ? point.valLoss : point.valMetric ?? point.metric;
    if (!Number.isFinite(value)) return best;
    if (!best || (lowerIsBetter ? value < best.value : value > best.value)) return { index, point, value };
    return best;
  }, null);
}

export function TrainingChart({ history, metric }: { history: MetricPoint[]; metric: string; accent: keyof typeof accentColors; compact?: boolean }) {
  const element = useRef<HTMLDivElement>(null);
  const isLoss = metric === "loss";
  const hasBoundedScale = /^(accuracy|iou|miou|dice|map(?:@.*)?)$/i.test(metric.trim());
  const { t } = useTranslation(["common", "charts", "app"]);
  const bestValidation = findBestValidation(history, metric, isLoss);
  const accessibleBestLabel = bestValidation
    ? isLoss
      ? t("app:training.bestValLoss", { loss: bestValidation.value.toFixed(4), epoch: bestValidation.point.epoch })
      : t("app:training.bestMetric", { metric, value: bestValidation.value.toFixed(4), epoch: bestValidation.point.epoch })
    : "";
  useEffect(() => {
    if (!element.current) return;
    const trainLabel = t("common:train");
    const validationLabel = t("common:validation");
    const bestLabel = accessibleBestLabel || null;
    const chart = echarts.init(element.current);
    const validationColor = isLoss ? lossColors.validation : metricColors.validation;
    const trainColor = isLoss ? lossColors.train : metricColors.train;
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
      legend: { data: [trainLabel, validationLabel], textStyle: { color: "#a8b3c2" }, top: 0 },
      grid: { left: 44, right: 16, top: 38, bottom: 30 },
      xAxis: { type: "category", name: t("charts:axis.epoch"), data: history.map(point => point.epoch), axisLabel: { color: "#768397" }, axisLine: { lineStyle: { color: "#303842" } } },
      yAxis: { type: "value", ...(hasBoundedScale ? { min: 0, max: 1, interval: 0.2 } : { scale: true }), axisLabel: { color: "#768397" }, splitLine: { lineStyle: { color: "#262c34", type: "dashed" } } },
      series: isLoss ? [
        { name: trainLabel, type: "line", smooth: true, showSymbol: false, symbol: "none", data: history.map(point => point.trainLoss), lineStyle: { color: trainColor, width: 2 }, itemStyle: { color: trainColor } },
        { name: validationLabel, type: "line", smooth: true, showSymbol: false, symbol: "none", data: history.map(point => point.valLoss), lineStyle: { color: validationColor, width: 2, type: "dashed" }, itemStyle: { color: validationColor }, markLine: bestValidation && bestLabel ? {
          symbol: ["none", "none"],
          silent: true,
          lineStyle: { color: "#e2e8f0", width: 1.5, type: "dotted" },
          label: { show: true, formatter: bestLabel, position: "insideEndTop", color: "#f1f5f9", backgroundColor: "#171a1f", padding: [4, 6], borderRadius: 3 },
          data: [{ xAxis: bestValidation.index, name: bestLabel }]
        } : undefined }
      ] : [
        { name: trainLabel, type: "line", smooth: true, showSymbol: false, symbol: "none", data: history.map(point => point.trainMetric ?? null), lineStyle: { color: trainColor, width: 2 }, itemStyle: { color: trainColor } },
        { name: validationLabel, type: "line", smooth: true, showSymbol: false, symbol: "none", data: history.map(point => point.valMetric ?? point.metric), lineStyle: { color: validationColor, width: 2, type: "dashed" }, itemStyle: { color: validationColor }, markLine: bestValidation && bestLabel ? {
          symbol: ["none", "none"],
          silent: true,
          lineStyle: { color: "#e2e8f0", width: 1.5, type: "dotted" },
          label: { show: true, formatter: bestLabel, position: "insideEndTop", color: "#f1f5f9", backgroundColor: "#171a1f", padding: [4, 6], borderRadius: 3 },
          data: [{ xAxis: bestValidation.index, name: bestLabel }]
        } : undefined }
      ]
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
  }, [history, metric, isLoss, hasBoundedScale, accessibleBestLabel, t]);
  return <div ref={element} className="echarts-for-react" role="img" aria-label={`${t("charts:labels.chart")} ${metric}${accessibleBestLabel ? `. ${accessibleBestLabel}` : ""}`} style={{ flex: 1, minHeight: 180, height: "100%", width: "100%" }} />;
}
