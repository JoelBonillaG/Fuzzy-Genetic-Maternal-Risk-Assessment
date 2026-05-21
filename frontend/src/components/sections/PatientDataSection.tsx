import ReactECharts from "echarts-for-react";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import {
  Activity,
  Droplets,
  HeartPulse,
  LoaderCircle,
  Stethoscope,
  Thermometer,
  UserRound,
  type LucideIcon,
} from "lucide-react";
import { generateMembershipSeries, resolveMembershipPoints, trapezoidMembership } from "../../lib/membership";
import {
  getFieldLabel,
  getFieldUnit,
  obtenerDefinicionesDifusas,
  VARIABLE_ORDER,
  type FuzzyDefinicionesResponse,
  type VariableDefinicion,
} from "../../lib/riesgoMaterno";
import type { PatientValues } from "../../data/mockData";
import { GlassPanel } from "../ui/GlassPanel";
import { SectionHeader } from "../ui/SectionHeader";

const CATEGORY_COLORS = ["#22d3ee", "#60a5fa", "#c084fc", "#f472b6", "#fb923c"];

const iconByVariable: Record<string, LucideIcon> = {
  edad: UserRound,
  presion_sistolica: Stethoscope,
  presion_diastolica: Activity,
  azucar_sangre: Droplets,
  temperatura_corporal: Thermometer,
  frecuencia_cardiaca: HeartPulse,
};

interface PatientDataSectionProps {
  values: PatientValues;
  isAnalyzing: boolean;
  onValueChange: (variable: string, value: number) => void;
  onAnalyze: () => void;
  onClear: () => void;
}

export function PatientDataSection({
  values,
  isAnalyzing,
  onValueChange,
  onAnalyze,
  onClear,
}: PatientDataSectionProps) {
  const [definitions, setDefinitions] = useState<FuzzyDefinicionesResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    obtenerDefinicionesDifusas()
      .then(setDefinitions)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const allFilled = VARIABLE_ORDER.every((v) => values[v] !== undefined);
  const anyTouched = VARIABLE_ORDER.some((v) => values[v] !== undefined);

  return (
    <section className="section-anchor pt-10" id="patient-entry">
      <SectionHeader
        eyebrow="Ingreso clinico"
        title="Variables del paciente"
        description="Ajuste cada indicador clinico con el control deslizante. Las curvas muestran los grados de pertenencia en tiempo real."
      />

      {loading ? (
        <GlassPanel className="flex min-h-48 items-center justify-center gap-3 p-8 text-slate-600">
          <LoaderCircle className="h-5 w-5 animate-spin text-cyan-600" />
          <span className="text-sm">Cargando definiciones del sistema difuso...</span>
        </GlassPanel>
      ) : definitions ? (
        <>
          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            {VARIABLE_ORDER.map((variable, index) => {
              const varDef = definitions.variables[variable];
              if (!varDef) return null;
              const touched = values[variable] !== undefined;
              const currentValue = values[variable] ?? varDef.limites[0];
              const Icon = iconByVariable[variable] ?? UserRound;

              return (
                <motion.div
                  key={variable}
                  animate={{ opacity: 1, y: 0 }}
                  initial={{ opacity: 0, y: 16 }}
                  transition={{ duration: 0.4, delay: index * 0.06, ease: [0.22, 1, 0.36, 1] }}
                >
                  <VariableCard
                    variable={variable}
                    varDef={varDef}
                    currentValue={currentValue}
                    touched={touched}
                    Icon={Icon}
                    onValueChange={onValueChange}
                  />
                </motion.div>
              );
            })}
          </div>

          <div className="mt-6 flex items-center gap-3">
            <button
              className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-cyan-600 to-sky-600 px-6 py-3 text-sm font-semibold text-white transition hover:scale-[1.01] hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-60"
              disabled={isAnalyzing || !allFilled}
              onClick={onAnalyze}
              type="button"
            >
              {isAnalyzing ? (
                <>
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                  Ejecutando inferencia...
                </>
              ) : (
                "Ejecutar inferencia"
              )}
            </button>
            {anyTouched && (
              <button
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-3 text-sm font-medium text-slate-600 transition hover:border-rose-200 hover:bg-rose-50 hover:text-rose-600 disabled:opacity-50"
                disabled={isAnalyzing}
                onClick={onClear}
                type="button"
              >
                Limpiar
              </button>
            )}
            {!allFilled && (
              <span className="text-xs text-slate-400">
                Ajuste los 6 indicadores para continuar.
              </span>
            )}
          </div>
        </>
      ) : (
        <GlassPanel className="p-6 text-sm text-rose-700">
          No se pudieron cargar las definiciones del sistema difuso.
        </GlassPanel>
      )}
    </section>
  );
}

function VariableCard({
  variable,
  varDef,
  currentValue,
  touched,
  Icon,
  onValueChange,
}: {
  variable: string;
  varDef: VariableDefinicion;
  currentValue: number;
  touched: boolean;
  Icon: LucideIcon;
  onValueChange: (variable: string, value: number) => void;
}) {
  const [domainMin, domainMax] = varDef.limites;
  const domain: [number, number] = [domainMin, domainMax];
  const categoryNames = Object.keys(varDef.categorias);

  const activeMemberships = categoryNames
    .map((cat, idx) => ({
      cat,
      mu: trapezoidMembership(currentValue, resolveMembershipPoints(varDef.categorias[cat])),
      color: CATEGORY_COLORS[idx % CATEGORY_COLORS.length],
    }))
    .filter((m) => m.mu > 0.01)
    .sort((a, b) => b.mu - a.mu)
    .slice(0, 3);

  const range = domainMax - domainMin;
  const step = range <= 20 ? 0.1 : 1;

  const curveSeries = categoryNames.map((cat, idx) => {
    const color = CATEGORY_COLORS[idx % CATEGORY_COLORS.length];
    const points = resolveMembershipPoints(varDef.categorias[cat]);
    return {
      name: cat,
      type: "line",
      smooth: false,
      symbol: "none",
      color,
      itemStyle: { color },
      lineStyle: { width: 2, color },
      areaStyle: { opacity: 0.07, color },
      data: generateMembershipSeries(domain, points).map((p) => [p.x, p.membership]),
    };
  });

  const markerSeries = touched
    ? [
        {
          name: "__marker",
          type: "line",
          symbol: "none",
          color: "#f59e0b",
          itemStyle: { color: "#f59e0b" },
          lineStyle: { width: 0 },
          data: [] as number[][],
          markLine: {
            symbol: "none",
            animation: false,
            lineStyle: { color: "#f59e0b", type: "solid" as const, width: 2 },
            label: { show: false },
            data: [{ xAxis: currentValue }],
          },
        },
      ]
    : [];

  const chartOption = {
    backgroundColor: "transparent",
    animation: false,
    grid: { left: 6, right: 6, top: 6, bottom: 6, containLabel: true },
    xAxis: {
      type: "value",
      min: domainMin,
      max: domainMax,
      axisLine: { lineStyle: { color: "rgba(148,163,184,0.3)" } },
      splitLine: { show: false },
      axisLabel: { color: "rgba(30,41,59,0.55)", fontSize: 9 },
    },
    yAxis: {
      type: "value",
      min: 0,
      max: 1,
      axisLine: { show: false },
      splitLine: { lineStyle: { color: "rgba(148,163,184,0.12)" } },
      axisLabel: { show: false },
    },
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(255,255,255,0.96)",
      borderColor: "rgba(125,211,252,0.65)",
      textStyle: { color: "#0f172a", fontSize: 11 },
    },
    series: [...curveSeries, ...markerSeries],
  };

  return (
    <div
      className={`rounded-3xl border p-4 transition-all duration-300 ${
        touched
          ? "border-emerald-300/60 bg-emerald-50/40 shadow-sm shadow-emerald-100"
          : "border-sky-100 bg-white/78"
      }`}
    >
      {/* Header */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className={`rounded-2xl border p-2.5 transition-colors ${
              touched
                ? "border-emerald-300/40 bg-emerald-100 text-emerald-700"
                : "border-cyan-300/30 bg-cyan-50 text-cyan-700"
            }`}
          >
            <Icon className="h-4 w-4" />
          </div>
          <span className="text-sm font-semibold text-slate-900">{getFieldLabel(variable)}</span>
        </div>
        <span className="rounded-full border border-sky-100 bg-sky-50 px-2.5 py-1 text-xs font-medium text-slate-600">
          {getFieldUnit(variable)}
        </span>
      </div>

      {/* Membership curves chart */}
      <div className="h-[108px]">
        <ReactECharts
          notMerge={false}
          lazyUpdate={false}
          option={chartOption}
          style={{ height: "100%", width: "100%" }}
        />
      </div>

      {/* Slider */}
      <div className="mt-2 px-0.5">
        <input
          type="range"
          min={domainMin}
          max={domainMax}
          step={step}
          value={touched ? currentValue : domainMin}
          className="w-full cursor-pointer accent-cyan-500"
          onChange={(e) => onValueChange(variable, parseFloat(e.target.value))}
        />
      </div>

      {/* Value display */}
      <div className="mt-1.5 flex items-start justify-between gap-2 min-h-[20px]">
        {touched ? (
          <>
            <span className="shrink-0 font-mono text-sm font-semibold text-slate-800">
              {step < 1 ? currentValue.toFixed(1) : currentValue.toFixed(0)}
            </span>
            <div className="flex flex-wrap justify-end gap-1">
              {activeMemberships.map(({ cat, mu, color }) => (
                <span
                  key={cat}
                  className="rounded-full border px-2 py-0.5 text-xs font-semibold"
                  style={{ color, backgroundColor: `${color}18`, borderColor: `${color}40` }}
                >
                  {cat.replaceAll("_", " ")} {mu.toFixed(2)}
                </span>
              ))}
            </div>
          </>
        ) : (
          <span className="text-xs text-slate-400">Mueva el control para seleccionar</span>
        )}
      </div>
    </div>
  );
}
