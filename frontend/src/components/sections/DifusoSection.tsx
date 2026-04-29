import ReactECharts from "echarts-for-react";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { AlertTriangle, BookOpen, List, LoaderCircle } from "lucide-react";
import {
  buildClinicalNarrative,
  buildRuleNarrative,
  formatPercentage,
  getFieldLabel,
  getRiskUi,
  obtenerDefinicionesDifusas,
  obtenerReglasDifusas,
  type ExplicacionResponse,
  type FuzzyDefinicionesResponse,
  type FuzzyReglasResponse,
} from "../../lib/riesgoMaterno";
import { generateMembershipSeries } from "../../lib/membership";
import { ChartPanel } from "../ui/ChartPanel";
import { GlassPanel } from "../ui/GlassPanel";
import { SectionHeader } from "../ui/SectionHeader";

interface DifusoSectionProps {
  explanationResult: ExplicacionResponse | null;
}

const CATEGORY_COLORS = ["#22d3ee", "#60a5fa", "#c084fc", "#f472b6", "#fb923c"];

const toneClassMap = {
  low: "text-emerald-700 border-emerald-200 bg-emerald-50",
  mid: "text-amber-700 border-amber-200 bg-amber-50",
  high: "text-rose-700 border-rose-200 bg-rose-50",
};

const activationOrder = [
  { key: "alto", label: "Riesgo alto" },
  { key: "medio", label: "Riesgo medio" },
  { key: "bajo", label: "Riesgo bajo" },
] as const;

export function DifusoSection({ explanationResult }: DifusoSectionProps) {
  const [definitions, setDefinitions] = useState<FuzzyDefinicionesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const variableNames = definitions ? Object.keys(definitions.variables) : [];
  const [selectedVariable, setSelectedVariable] = useState<string>("");
  const [reglasData, setReglasData] = useState<FuzzyReglasResponse | null>(null);
  const [showReglas, setShowReglas] = useState(false);

  useEffect(() => {
    obtenerDefinicionesDifusas()
      .then((data) => {
        setDefinitions(data);
        const first = Object.keys(data.variables)[0];
        if (first) setSelectedVariable(first);
      })
      .catch((e: Error) => setFetchError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (showReglas && !reglasData) {
      obtenerReglasDifusas()
        .then(setReglasData)
        .catch(() => {});
    }
  }, [showReglas, reglasData]);

  return (
    <section className="section-anchor pt-10" id="difuso">
      <SectionHeader
        eyebrow="Inferencia Mamdani"
        title="Funciones de pertenencia y reglas"
        description="Curvas trapezoidales (base vs optimizado), fuzzificacion del caso actual y reglas activadas."
      />

      {loading ? (
        <GlassPanel className="flex min-h-48 items-center justify-center gap-3 p-8 text-slate-600">
          <LoaderCircle className="h-5 w-5 animate-spin text-cyan-600" />
          <span className="text-sm">Cargando definiciones del sistema difuso...</span>
        </GlassPanel>
      ) : fetchError ? (
        <GlassPanel className="p-6 text-sm text-rose-700">
          Error al cargar definiciones: {fetchError}
        </GlassPanel>
      ) : definitions ? (
        <>
          <MembershipCurvesPanel
            definitions={definitions}
            variableNames={variableNames}
            selectedVariable={selectedVariable}
            onSelectVariable={setSelectedVariable}
            explanationResult={explanationResult}
          />

          {/* Tabla de todas las reglas aprendidas por RIPPER */}
          <div className="mt-6">
            <button
              onClick={() => setShowReglas((v) => !v)}
              className="inline-flex items-center gap-2 rounded-full border border-sky-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-cyan-300 hover:bg-cyan-50"
            >
              <List className="h-4 w-4 text-cyan-600" />
              {showReglas ? "Ocultar reglas" : "Ver reglas"}
              {reglasData && (
                <span className="rounded-full bg-cyan-100 px-2 py-0.5 text-xs font-semibold text-cyan-700">
                  {reglasData.total}
                </span>
              )}
            </button>

            {showReglas && reglasData && (
              <GlassPanel className="mt-3 p-5 sm:p-6">
                <div className="text-xs uppercase tracking-[0.22em] text-cyan-700/80 mb-4">
                  Reglas aprendidas por RIPPER — {reglasData.total} reglas IF-THEN
                </div>
                <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
                  {reglasData.reglas.map((regla) => {
                    const riskUi = getRiskUi(regla.consecuente);
                    return (
                      <div
                        key={regla.numero}
                        className="flex flex-wrap items-center gap-2 rounded-2xl border border-slate-200/70 bg-white/90 px-4 py-3 text-sm"
                      >
                        <span className="shrink-0 rounded-full border border-cyan-300/30 bg-cyan-50 px-2.5 py-0.5 text-xs font-semibold text-cyan-700">
                          #{regla.numero}
                        </span>
                        <span className="text-slate-500 text-xs">IF</span>
                        {regla.antecedentes.map((ant, i) => (
                          <span key={i} className="flex items-center gap-1 text-xs">
                            <span className="font-medium text-slate-700">{getFieldLabel(ant.variable)}</span>
                            <span className="text-slate-400">=</span>
                            <span className="rounded-full bg-sky-50 px-2 py-0.5 font-mono text-sky-700">{ant.categoria}</span>
                            {i < regla.antecedentes.length - 1 && (
                              <span className="text-slate-400 ml-1">AND</span>
                            )}
                          </span>
                        ))}
                        <span className="text-slate-500 text-xs ml-1">THEN</span>
                        <span
                          className="rounded-full border px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wide"
                          style={{
                            borderColor: `${riskUi.accent}55`,
                            color: riskUi.accent,
                            backgroundColor: `${riskUi.accent}18`,
                          }}
                        >
                          {riskUi.label}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </GlassPanel>
            )}
          </div>

          {explanationResult ? (
            <>
              {explanationResult.sin_activacion && (
                <div className="flex items-start gap-3 rounded-2xl border border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-800">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
                  <div>
                    <p className="font-semibold">Ninguna regla se activo para este perfil</p>
                    <p className="mt-1 text-amber-700">
                      Los valores ingresados no coincidieron con ninguna regla aprendida.
                      El puntaje de 50 es un valor neutro por defecto, no una clasificacion clinica real.
                    </p>
                  </div>
                </div>
              )}
              <FuzzificationPanel result={explanationResult} />
              <RulesPanel result={explanationResult} />
              {!explanationResult.sin_activacion && (
                <AggregacionPanel result={explanationResult} />
              )}
            </>
          ) : (
            <GlassPanel className="mt-6 p-6 text-sm text-slate-500">
              Ejecute una prediccion en la pantalla de Prediccion para ver la fuzzificacion,
              reglas activadas y agregacion del caso actual.
            </GlassPanel>
          )}
        </>
      ) : null}
    </section>
  );
}

// ── Curvas de pertenencia ─────────────────────────────────────────────────────

function MembershipCurvesPanel({
  definitions,
  variableNames,
  selectedVariable,
  onSelectVariable,
  explanationResult,
}: {
  definitions: FuzzyDefinicionesResponse;
  variableNames: string[];
  selectedVariable: string;
  onSelectVariable: (v: string) => void;
  explanationResult: ExplicacionResponse | null;
}) {
  const varDef = definitions.variables[selectedVariable];
  if (!varDef) return null;

  const [domainMin, domainMax] = varDef.limites;
  const domain: [number, number] = [domainMin, domainMax];
  const categoryNames = Object.keys(varDef.categorias);
  const currentPoint = explanationResult?.entrada_validada?.[selectedVariable] ?? null;

  const series = categoryNames.flatMap((cat, idx) => {
    const color = CATEGORY_COLORS[idx % CATEGORY_COLORS.length];
    const { puntos_base, puntos_optimizados } = varDef.categorias[cat];

    const basePoints = puntos_base as [number, number, number, number];
    const optPoints = puntos_optimizados as [number, number, number, number];

    const baseSeries = generateMembershipSeries(domain, basePoints);
    const optSeries = generateMembershipSeries(domain, optPoints);

    const mu = explanationResult?.pertenencias?.[selectedVariable]?.[cat] ?? null;

    return [
      {
        name: `${cat} (base)`,
        type: "line",
        smooth: false,
        symbol: "none",
        lineStyle: { width: 1.5, type: "dashed", color, opacity: 0.5 },
        data: baseSeries.map((p) => [p.x, p.membership]),
      },
      {
        name: `${cat} (opt)  μ=${mu !== null ? mu.toFixed(2) : "—"}`,
        type: "line",
        smooth: false,
        symbol: "none",
        lineStyle: { width: 2.5, color },
        areaStyle: { opacity: 0.06, color },
        data: optSeries.map((p) => [p.x, p.membership]),
      },
    ];
  });

  // Vertical marker for current patient value
  const markerSeries =
    currentPoint !== null
      ? [
          {
            name: `Valor actual (${currentPoint})`,
            type: "line",
            symbol: "none",
            lineStyle: { width: 2, color: "#f59e0b", type: "dotted" },
            markLine: {
              symbol: "none",
              lineStyle: { color: "#f59e0b", type: "solid", width: 2 },
              label: { formatter: `x=${currentPoint}`, color: "#f59e0b", fontSize: 11 },
              data: [{ xAxis: currentPoint }],
            },
            data: [],
          },
        ]
      : [];

  const chartOption = {
    backgroundColor: "transparent",
    animation: false,
    animationDuration: 0,
    animationDurationUpdate: 0,
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(255,255,255,0.96)",
      borderColor: "rgba(125,211,252,0.65)",
      textStyle: { color: "#0f172a" },
    },
    legend: { show: false },
    grid: { left: 12, right: 12, top: 16, bottom: 12, containLabel: true },
    xAxis: {
      type: "value",
      min: domainMin,
      max: domainMax,
      axisLine: { lineStyle: { color: "rgba(148,163,184,0.35)" } },
      splitLine: { lineStyle: { color: "rgba(148,163,184,0.16)" } },
      axisLabel: { color: "rgba(30,41,59,0.82)" },
    },
    yAxis: {
      type: "value",
      min: 0,
      max: 1,
      name: "μ",
      nameTextStyle: { color: "rgba(71,85,105,0.8)" },
      axisLine: { lineStyle: { color: "rgba(148,163,184,0.35)" } },
      splitLine: { lineStyle: { color: "rgba(148,163,184,0.16)" } },
      axisLabel: { color: "rgba(30,41,59,0.82)" },
    },
    series: [...series, ...markerSeries],
  };

  return (
    <div className="w-full grid gap-4 xl:grid-cols-[260px_1fr] xl:items-stretch">
      <GlassPanel className="p-5 sm:p-6 flex flex-col">
        <div className="text-xs uppercase tracking-[0.22em] text-cyan-700/80">Variables</div>
        <div className="mt-4 flex flex-col gap-2">
          {variableNames.map((v) => (
            <button
              key={v}
              className={`rounded-2xl border px-4 py-2.5 text-left text-sm font-medium transition ${
                selectedVariable === v
                  ? "border-cyan-400/50 bg-cyan-50 text-cyan-900"
                  : "border-sky-100 bg-white text-slate-700 hover:border-cyan-300/40 hover:bg-cyan-50"
              }`}
              onClick={() => onSelectVariable(v)}
              type="button"
            >
              {getFieldLabel(v)}
            </button>
          ))}
        </div>

        {varDef && (
          <div className="mt-5 space-y-2">
            <div className="text-xs uppercase tracking-[0.18em] text-slate-500">Categorias (opt)</div>
            {categoryNames.map((cat, idx) => {
              const color = CATEGORY_COLORS[idx % CATEGORY_COLORS.length];
              const pts = varDef.categorias[cat].puntos_optimizados;
              const mu = explanationResult?.pertenencias?.[selectedVariable]?.[cat];
              return (
                <div
                  key={cat}
                  className="rounded-2xl border border-sky-100 bg-white/82 px-3 py-2 text-xs"
                >
                  <div className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
                    <span className="font-semibold text-slate-800">{cat}</span>
                    {mu !== undefined && mu > 0 && (
                      <span className="ml-auto rounded-full bg-cyan-50 px-2 py-0.5 text-cyan-700 font-mono">
                        μ={mu.toFixed(2)}
                      </span>
                    )}
                  </div>
                  <div className="mt-1 font-mono text-[11px] text-slate-400">
                    [{pts.map((n) => n.toFixed(1)).join(", ")}]
                  </div>
                </div>
              );
            })}
            {currentPoint !== null && (
              <div className="mt-3 rounded-2xl border border-cyan-200 bg-cyan-50 px-3 py-2 text-xs font-mono text-cyan-800">
                Valor actual: {currentPoint}
              </div>
            )}
          </div>
        )}
      </GlassPanel>

      <ChartPanel
        title={`${getFieldLabel(selectedVariable)} — curvas de pertenencia`}
        subtitle="Linea discontinua = base / Linea solida = optimizado"
      >
        <div className="h-[420px]">
          <ReactECharts
            key={selectedVariable}
            notMerge={true}
            lazyUpdate={false}
            option={chartOption}
            style={{ height: "100%", width: "100%" }}
          />
        </div>
      </ChartPanel>
    </div>
  );
}

// ── Fuzzificacion ─────────────────────────────────────────────────────────────

function FuzzificationPanel({ result }: { result: ExplicacionResponse }) {
  const variables = Object.entries(result.pertenencias);
  // max categories across all variables to know how many columns to render
  const maxCats = Math.max(...variables.map(([, cats]) => Object.keys(cats).length));

  return (
    <GlassPanel className="mt-6 p-5 sm:p-6">
      <div className="text-xs uppercase tracking-[0.22em] text-cyan-700/80 mb-3">
        Fuzzificacion — grados de pertenencia del caso actual
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-sky-100 text-xs uppercase tracking-wide text-slate-500">
              <th className="py-1 pr-4 text-left font-semibold">Variable</th>
              <th className="py-1 pr-4 text-right font-semibold">Valor</th>
              {Array.from({ length: maxCats }, (_, i) => (
                <th key={i} className="py-1 px-2 text-center font-semibold text-slate-400">
                  Cat. {i + 1}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {variables.map(([variable, cats]) => {
              const valor = result.entrada_validada[variable];
              const catEntries = Object.entries(cats);
              return (
                <tr key={variable} className="border-b border-sky-50 hover:bg-sky-50/30">
                  <td className="py-1.5 pr-4 font-medium text-slate-800 whitespace-nowrap text-xs">
                    {getFieldLabel(variable)}
                  </td>
                  <td className="py-1.5 pr-4 text-right font-mono text-slate-500 whitespace-nowrap text-xs">
                    {valor !== undefined ? valor : "—"}
                  </td>
                  {catEntries.map(([cat, mu]) => (
                    <td key={cat} className="py-1.5 px-2 text-center whitespace-nowrap">
                      <span className="text-[10px] text-slate-400 capitalize">{cat} </span>
                      <span
                        className={`font-mono text-xs font-bold ${
                          mu > 0.5
                            ? "text-cyan-700"
                            : mu > 0.1
                              ? "text-sky-600"
                              : "text-slate-500"
                        }`}
                      >
                        {mu.toFixed(3)}
                      </span>
                    </td>
                  ))}
                  {Array.from({ length: maxCats - catEntries.length }, (_, i) => (
                    <td key={`empty-${i}`} className="py-1.5 px-2" />
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </GlassPanel>
  );
}

// ── Agregacion ────────────────────────────────────────────────────────────────

function AggregacionPanel({ result }: { result: ExplicacionResponse }) {
  const narrative = buildClinicalNarrative(result);
  const esFallback = result.sin_activacion;

  return (
    <GlassPanel className="mt-6 p-6 sm:p-7">
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <div className="text-xs uppercase tracking-[0.22em] text-cyan-700/80 mb-4">
            Agregacion — fuerza por nivel de riesgo
          </div>
          <div className="space-y-4">
            {activationOrder.map(({ key, label }) => {
              const risk = getRiskUi(key);
              const activation = result.activaciones[key] ?? 0;
              return (
                <div key={key}>
                  <div className="mb-1.5 flex items-center justify-between text-sm">
                    <span
                      className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold uppercase tracking-[0.16em] ${toneClassMap[risk.tone]}`}
                    >
                      {label}
                    </span>
                    <span className="font-mono font-semibold text-slate-700">
                      {formatPercentage(activation)}
                    </span>
                  </div>
                  <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ backgroundColor: risk.accent }}
                      initial={{ width: 0 }}
                      animate={{ width: `${activation > 0 ? Math.max(4, activation * 100) : 0}%` }}
                      transition={{ duration: 0.6, ease: "easeOut" }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="flex flex-col gap-4 lg:pl-6 lg:border-l lg:border-sky-100">
          <div className="text-xs uppercase tracking-[0.22em] text-cyan-700/80">
            Defuzzificacion (centroide)
          </div>
          <div className="flex items-end gap-3">
            <div
              className="text-5xl font-semibold"
              style={{ color: esFallback ? "#94a3b8" : getRiskUi(result.riesgo).accent }}
            >
              {result.puntaje.toFixed(1)}
            </div>
            <div className="mb-1 text-sm text-slate-500">/ 100</div>
          </div>
          {esFallback ? (
            <div className="inline-flex w-fit items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-xs font-semibold text-amber-700">
              <AlertTriangle className="h-3 w-3" />
              Valor neutro de respaldo
            </div>
          ) : (
            <div
              className="inline-flex w-fit rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em]"
              style={{
                borderColor: `${getRiskUi(result.riesgo).accent}55`,
                color: getRiskUi(result.riesgo).accent,
                backgroundColor: `${getRiskUi(result.riesgo).accent}18`,
              }}
            >
              {getRiskUi(result.riesgo).label}
            </div>
          )}

          {/* Barra visual del centroide sobre el universo de salida 0-100 */}
          <div className="mt-2">
            <div className="mb-1 flex justify-between text-xs text-slate-400">
              <span>0</span>
              <span>40</span>
              <span>65</span>
              <span>100</span>
            </div>
            {/* Zonas coloreadas: bajo=verde, medio=ambar, alto=rojo */}
            <div className="relative h-4 w-full overflow-hidden rounded-full flex">
              <div className="h-full bg-emerald-200" style={{ width: "40%" }} />
              <div className="h-full bg-amber-200" style={{ width: "25%" }} />
              <div className="h-full bg-rose-200" style={{ width: "35%" }} />
              {/* Marcador del centroide */}
              <motion.div
                className="absolute top-0 h-full w-1 rounded-full bg-slate-800 shadow"
                style={{ left: `${result.puntaje}%` }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.4 }}
              />
            </div>
            <div className="mt-1 flex justify-between text-xs text-slate-400">
              <span>Bajo</span>
              <span>Medio</span>
              <span>Alto</span>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-5 border-t border-sky-100 pt-5">
        <div className="flex items-start gap-2 text-sm text-slate-600">
          <BookOpen className="mt-0.5 h-4 w-4 shrink-0 text-cyan-600" />
          <p className="leading-6">{narrative.intro} {narrative.details}</p>
        </div>
      </div>
    </GlassPanel>
  );
}

// ── Reglas activadas ──────────────────────────────────────────────────────────

function RulesPanel({ result }: { result: ExplicacionResponse }) {
  const sortedRules = [...result.reglas_activadas].sort((a, b) => b.fuerza - a.fuerza);

  return (
    <GlassPanel className="mt-6 p-5 sm:p-6">
      <div className="flex items-start gap-4">
        <div className="shrink-0 rounded-2xl border border-cyan-300/30 bg-cyan-50 p-3 text-cyan-700">
          <BookOpen className="h-5 w-5" />
        </div>
        <div>
          <div className="text-base font-semibold text-slate-900">
            Reglas activadas ({sortedRules.length})
          </div>
          <p className="mt-1 text-sm text-slate-500">
            Ordenadas por fuerza de activacion, de mayor a menor.
          </p>
        </div>
      </div>

      <div className="mt-5 space-y-3">
        {sortedRules.length === 0 && (
          <div className="flex items-start gap-3 rounded-2xl border border-amber-100 bg-amber-50/60 px-4 py-3 text-sm text-amber-700">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
            <span>
              Ninguna regla se activo para este perfil. El puntaje 50 es un valor neutro de
              respaldo generado cuando no hay evidencia clinica disponible.
            </span>
          </div>
        )}
        {sortedRules.map((rule, index) => {
          const ruleRisk = getRiskUi(rule.consecuente);
          const narrative = buildRuleNarrative(rule);

          return (
            <motion.div
              key={rule.numero}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, delay: index * 0.03, ease: [0.22, 1, 0.36, 1] }}
            >
              <div className="rounded-[1.75rem] border border-slate-200/70 bg-white/92 p-5 sm:p-6">
                <div className="flex flex-wrap items-center gap-3">
                  <div className="flex flex-wrap items-center gap-3">
                    <span className="rounded-full border border-cyan-300/30 bg-cyan-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-cyan-700">
                      Regla {rule.numero}
                    </span>
                    <span
                      className={`rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] ${toneClassMap[ruleRisk.tone]}`}
                    >
                      Indica {ruleRisk.label.toLowerCase()}
                    </span>
                    <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-slate-600">
                      Peso {formatPercentage(rule.fuerza)}
                    </span>
                  </div>
                </div>
                <p className="mt-4 text-sm leading-7 text-slate-700">
                  <span className="font-medium text-slate-500">Se activo porque: </span>
                  {narrative}
                </p>
                <div className="mt-4 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className="h-1.5 rounded-full"
                    style={{
                      backgroundColor: ruleRisk.accent,
                      width: `${Math.max(4, Math.round(rule.fuerza * 100))}%`,
                    }}
                  />
                </div>
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {rule.antecedentes.map((ant) => (
                    <span
                      key={`${ant.variable}-${ant.categoria}`}
                      className="rounded-full border border-sky-100 bg-white px-2.5 py-0.5 text-xs text-slate-600"
                    >
                      {getFieldLabel(ant.variable)}{" "}
                      <span className="font-semibold text-cyan-700">{ant.categoria}</span>{" "}
                      <span className="font-mono text-slate-400">μ={ant.pertenencia.toFixed(2)}</span>
                    </span>
                  ))}
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </GlassPanel>
  );
}
