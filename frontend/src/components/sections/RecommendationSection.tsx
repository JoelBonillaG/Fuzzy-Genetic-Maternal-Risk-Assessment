import { AnimatePresence, motion } from "framer-motion";
import ReactECharts from "echarts-for-react";
import { AlertTriangle, BookOpen, CheckCircle2, LoaderCircle, ShieldAlert } from "lucide-react";
import {
  buildClinicalNarrative,
  buildResultSummary,
  buildRuleNarrative,
  formatPercentage,
  formatScore,
  formatValue,
  getFieldLabel,
  getRiskUi,
  type ExplicacionResponse,
} from "../../lib/riesgoMaterno";
import { GlassPanel } from "../ui/GlassPanel";
import { SectionHeader } from "../ui/SectionHeader";

interface RecommendationSectionProps {
  result: ExplicacionResponse | null;
  isLoading: boolean;
  error: string | null;
}

const iconMap = {
  low: CheckCircle2,
  mid: AlertTriangle,
  high: ShieldAlert,
};

const toneClassMap = {
  low: "text-emerald-700 border-emerald-200 bg-emerald-50",
  mid: "text-amber-700 border-amber-200 bg-amber-50",
  high: "text-rose-700 border-rose-200 bg-rose-50",
};

export function RecommendationSection({ result, isLoading, error }: RecommendationSectionProps) {
  return (
    <section className="section-anchor pt-10" id="recommendation">
      <SectionHeader
        eyebrow="Salida del sistema"
        title="Resultado del analisis"
        description="Puntaje, clasificacion y razonamiento del caso analizado."
      />

      <AnimatePresence mode="wait">
        {isLoading ? (
          <motion.div
            key="loading"
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            initial={{ opacity: 0, y: 8 }}
            transition={{ duration: 0.28, ease: "easeOut" }}
          >
            <GlassPanel className="flex min-h-64 flex-col items-center justify-center gap-4 p-8 text-center">
              <LoaderCircle className="h-8 w-8 animate-spin text-cyan-600" />
              <div>
                <div className="text-lg font-semibold text-slate-900">Analizando el caso...</div>
                <p className="mt-2 max-w-xl text-sm leading-7 text-slate-600">
                  El sistema esta evaluando los indicadores clinicos del paciente.
                </p>
              </div>
            </GlassPanel>
          </motion.div>
        ) : error ? (
          <motion.div
            key="error"
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            initial={{ opacity: 0, y: 8 }}
            transition={{ duration: 0.28, ease: "easeOut" }}
          >
            <GlassPanel className="flex items-start gap-3 p-6 text-rose-700">
              <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
              <div>
                <div className="font-semibold">No se pudo ejecutar el analisis.</div>
                <p className="mt-2 text-sm leading-7">{error}</p>
              </div>
            </GlassPanel>
          </motion.div>
        ) : result ? (
          <motion.div
            key="result"
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            initial={{ opacity: 0, y: 10 }}
            transition={{ duration: 0.38, ease: [0.22, 1, 0.36, 1] }}
          >
            <ResultContent result={result} />
          </motion.div>
        ) : null}
      </AnimatePresence>
    </section>
  );
}

function ResultContent({ result }: { result: ExplicacionResponse }) {
  const risk = getRiskUi(result.riesgo);
  const Icon = iconMap[risk.tone];
  const summary = buildResultSummary(result);
  const narrative = buildClinicalNarrative(result);
  const sortedRules = [...result.reglas_activadas].sort((a, b) => b.fuerza - a.fuerza);

  const gaugeOption = {
    backgroundColor: "transparent",
    animation: false,
    animationDuration: 0,
    animationDurationUpdate: 0,
    series: [
      {
        type: "gauge",
        min: 0,
        max: 100,
        progress: {
          show: true,
          roundCap: true,
          width: 18,
          itemStyle: { color: risk.accent },
        },
        axisLine: {
          lineStyle: {
            width: 18,
            color: [[1, "rgba(148, 163, 184, 0.15)"]],
          },
        },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        pointer: { show: false },
        detail: {
          valueAnimation: false,
          offsetCenter: [0, "0%"],
          formatter: "{value}",
          color: "#0f172a",
          fontSize: 40,
          fontWeight: 700,
        },
        title: {
          offsetCenter: [0, "46%"],
          color: "rgba(71, 85, 105, 0.8)",
          fontSize: 13,
        },
        data: [{ value: Number(result.puntaje.toFixed(1)), name: "Puntaje" }],
      },
    ],
  };

  return (
    <div className="space-y-6">
      {result.sin_activacion && (
        <div className="flex items-start gap-3 rounded-2xl border border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-800">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
          <div>
            <p className="font-semibold">Ninguna regla se activo para este perfil</p>
            <p className="mt-1 text-amber-700">
              Los valores ingresados no coincidieron con ninguna regla aprendida por RIPPER.
              El puntaje de 50 es un valor neutro por defecto, no una clasificacion clinica real.
              Verifica que los valores ingresados sean correctos o consulta a un profesional.
            </p>
          </div>
        </div>
      )}
      <GlassPanel className="overflow-hidden p-5 sm:p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-xs uppercase tracking-[0.22em] text-cyan-700/80">
              Resultado obtenido
            </div>
            <div className="mt-3 flex items-center gap-3">
              <div
                className="rounded-2xl p-3"
                style={{ backgroundColor: `${risk.accent}22`, color: risk.accent }}
              >
                <Icon className="h-6 w-6" />
              </div>
              <div>
                <h3 className="text-2xl font-semibold text-slate-900 sm:text-[1.9rem]">
                  {result.sin_activacion ? "Sin clasificacion" : risk.label}
                </h3>
                <p className="mt-1 text-sm text-slate-500">
                  {result.sin_activacion
                    ? "El perfil no activo ninguna regla del sistema."
                    : "Clasificacion final del caso analizado."}
                </p>
              </div>
            </div>
          </div>

          <div className="rounded-full border border-slate-200/80 bg-white/80 px-3 py-1 text-xs font-medium uppercase tracking-[0.16em] text-slate-500">
            Reglas activadas {result.reglas_activadas.length}
          </div>
        </div>

        <div className="mt-5 grid gap-5 lg:grid-cols-[220px_minmax(0,1fr)] lg:items-center">
          <div className="h-56 rounded-[1.75rem] border border-slate-200/70 bg-slate-50/70 p-3 sm:h-60">
            <ReactECharts
              lazyUpdate={false}
              notMerge={true}
              option={gaugeOption}
              style={{ height: "100%", width: "100%" }}
            />
          </div>

          <div>
            <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
              Puntaje final
            </div>
            <div className="mt-2 flex items-end gap-2">
              <span className="text-4xl font-semibold leading-none text-slate-900 sm:text-5xl">
                {formatScore(result.puntaje)}
              </span>
              <span className="pb-1 text-sm font-medium text-slate-500">/100</span>
            </div>
            <p className="mt-3 text-base font-semibold leading-6 text-slate-900">
              {summary.headline}
            </p>
          </div>
        </div>

        {result.ajustes_entrada.length > 0 ? (
          <div className="mt-5 border-t border-sky-100 pt-4">
            <div className="text-xs uppercase tracking-[0.22em] text-cyan-700/80">
              Valores ajustados automaticamente
            </div>
            <div className="mt-3 grid gap-3">
              {result.ajustes_entrada.map((adjustment) => (
                <div
                  key={adjustment.variable}
                  className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-200/70 bg-slate-50/75 px-4 py-3 text-sm"
                >
                  <span className="font-medium text-slate-700">
                    {getFieldLabel(adjustment.variable)}
                  </span>
                  <span className="text-slate-500">
                    {formatValue(adjustment.variable, adjustment.valor_original)}
                    <span className="mx-2 text-cyan-500">+</span>
                    <span className="font-semibold text-cyan-700">
                      {formatValue(adjustment.variable, adjustment.valor_ajustado)}
                    </span>
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </GlassPanel>

      {!result.sin_activacion && (
        <GlassPanel className="p-6 sm:p-8">
          <div className="flex items-start gap-4">
            <div className="shrink-0 rounded-2xl border border-cyan-300/30 bg-cyan-50 p-3 text-cyan-700">
              <BookOpen className="h-5 w-5" />
            </div>
            <div>
              <div className="text-xs uppercase tracking-[0.22em] text-cyan-700/80">
                Por que el sistema tomo esta decision
              </div>
              <p className="mt-3 text-base font-semibold leading-relaxed text-slate-900">
                {narrative.intro}
              </p>
              <p className="mt-2 text-sm leading-7 text-slate-600">{narrative.details}</p>
              <p className="mt-2 text-sm leading-7 text-slate-600">{narrative.conclusion}</p>
            </div>
          </div>
        </GlassPanel>
      )}

      <GlassPanel className="p-5 sm:p-6">
        <div className="flex items-start gap-4">
          <div className="shrink-0 rounded-2xl border border-cyan-300/30 bg-cyan-50 p-3 text-cyan-700">
            <BookOpen className="h-5 w-5" />
          </div>
          <div>
            <div className="text-base font-semibold text-slate-900">
              Reglas que influyeron en la decision
            </div>
            <p className="mt-1 text-sm text-slate-500">
              Ordenadas por peso de influencia, de mayor a menor impacto en el resultado final.
            </p>
          </div>
        </div>

        <div className="mt-5 space-y-4">
          {sortedRules.length === 0 && (
            <div className="flex items-start gap-3 rounded-2xl border border-amber-100 bg-amber-50/60 px-4 py-3 text-sm text-amber-700">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
              <span>
                Ninguna regla se activo para este perfil. El puntaje de 50 es un valor neutro de
                respaldo, no el resultado de una inferencia clinica real.
              </span>
            </div>
          )}
          {sortedRules.map((rule, index) => {
            const ruleRisk = getRiskUi(rule.consecuente);
            const ruleNarrative = buildRuleNarrative(rule);

            return (
              <motion.div
                key={rule.numero}
                initial={{ opacity: 0, y: 18 }}
                transition={{ duration: 0.4, delay: index * 0.04 }}
                viewport={{ once: true, amount: 0.2 }}
                whileInView={{ opacity: 1, y: 0 }}
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
                    {ruleNarrative}
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
                </div>
              </motion.div>
            );
          })}
        </div>
      </GlassPanel>
    </div>
  );
}
