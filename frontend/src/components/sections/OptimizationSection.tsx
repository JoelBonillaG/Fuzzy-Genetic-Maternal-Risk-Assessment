import ReactECharts from "echarts-for-react";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { Binary, LoaderCircle, RefreshCw, Terminal, X } from "lucide-react";
import {
  defaultReentrenarParams,
  getFieldLabel,
  obtenerComparacionGA,
  obtenerEstadoGA,
  obtenerHistorialGA,
  reentrenarGAStream,
  type GAComparacionResponse,
  type GAHistorialResponse,
  type GeneracionHistorial,
  type ReentrenarParams,
} from "../../lib/riesgoMaterno";
import { cn } from "../../lib/utils";
import { ChartPanel } from "../ui/ChartPanel";
import { GlassPanel } from "../ui/GlassPanel";
import { SectionHeader } from "../ui/SectionHeader";

// ── Tipos ─────────────────────────────────────────────────────────────────────

interface ParamSpec {
  key: keyof ReentrenarParams;
  label: string;
  hint: string;
  step: number;
  min: number;
  max: number;
  isFloat: boolean;
}

const PARAM_SPECS: ParamSpec[] = [
  { key: "tamano_poblacion",     label: "Tamaño de poblacion",    hint: "Minimo 4",           step: 1,     min: 4,    max: 500,  isFloat: false },
  { key: "cantidad_hijos",       label: "Cantidad de hijos",      hint: "Minimo 2",           step: 1,     min: 2,    max: 500,  isFloat: false },
  { key: "maximo_generaciones",  label: "Maximo de generaciones", hint: "Minimo 1",           step: 1,     min: 1,    max: 1000, isFloat: false },
  { key: "probabilidad_cruce",   label: "Probabilidad de cruce",  hint: "Entre 0.01 y 1.0",   step: 0.01,  min: 0.01, max: 1.0,  isFloat: true  },
  { key: "probabilidad_mutacion",label: "Probabilidad de mutacion",hint: "Entre 0.001 y 1.0", step: 0.001, min: 0.001,max: 1.0,  isFloat: true  },
];

interface SummaryData {
  fitness: number;
  generaciones: number;
  macro_f1: number;
  recall_alto: number;
}

type TabKey = "log" | "cromosoma";

// ── Helper: overlay de actualización ─────────────────────────────────────────

function RefreshOverlay() {
  return (
    <div className="absolute inset-0 z-10 flex items-center justify-center rounded-[1.75rem] bg-white/60 backdrop-blur-[2px]">
      <LoaderCircle className="h-5 w-5 animate-spin text-cyan-600" />
    </div>
  );
}

// ── Seccion principal ─────────────────────────────────────────────────────────

export function OptimizationSection() {
  const [historial, setHistorial] = useState<GAHistorialResponse | null>(null);
  const [comparacion, setComparacion] = useState<GAComparacionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);
  const [serverTraining, setServerTraining] = useState(false);
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [liveGens, setLiveGens] = useState<GeneracionHistorial[]>([]);
  const [liveMembresias, setLiveMembresias] = useState<Record<string, Record<string, number[]>> | null>(null);
  const [activeTab, setActiveTab] = useState<TabKey>("log");
  const logContainerRef = useRef<HTMLDivElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function fetchData(): Promise<GAHistorialResponse | null> {
    setLoading(true);
    setFetchError(null);
    try {
      const [h, c] = await Promise.all([obtenerHistorialGA(), obtenerComparacionGA()]);
      setHistorial(h);
      setComparacion(c);
      return h;
    } catch (e) {
      setFetchError(e instanceof Error ? e.message : "Error al cargar datos del GA.");
      return null;
    } finally {
      setLoading(false);
    }
  }

  function iniciarPolling() {
    async function poll() {
      try {
        const estado = await obtenerEstadoGA();
        if (estado.en_entrenamiento) {
          setServerTraining(true);
          pollRef.current = setTimeout(poll, 2000);
        } else {
          setServerTraining(false);
          await fetchData();
        }
      } catch {
        setServerTraining(false);
        pollRef.current = setTimeout(poll, 3000);
      }
    }
    poll();
  }

  async function handleStartTraining(params: ReentrenarParams) {
    setShowModal(false);
    setLogLines([]);
    setLiveGens([]);
    setLiveMembresias(null);
    setRetraining(true);
    setFetchError(null);
    setActiveTab("log");

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const stream = reentrenarGAStream(params, controller.signal);
      for await (const evento of stream) {
        if (controller.signal.aborted) break;

        if (evento.tipo === "generacion") {
          const gen: GeneracionHistorial = {
            generacion: evento.generacion,
            fitness: evento.fitness,
            fitness_promedio: evento.fitness_promedio,
            macro_f1: evento.macro_f1,
            recall_alto: evento.recall_alto,
          };
          setLiveGens((prev) => [...prev, gen]);
          setLiveMembresias(evento.membresias_decodificadas);
          const line =
            `Gen ${String(evento.generacion).padStart(3, "0")}  ` +
            `fitness=${evento.fitness.toFixed(4)}  ` +
            `macro_f1=${evento.macro_f1.toFixed(4)}  ` +
            `recall_alto=${evento.recall_alto.toFixed(4)}`;
          setLogLines((prev) => [...prev, line]);
        } else if (evento.tipo === "done") {
          setLogLines((prev) => [
            ...prev,
            `─── Completado: ${evento.generaciones} generaciones | fitness=${evento.fitness.toFixed(4)} ───`,
          ]);
        } else if (evento.tipo === "error") {
          setFetchError(evento.mensaje);
        }
      }
    } catch (e) {
      if (!controller.signal.aborted) {
        setFetchError(e instanceof Error ? e.message : "Error en reentrenamiento.");
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setRetraining(false);
      await fetchData();
    }
  }

  // Scroll interno del log sin mover la página
  useEffect(() => {
    const c = logContainerRef.current;
    if (c) c.scrollTop = c.scrollHeight;
  }, [logLines]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (pollRef.current) clearTimeout(pollRef.current);
    };
  }, []);

  useEffect(() => {
    async function iniciar() {
      // Primero verificar si hay entrenamiento activo en el servidor
      try {
        const estado = await obtenerEstadoGA();
        if (estado.en_entrenamiento) {
          setServerTraining(true);
          iniciarPolling();
          return;
        }
      } catch { /* ignorar, continuar con carga normal */ }
      fetchData();
    }
    iniciar();
  }, []);

  // ── Derivados ─────────────────────────────────────────────────────────────

  const noData = !loading && !retraining && historial && !historial.disponible;
  const lastLive = liveGens.length > 0 ? liveGens[liveGens.length - 1] : null;
  const isLive = retraining && liveGens.length > 0;
  const isRefreshing = !retraining && loading && liveGens.length > 0;

  const cardData: SummaryData | null = lastLive
    ? { fitness: lastLive.fitness, generaciones: lastLive.generacion, macro_f1: lastLive.macro_f1, recall_alto: lastLive.recall_alto }
    : historial?.disponible
    ? { fitness: historial.fitness, generaciones: historial.generaciones, macro_f1: historial.macro_f1, recall_alto: historial.recall_alto }
    : null;

  const chartGens: GeneracionHistorial[] | null =
    liveGens.length > 0 ? liveGens : historial?.disponible ? historial.historial_generaciones : null;

  const comparacionDisponible = comparacion?.disponible && comparacion.tabla_comparativa.length > 0;
  const membresiasActivas = liveMembresias ?? (comparacion?.disponible ? comparacion.membresias_decodificadas : null);

  const showLogTab = logLines.length > 0 || retraining;
  const showCromoTab = membresiasActivas !== null;
  const showTabs = showLogTab || showCromoTab;

  return (
    <section className="section-anchor pt-10" id="optimization">

      {/* ── 1. Cabecera ─────────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-4">
        <SectionHeader
          eyebrow="Optimizacion"
          title="Algoritmo genetico"
          description="Convergencia del fitness, comparacion base vs optimizado y cromosoma decodificado."
        />
        <button
          className="mt-10 inline-flex shrink-0 items-center gap-2 rounded-full border border-sky-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-cyan-300 hover:bg-cyan-50 disabled:opacity-60"
          disabled={loading || retraining}
          onClick={() => setShowModal(true)}
          type="button"
        >
          {retraining ? (
            <><LoaderCircle className="h-4 w-4 animate-spin" />{noData ? "Entrenando..." : "Reentrenando..."}</>
          ) : (
            <><RefreshCw className="h-4 w-4" />{noData ? "Entrenar GA" : "Reentrenar GA"}</>
          )}
        </button>
      </div>

      {/* ── 2a. Banner entrenamiento detectado en servidor (tras F5) ───────── */}
      {serverTraining && !retraining && (
        <div className="mt-3 flex items-center gap-2.5 rounded-2xl border border-amber-200 bg-amber-50/80 px-4 py-3">
          <LoaderCircle className="h-3.5 w-3.5 shrink-0 animate-spin text-amber-600" />
          <span className="text-xs text-amber-800">
            Entrenamiento en curso en el servidor. Los resultados apareceran automaticamente al finalizar.
          </span>
        </div>
      )}

      {/* ── 2. Barra de estado compacta (solo durante entrenamiento) ────────── */}
      {retraining && (
        <div className="mt-3 flex items-center gap-2.5 rounded-2xl border border-cyan-200 bg-cyan-50/70 px-4 py-2.5">
          <LoaderCircle className="h-3.5 w-3.5 shrink-0 animate-spin text-cyan-600" />
          {lastLive ? (
            <span className="font-mono text-xs text-slate-700">
              <span className="font-semibold text-cyan-700">
                Gen {String(lastLive.generacion).padStart(3, "0")}
              </span>
              <span className="mx-2 text-slate-300">|</span>
              fitness <span className="font-semibold">{lastLive.fitness.toFixed(4)}</span>
              <span className="mx-2 text-slate-300">|</span>
              macro_f1 <span className="font-semibold">{lastLive.macro_f1.toFixed(4)}</span>
              <span className="mx-2 text-slate-300">|</span>
              recall_alto <span className="font-semibold">{lastLive.recall_alto.toFixed(4)}</span>
            </span>
          ) : (
            <span className="text-xs text-cyan-700">Iniciando algoritmo genetico...</span>
          )}
        </div>
      )}

      {/* ── 3. Estados iniciales ─────────────────────────────────────────────── */}
      {loading && !retraining && liveGens.length === 0 && (
        <GlassPanel className="mt-6 flex min-h-48 items-center justify-center gap-3 p-8 text-slate-600">
          <LoaderCircle className="h-5 w-5 animate-spin text-cyan-600" />
          <span className="text-sm">Cargando datos del algoritmo genetico...</span>
        </GlassPanel>
      )}
      {fetchError && !retraining && (
        <GlassPanel className="mt-4 flex items-center justify-between gap-4 p-5 text-sm text-rose-700">
          <span>{fetchError}</span>
          <button
            onClick={() => fetchData()}
            className="shrink-0 inline-flex items-center gap-1.5 rounded-full border border-rose-200 bg-white px-3 py-1.5 text-xs font-medium text-rose-600 hover:bg-rose-50 transition"
            type="button"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Reintentar
          </button>
        </GlassPanel>
      )}
      {noData && (
        <GlassPanel className="mt-6 flex items-center justify-between gap-4 p-5 text-sm text-amber-700 border-amber-200 bg-amber-50">
          <span>
            El modelo genetico aun no ha sido entrenado. Presione{" "}
            <strong>Entrenar GA</strong> para ejecutar el algoritmo.
          </span>
          <button
            onClick={() => fetchData()}
            className="shrink-0 inline-flex items-center gap-1.5 rounded-full border border-amber-200 bg-white px-3 py-1.5 text-xs font-medium text-amber-700 hover:bg-amber-50 transition"
            type="button"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Recargar
          </button>
        </GlassPanel>
      )}

      {/* ── 4. Cards de métricas clave ───────────────────────────────────────── */}
      {cardData && (
        <div className="relative mt-5">
          <SummaryCards data={cardData} isLive={isLive} />
          {isRefreshing && <RefreshOverlay />}
        </div>
      )}

      {/* ── 5. Gráfica de convergencia — ancho completo ──────────────────────── */}
      {chartGens && (
        <div className="relative mt-6">
          <ConvergenceChart gens={chartGens} isLive={isLive} />
          {isRefreshing && <RefreshOverlay />}
        </div>
      )}

      {/* ── 6. Contenido secundario: tabs Log / Cromosoma ────────────────────── */}
      {showTabs && (
        <motion.div
          className="mt-8 overflow-hidden"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          transition={{ duration: 0.38, ease: [0.22, 1, 0.36, 1] }}
        >
          {/* Barra de tabs */}
          <div className="flex border-b border-sky-100">
            {showLogTab && (
              <TabButton
                active={activeTab === "log"}
                onClick={() => setActiveTab("log")}
                icon={<Terminal className="h-3.5 w-3.5" />}
                label="Log de entrenamiento"
                badge={!retraining && logLines.length > 0 ? String(logLines.length) : undefined}
                spinner={retraining}
              />
            )}
            {showCromoTab && (
              <TabButton
                active={activeTab === "cromosoma"}
                onClick={() => setActiveTab("cromosoma")}
                icon={<Binary className="h-3.5 w-3.5" />}
                label="Cromosoma decodificado"
                live={isLive}
              />
            )}
          </div>

          {/* Contenido del tab activo — crossfade al cambiar */}
          <AnimatePresence mode="wait">
            {activeTab === "log" && showLogTab && (
              <motion.div
                key="log"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.2, ease: "easeOut" }}
              >
                <GlassPanel className="overflow-hidden rounded-tl-none p-0">
                  <div
                    ref={logContainerRef}
                    className="max-h-56 overflow-y-auto bg-slate-950 px-4 py-3 font-mono text-xs leading-6"
                  >
                    {logLines.length === 0 && retraining && (
                      <span className="text-slate-500">Esperando primera generacion...</span>
                    )}
                    {logLines.map((line, i) => (
                      <div
                        key={i}
                        className={
                          line.startsWith("───")
                            ? "mt-1 font-semibold text-cyan-400"
                            : "text-slate-300"
                        }
                      >
                        {line}
                      </div>
                    ))}
                  </div>
                </GlassPanel>
              </motion.div>
            )}

            {activeTab === "cromosoma" && showCromoTab && (
              <motion.div
                key="cromosoma"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.2, ease: "easeOut" }}
                className="relative"
              >
                <ChromosomeTable
                  membresias={membresiasActivas!}
                  isLive={isLive && liveMembresias !== null}
                />
                {isRefreshing && <RefreshOverlay />}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}

      {/* ── 7. Sección comparativa base vs optimizado ────────────────────────── */}
      {comparacionDisponible && (
        <div className="relative mt-12">
          <GlassPanel className="mb-6 p-5 sm:p-6">
            <div className="text-xs uppercase tracking-[0.22em] text-cyan-700/80">Evaluacion comparativa</div>
            <h2 className="mt-1 text-xl font-semibold text-slate-900">Base vs optimizado</h2>
            <p className="mt-1 text-sm text-slate-500">
              Comparacion de metricas entre el cromosoma base y el optimizado sobre el split de validacion. El optimizado corresponde al ultimo entrenamiento guardado.
            </p>
          </GlassPanel>
          <div className="relative">
            <ComparisonPanel comparacion={comparacion!} />
            {isRefreshing && <RefreshOverlay />}
          </div>
        </div>
      )}

      {/* Modal de parámetros */}
      {showModal && (
        <ReentrenarModal
          onConfirm={handleStartTraining}
          onCancel={() => setShowModal(false)}
        />
      )}
    </section>
  );
}

// ── Botón de tab ──────────────────────────────────────────────────────────────

function TabButton({
  active, onClick, icon, label, badge, spinner, live,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  badge?: string;
  spinner?: boolean;
  live?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      type="button"
      className={cn(
        "-mb-px flex items-center gap-2 border-b-2 px-5 py-2.5 text-sm font-medium transition",
        active
          ? "border-cyan-500 text-cyan-800"
          : "border-transparent text-slate-500 hover:text-slate-700",
      )}
    >
      {icon}
      {label}
      {spinner && <LoaderCircle className="h-3 w-3 animate-spin" />}
      {badge && (
        <span className="rounded-full bg-slate-100 px-1.5 py-0.5 text-xs font-normal text-slate-500">
          {badge}
        </span>
      )}
      {live && <span className="h-1.5 w-1.5 rounded-full bg-cyan-500 animate-pulse" />}
    </button>
  );
}

// ── Modal de parámetros ───────────────────────────────────────────────────────

function ReentrenarModal({
  onConfirm,
  onCancel,
}: {
  onConfirm: (params: ReentrenarParams) => void;
  onCancel: () => void;
}) {
  const [params, setParams] = useState<ReentrenarParams>({ ...defaultReentrenarParams });
  const [errors, setErrors] = useState<Partial<Record<keyof ReentrenarParams, string>>>({});

  function handleChange(key: keyof ReentrenarParams, value: string, isFloat: boolean) {
    const num = isFloat ? parseFloat(value) : parseInt(value, 10);
    setParams((prev) => ({ ...prev, [key]: isNaN(num) ? 0 : num }));
    setErrors((prev) => ({ ...prev, [key]: undefined }));
  }

  function validate(): boolean {
    const e: Partial<Record<keyof ReentrenarParams, string>> = {};
    if (params.tamano_poblacion < 4)    e.tamano_poblacion    = "Minimo 4";
    if (params.cantidad_hijos < 2)      e.cantidad_hijos      = "Minimo 2";
    if (params.maximo_generaciones < 1) e.maximo_generaciones = "Minimo 1";
    if (params.probabilidad_cruce <= 0 || params.probabilidad_cruce > 1)
      e.probabilidad_cruce   = "Debe estar entre 0.01 y 1.0";
    if (params.probabilidad_mutacion <= 0 || params.probabilidad_mutacion > 1)
      e.probabilidad_mutacion = "Debe estar entre 0.001 y 1.0";
    setErrors(e);
    return Object.keys(e).length === 0;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm" onClick={onCancel} />
      <motion.div
        initial={{ opacity: 0, scale: 0.96, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
        className="relative w-full max-w-md rounded-3xl border border-sky-100 bg-white/95 shadow-2xl backdrop-blur-xl"
      >
        <div className="flex items-center justify-between border-b border-sky-100 px-6 py-4">
          <div>
            <div className="text-xs uppercase tracking-[0.2em] text-cyan-700/80">Configuracion</div>
            <div className="mt-0.5 text-base font-semibold text-slate-900">Parametros del algoritmo genetico</div>
          </div>
          <button className="rounded-full p-1.5 text-slate-400 hover:bg-slate-100 transition" onClick={onCancel} type="button">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="space-y-4 px-6 py-5">
          {PARAM_SPECS.map((spec) => (
            <div key={spec.key}>
              <label className="mb-1.5 flex items-baseline justify-between text-sm font-medium text-slate-700">
                {spec.label}
                <span className="text-xs font-normal text-slate-400">{spec.hint}</span>
              </label>
              <input
                type="number"
                step={spec.step}
                min={spec.min}
                max={spec.max}
                value={params[spec.key]}
                onChange={(e) => handleChange(spec.key, e.target.value, spec.isFloat)}
                className={cn(
                  "w-full rounded-xl border px-3 py-2 text-sm text-slate-900 outline-none transition focus:ring-2 focus:ring-cyan-300",
                  errors[spec.key]
                    ? "border-rose-300 bg-rose-50 focus:ring-rose-200"
                    : "border-sky-200 bg-white hover:border-cyan-300",
                )}
              />
              {errors[spec.key] && <p className="mt-1 text-xs text-rose-600">{errors[spec.key]}</p>}
            </div>
          ))}
        </div>
        <div className="flex justify-end gap-3 border-t border-sky-100 px-6 py-4">
          <button className="rounded-full border border-sky-200 bg-white px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50 transition" onClick={onCancel} type="button">
            Cancelar
          </button>
          <button className="rounded-full bg-cyan-600 px-5 py-2 text-sm font-semibold text-white hover:bg-cyan-700 active:scale-[0.97] transition" onClick={() => { if (validate()) onConfirm(params); }} type="button">
            Iniciar entrenamiento
          </button>
        </div>
      </motion.div>
    </div>
  );
}

// ── Cards de métricas ─────────────────────────────────────────────────────────

function SummaryCards({ data, isLive }: { data: SummaryData; isLive?: boolean }) {
  const cards = [
    { label: "Mejor fitness",     value: data.fitness.toFixed(4),                                detail: "Fitness maximo observado." },
    { label: "Generaciones",      value: data.generaciones.toString(),                           detail: isLive ? "Generacion actual en curso." : "Generaciones hasta convergencia." },
    { label: "Macro F1",          value: (data.macro_f1 * 100).toFixed(1) + "%",                 detail: "F1 macro ponderado." },
    { label: "Recall alto riesgo",value: (data.recall_alto * 100).toFixed(1) + "%",              detail: "Recall de la clase de riesgo alto." },
  ];

  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      {cards.map((card, i) => (
        <motion.div
          key={card.label}
          initial={{ opacity: 0, y: 14 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.35, delay: i * 0.06 }}
        >
          <GlassPanel className={cn("h-full p-5", isLive && "border-cyan-200")}>
            <div className="flex items-center gap-2">
              <div className="text-xs uppercase tracking-[0.22em] text-slate-500">{card.label}</div>
              {isLive && <span className="h-1.5 w-1.5 rounded-full bg-cyan-500 animate-pulse" />}
            </div>
            <div className="mt-3 font-mono text-3xl font-semibold text-slate-900">{card.value}</div>
            <p className="mt-2 text-xs leading-5 text-slate-500">{card.detail}</p>
          </GlassPanel>
        </motion.div>
      ))}
    </div>
  );
}

// ── Gráfica de convergencia ───────────────────────────────────────────────────

function ConvergenceChart({ gens, isLive }: { gens: GeneracionHistorial[]; isLive?: boolean }) {
  const option = {
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
    grid: { left: 12, right: 18, top: 44, bottom: 12, containLabel: true },
    xAxis: {
      type: "category",
      data: gens.map((g) => g.generacion),
      name: "Generacion",
      nameTextStyle: { color: "rgba(71,85,105,0.8)" },
      axisLabel: { color: "rgba(30,41,59,0.82)" },
      axisLine: { lineStyle: { color: "rgba(148,163,184,0.35)" } },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "rgba(30,41,59,0.82)" },
      splitLine: { lineStyle: { color: "rgba(148,163,184,0.16)" } },
    },
    series: [
      {
        name: "Mejor fitness",
        type: "line",
        smooth: true,
        symbol: "none",
        lineStyle: { width: 3, color: "#38bdf8" },
        areaStyle: { color: "rgba(56,189,248,0.10)" },
        data: gens.map((g) => g.fitness),
      },
      {
        name: "Fitness promedio",
        type: "line",
        smooth: true,
        symbol: "none",
        lineStyle: { width: 2, color: "#818cf8", type: "dashed" },
        data: gens.map((g) => g.fitness_promedio),
      },
    ],
  };

  return (
    <ChartPanel
      title={isLive ? "Convergencia del fitness — en curso" : "Convergencia del fitness por generacion"}
      subtitle=""
    >
      <div className="mb-3 flex flex-wrap items-center justify-center gap-3 text-xs text-slate-600">
        {[
          { label: "Mejor fitness", color: "#38bdf8" },
          { label: "Fitness promedio", color: "#818cf8" },
        ].map((item) => (
          <span key={item.label} className="inline-flex items-center gap-2 rounded-full border border-sky-100 bg-white/75 px-3 py-1.5 shadow-sm">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
            {item.label}
          </span>
        ))}
      </div>
      <div className="h-[320px]">
        <ReactECharts notMerge={true} lazyUpdate={false} option={option} style={{ height: "100%", width: "100%" }} />
      </div>
    </ChartPanel>
  );
}

// ── Panel de comparación ──────────────────────────────────────────────────────

const METRICAS_COMPARACION = ["Macro F1", "Recall alto", "Fitness"] as const;

function ComparisonPanel({ comparacion }: { comparacion: GAComparacionResponse }) {
  const rows = comparacion.tabla_comparativa;

  const chartOption = {
    backgroundColor: "transparent",
    animation: false,
    animationDuration: 0,
    animationDurationUpdate: 0,
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      backgroundColor: "rgba(255,255,255,0.96)",
      borderColor: "rgba(125,211,252,0.65)",
      textStyle: { color: "#0f172a" },
      formatter: (params: { seriesName: string; value: number }[]) =>
        params.map((p) => `${p.seriesName}: ${Number(p.value).toFixed(4)}`).join("<br/>"),
    },
    legend: {
      data: ["Base", "Opt."],
      textStyle: { color: "rgba(15,23,42,0.82)" },
      top: 4,
    },
    grid: { left: 8, right: 8, top: 40, bottom: 8, containLabel: true },
    xAxis: {
      type: "category",
      data: METRICAS_COMPARACION,
      axisLabel: { color: "rgba(30,41,59,0.82)", fontSize: 11 },
      axisLine: { lineStyle: { color: "rgba(148,163,184,0.35)" } },
    },
    yAxis: {
      type: "value",
      min: 0,
      axisLabel: { color: "rgba(30,41,59,0.82)", fontSize: 10 },
      splitLine: { lineStyle: { color: "rgba(148,163,184,0.16)" } },
    },
    series: [
      {
        name: "Base",
        type: "bar",
        barMaxWidth: 40,
        itemStyle: { color: "rgba(148,163,184,0.55)", borderRadius: [5, 5, 0, 0] },
        data: METRICAS_COMPARACION.map((m) => {
          const r = rows.find((row) => row.metrica === m);
          return r ? Number(r.base.toFixed(4)) : 0;
        }),
      },
      {
        name: "Opt.",
        type: "bar",
        barMaxWidth: 40,
        itemStyle: { color: "#38bdf8", borderRadius: [5, 5, 0, 0] },
        data: METRICAS_COMPARACION.map((m) => {
          const r = rows.find((row) => row.metrica === m);
          return r ? Number(r.optimizado.toFixed(4)) : 0;
        }),
      },
    ],
  };

  return (
    <div className="grid gap-6 xl:grid-cols-[1fr_380px]">
      {/* Gráfica agrupada */}
      <ChartPanel
        title="Macro F1, Recall alto y Fitness"
        subtitle="Cromosoma base (gris) vs modelo optimizado (azul) - split de validacion."
      >
        <div className="h-[300px]">
          <ReactECharts notMerge={true} lazyUpdate={false} option={chartOption} style={{ height: "100%", width: "100%" }} />
        </div>
      </ChartPanel>

      {/* Tabla de métricas */}
      <GlassPanel className="p-5 sm:p-6">
        <div className="mb-4 text-xs uppercase tracking-[0.22em] text-cyan-700/80">Tabla comparativa</div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-sky-100 uppercase tracking-[0.12em] text-slate-400">
                <th className="py-2 pr-4 text-left font-medium">Metrica</th>
                <th className="py-2 px-3 text-right font-medium">Base</th>
                <th className="py-2 px-3 text-right font-medium">Opt.</th>
                <th className="py-2 pl-3 text-right font-medium">Δ</th>
              </tr>
            </thead>
            <tbody>
              {METRICAS_COMPARACION.map((metrica) => {
                const row = rows.find((r) => r.metrica === metrica);
                if (!row) return null;
                return (
                    <tr key={row.metrica} className="border-b border-sky-50">
                      <td className="py-2.5 pr-4 text-slate-700">{row.metrica}</td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-400">{row.base.toFixed(3)}</td>
                      <td className="py-2.5 px-3 text-right font-mono font-semibold text-cyan-700">{row.optimizado.toFixed(3)}</td>
                      <td className={cn("py-2.5 pl-3 text-right font-mono", row.delta >= 0 ? "text-emerald-600" : "text-rose-600")}>
                        {row.delta >= 0 ? "+" : ""}{row.delta.toFixed(3)}
                      </td>
                    </tr>
                  );
              })}
            </tbody>
          </table>
        </div>
      </GlassPanel>
    </div>
  );
}

// ── Cromosoma decodificado ────────────────────────────────────────────────────

function ChromosomeTable({
  membresias,
  isLive,
}: {
  membresias: Record<string, Record<string, number[]>>;
  isLive?: boolean;
}) {
  return (
    <GlassPanel className={cn("overflow-hidden rounded-tl-none p-5 sm:p-6", isLive && "border-cyan-200")}>
      <div className="mb-3 flex items-center gap-2">
        <span className="text-xs uppercase tracking-[0.22em] text-cyan-700/80">
          Mejor cromosoma — membresías decodificadas [a, b, c, d]
        </span>
        {isLive && <span className="h-1.5 w-1.5 rounded-full bg-cyan-500 animate-pulse" />}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-sky-100 text-xs uppercase tracking-[0.14em] text-slate-500">
              <th className="py-2 pr-4 text-left">Variable</th>
              <th className="py-2 pr-4 text-left">Categoria</th>
              <th className="py-2 text-left font-mono">a</th>
              <th className="py-2 text-left font-mono">b</th>
              <th className="py-2 text-left font-mono">c</th>
              <th className="py-2 text-left font-mono">d</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(membresias).flatMap(([variable, cats]) =>
              Object.entries(cats).map(([cat, pts], catIdx) => (
                <tr key={`${variable}-${cat}`} className="border-b border-sky-50 hover:bg-sky-50/40">
                  {catIdx === 0 ? (
                    <td className="py-2.5 pr-4 font-semibold text-slate-800" rowSpan={Object.keys(cats).length}>
                      {getFieldLabel(variable)}
                    </td>
                  ) : null}
                  <td className="py-2.5 pr-4 text-slate-600">{cat}</td>
                  {pts.map((v, i) => (
                    <td key={i} className="py-2.5 pr-3 font-mono text-xs text-slate-700">{v.toFixed(2)}</td>
                  ))}
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
    </GlassPanel>
  );
}

