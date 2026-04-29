import React, { Component, useEffect, useRef, useState, type ChangeEvent } from "react";
import { AnimatePresence, motion, useAnimationControls } from "framer-motion";
import { ActivitySquare, AlertTriangle, FlaskConical, Microscope, Stethoscope } from "lucide-react";
import { DifusoSection } from "./components/sections/DifusoSection";
import { OptimizationSection } from "./components/sections/OptimizationSection";
import { PatientDataSection } from "./components/sections/PatientDataSection";
import { RecommendationSection } from "./components/sections/RecommendationSection";
import { initialPatientForm, type PatientFormData } from "./data/mockData";
import {
  buildPredictionPayload,
  explicarPrediccion,
  type ExplicacionResponse,
  type PrediccionRequest,
} from "./lib/riesgoMaterno";
import { cn } from "./lib/utils";

const navigation = [
  { key: "prediccion", label: "Prediccion", icon: Stethoscope },
  { key: "difuso", label: "Sistema difuso", icon: FlaskConical },
  { key: "ga", label: "Algoritmo genetico", icon: Microscope },
] as const;

type SectionKey = (typeof navigation)[number]["key"];

export default function App() {
  const [formData, setFormData] = useState<PatientFormData>(initialPatientForm);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [explanationResult, setExplanationResult] = useState<ExplicacionResponse | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<SectionKey>("prediccion");
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  function handleFieldChange(
    field: keyof PatientFormData,
    event: ChangeEvent<HTMLInputElement>,
  ) {
    if (explanationResult || analysisError || isAnalyzing) {
      resetAnalysisState();
    }
    setFormData((current) => ({ ...current, [field]: event.target.value }));
  }

  async function handleAnalyze() {
    if (isAnalyzing) return;

    let payload: PrediccionRequest;
    try {
      payload = buildPredictionPayload(formData);
    } catch (error) {
      setAnalysisError(getErrorMessage(error));
      return;
    }

    resetAnalysisState();
    setIsAnalyzing(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const result = await explicarPrediccion(payload, controller.signal);
      if (abortRef.current !== controller) return;
      abortRef.current = null;
      setExplanationResult(result);
      setAnalysisError(null);
      setIsAnalyzing(false);
    } catch (error) {
      if (controller.signal.aborted) return;
      if (abortRef.current === controller) abortRef.current = null;
      setAnalysisError(getErrorMessage(error));
      setIsAnalyzing(false);
    }
  }

  function resetAnalysisState() {
    abortRef.current?.abort();
    abortRef.current = null;
    setExplanationResult(null);
    setAnalysisError(null);
    setIsAnalyzing(false);
  }

  const activeSectionLabel =
    navigation.find((item) => item.key === activeSection)?.label ?? "Prediccion";

  return (
    <div className="relative overflow-hidden">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute left-[-8rem] top-16 h-72 w-72 rounded-full bg-cyan-300/20 blur-3xl" />
        <div className="absolute right-[-12rem] top-32 h-96 w-96 rounded-full bg-blue-300/15 blur-3xl" />
      </div>

      <header className="sticky top-0 z-40 border-b border-sky-100/90 bg-white/78 backdrop-blur-xl">
        <div className="mx-auto max-w-[1480px] px-4 py-4 sm:px-6 xl:px-8">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
            <div className="flex items-center gap-4">
              <div className="rounded-2xl border border-cyan-300/30 bg-cyan-50 p-3 text-cyan-700">
                <ActivitySquare className="h-6 w-6" />
              </div>
              <div>
                <div className="text-base font-semibold leading-tight text-slate-900">
                  Riesgo materno
                </div>
                <div className="mt-1 text-xs uppercase tracking-[0.18em] text-slate-400">
                  Logica difusa Mamdani + AG
                </div>
              </div>
            </div>

            <nav>
              <div className="flex items-center gap-2.5">
                {navigation.map((item) => {
                  const Icon = item.icon;
                  const isActive = activeSection === item.key;
                  return (
                    <button
                      key={item.key}
                      className={cn(
                        "inline-flex items-center gap-2 rounded-full border px-4 py-2.5 text-sm font-medium transition",
                        isActive
                          ? "border-cyan-400/40 bg-cyan-50 text-cyan-900 shadow-sm"
                          : "border-sky-100 bg-white/80 text-slate-700 hover:border-cyan-300/45 hover:bg-cyan-50 hover:text-slate-900",
                      )}
                      onClick={() => setActiveSection(item.key)}
                      type="button"
                    >
                      <Icon className="h-4 w-4 shrink-0" />
                      {item.label}
                    </button>
                  );
                })}
              </div>
            </nav>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1480px] px-4 pb-16 sm:px-6 xl:px-8">
        {analysisError && activeSection !== "prediccion" ? (
          <div className="mt-4 flex items-start gap-3 rounded-[1.75rem] border border-rose-200 bg-rose-50 px-5 py-3 text-sm text-rose-700">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{analysisError}</span>
          </div>
        ) : null}


        <AnimatedSection isActive={activeSection === "prediccion"}>
          <div className="space-y-6">
            <PatientDataSection
              formData={formData}
              isAnalyzing={isAnalyzing}
              onAnalyze={handleAnalyze}
              onFieldChange={handleFieldChange}
            />
            <AnimatePresence>
              {(isAnalyzing || explanationResult || analysisError) && (
                <motion.div
                  key="recommendation"
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -10, scale: 0.99 }}
                  initial={{ opacity: 0, y: 22, scale: 0.99 }}
                  transition={{ duration: 0.42, ease: [0.22, 1, 0.36, 1] }}
                >
                  <RecommendationSection
                    error={analysisError}
                    isLoading={isAnalyzing}
                    result={explanationResult}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </AnimatedSection>

        <AnimatedSection isActive={activeSection === "difuso"}>
          <DifusoSection explanationResult={explanationResult} />
        </AnimatedSection>

        <AnimatedSection isActive={activeSection === "ga"}>
          <SectionErrorBoundary>
            <OptimizationSection />
          </SectionErrorBoundary>
        </AnimatedSection>
      </main>
    </div>
  );
}

class SectionErrorBoundary extends Component<
  { children: React.ReactNode },
  { error: string | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error: unknown) {
    return { error: error instanceof Error ? error.message : "Error inesperado." };
  }
  render() {
    if (this.state.error) {
      return (
        <div className="mt-6 flex items-start gap-3 rounded-[1.75rem] border border-rose-200 bg-rose-50 px-5 py-4 text-sm text-rose-700">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{this.state.error}</span>
        </div>
      );
    }
    return this.props.children;
  }
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : "Ocurrio un error inesperado.";
}

function AnimatedSection({ isActive, children }: { isActive: boolean; children: React.ReactNode }) {
  const controls = useAnimationControls();
  const prevIsActiveRef = useRef(isActive);

  useEffect(() => {
    if (isActive && !prevIsActiveRef.current) {
      void controls.start({
        opacity: [0, 1],
        y: [18, 0],
        scale: [0.99, 1],
        transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1] },
      });
    }
    prevIsActiveRef.current = isActive;
  }, [controls, isActive]);

  return (
    <motion.div
      animate={controls}
      className={isActive ? undefined : "hidden"}
      initial={{ opacity: isActive ? 1 : 0, y: 0, scale: 1 }}
    >
      {children}
    </motion.div>
  );
}
