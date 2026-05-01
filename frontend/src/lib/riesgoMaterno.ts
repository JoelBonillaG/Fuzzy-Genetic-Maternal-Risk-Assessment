// ── Prediccion ────────────────────────────────────────────────────────────────

export type RiskTone = "low" | "mid" | "high";

export interface PrediccionRequest {
  edad: number;
  presion_sistolica: number;
  presion_diastolica: number;
  azucar_sangre: number;
  temperatura_corporal: number;
  frecuencia_cardiaca: number;
}

export interface AjusteEntradaResponse {
  variable: keyof PrediccionRequest | string;
  valor_original: number;
  valor_ajustado: number;
}

export interface PrediccionResponse {
  puntaje: number;
  riesgo: string;
  sin_activacion: boolean;
  sistema: string;
  origen_modelo: string;
  ajustes_entrada: AjusteEntradaResponse[];
}

export interface AntecedentExplicacion {
  variable: keyof PrediccionRequest | string;
  categoria: string;
  pertenencia: number;
}

export interface ReglaActivada {
  numero: number;
  antecedentes: AntecedentExplicacion[];
  fuerza: number;
  consecuente: string;
}

export interface ExplicacionResponse {
  entrada_validada: Record<string, number>;
  pertenencias: Record<string, Record<string, number>>;
  reglas_activadas: ReglaActivada[];
  activaciones: Record<string, number>;
  puntaje: number;
  riesgo: string;
  sin_activacion: boolean;
  origen_modelo: string;
  ajustes_entrada: AjusteEntradaResponse[];
}

// ── Logica difusa ─────────────────────────────────────────────────────────────

export interface VariableDefinicion {
  limites: number[];
  categorias: Record<string, number[]>;
}

export interface FuzzyDefinicionesResponse {
  variables: Record<string, VariableDefinicion>;
  salida: {
    nombre: string;
    universo: number[];
    categorias: Record<string, number[]>;
  };
  origen_modelo: string;
}

export interface AntecedentRegla {
  variable: string;
  categoria: string;
}

export interface ReglaSchema {
  numero: number;
  antecedentes: AntecedentRegla[];
  consecuente: string;
  activa: boolean;
}

export interface FuzzyReglasResponse {
  reglas: ReglaSchema[];
  total: number;
  total_activas: number;
}

// ── Algoritmo genetico ────────────────────────────────────────────────────────

export interface SeleccionReglasResponse {
  disponible: boolean;
  cromosoma: number[];
  numeros_reglas_activas: number[];
  cantidad_reglas: number;
  fitness: number;
  metricas_prueba: Record<string, number> | null;
  historial: GeneracionHistorial[];
}

export interface GeneracionHistorial {
  generacion: number;
  mejor_fitness: number;
  fitness_promedio: number;
  aciertos: number;
  cantidad_reglas: number;
}

// ── Field specs (labels / units para display) ─────────────────────────────────

const fieldMetaByApiKey: Record<string, { label: string; unit: string }> = {
  edad:                { label: "Edad",                  unit: "años"   },
  presion_sistolica:   { label: "Presion sistolica",     unit: "mmHg"   },
  presion_diastolica:  { label: "Presion diastolica",    unit: "mmHg"   },
  azucar_sangre:       { label: "Glucemia",              unit: "mmol/L" },
  temperatura_corporal:{ label: "Temperatura corporal",  unit: "°F"     },
  frecuencia_cardiaca: { label: "Frecuencia cardiaca",   unit: "bpm"    },
};

export const VARIABLE_ORDER = [
  "edad",
  "presion_sistolica",
  "presion_diastolica",
  "azucar_sangre",
  "temperatura_corporal",
  "frecuencia_cardiaca",
] as const;

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "/api/v1").replace(/\/$/, "");

// ── Builders ──────────────────────────────────────────────────────────────────

export function buildPredictionPayload(values: Record<string, number>): PrediccionRequest {
  for (const key of VARIABLE_ORDER) {
    if (values[key] === undefined || !Number.isFinite(values[key])) {
      throw new Error(`Valor faltante o invalido para ${getFieldLabel(key)}.`);
    }
  }
  return values as unknown as PrediccionRequest;
}

export function getFieldLabel(variable: string): string {
  return fieldMetaByApiKey[variable]?.label ?? humanize(variable);
}

export function getFieldUnit(variable: string): string {
  return fieldMetaByApiKey[variable]?.unit ?? "";
}

export function getRiskTone(value: string): RiskTone {
  const n = value.toLowerCase();
  if (n.includes("high") || n.includes("alto")) return "high";
  if (n.includes("mid") || n.includes("medio")) return "mid";
  return "low";
}

export function getRiskUi(value: string) {
  const tone = getRiskTone(value);
  return { tone, ...riskToneConfig[tone] };
}

export function formatScore(value: number) {
  return scoreFormatter.format(value);
}

export function formatPercentage(value: number) {
  const rounded = Math.round(value * 100);
  if (rounded === 0 && value > 0) return "<1%";
  return `${rounded}%`;
}

export function formatValue(variable: string, value: number) {
  const unit = getFieldUnit(variable);
  return unit ? `${numberFormatter.format(value)} ${unit}` : numberFormatter.format(value);
}

export function formatAntecedentLabel(antecedent: AntecedentExplicacion) {
  return `${getFieldLabel(antecedent.variable)} es ${humanize(antecedent.categoria)}`;
}

export function buildRuleNarrative(rule: ReglaActivada): string {
  const parts = rule.antecedentes.map(
    (a) => `la ${getFieldLabel(a.variable).toLowerCase()} indica ${getCategoryLabel(a.categoria)}`,
  );
  if (parts.length === 0) return "";
  if (parts.length === 1) return capitalize(parts[0]) + ".";
  const last = parts.pop()!;
  return capitalize(parts.join(", ") + " y " + last) + ".";
}

export interface ClinicalNarrative {
  intro: string;
  details: string;
  conclusion: string;
}

export function buildClinicalNarrative(result: ExplicacionResponse): ClinicalNarrative {
  if (result.sin_activacion) {
    return {
      intro: "El perfil ingresado no coincidio con ninguna regla aprendida por el sistema.",
      details:
        "Ninguna combinacion de indicadores activo reglas del sistema difuso. El puntaje de 50 es un valor neutro de respaldo, no el resultado de una inferencia clinica.",
      conclusion:
        "Verifique que los valores ingresados sean correctos. Si los valores son validos, el caso puede requerir evaluacion medica directa.",
    };
  }

  const riskLabel = getRiskUi(result.riesgo).label.toLowerCase();
  const score = Math.round(result.puntaje);
  const intro = `El sistema clasifico este caso como ${riskLabel} con un puntaje de ${score} sobre 100.`;

  const alerts: string[] = [];
  for (const [variable, categories] of Object.entries(result.pertenencias)) {
    const top = Object.entries(categories).sort(([, a], [, b]) => b - a)[0];
    if (!top) continue;
    const [topCat, topVal] = top;
    if (topVal >= 0.4 && topCat !== "normal" && topCat !== "normoglucemia" && topCat !== "optima") {
      alerts.push(`${getFieldLabel(variable).toLowerCase()} en ${getCategoryLabel(topCat)}`);
    }
  }

  const details =
    alerts.length > 0
      ? `Los indicadores que mas influyeron: ${alerts.join(", ")}.`
      : "Los indicadores clinicos se encuentran dentro de los rangos esperados.";

  const high = result.activaciones["alto"] ?? 0;
  const mid = result.activaciones["medio"] ?? 0;
  const rulesCount = result.reglas_activadas.length;

  let conclusion = `Se evaluaron ${rulesCount} regla${rulesCount === 1 ? "" : "s"} del sistema difuso.`;
  if (high > 0.5) conclusion += ` Evidencia hacia riesgo alto: ${Math.round(high * 100)}%.`;
  else if (mid > 0.4) conclusion += ` Evidencia hacia riesgo medio: ${Math.round(mid * 100)}%.`;

  return { intro, details, conclusion };
}

export function buildResultSummary(
  result: Pick<ExplicacionResponse, "reglas_activadas" | "sin_activacion">,
) {
  const rulesCount = result.reglas_activadas.length;
  if (result.sin_activacion) {
    return {
      headline: "Perfil sin coincidencia en las reglas aprendidas.",
      description: "La entrada se evaluo sin ajustes previos.",
    };
  }
  return {
    headline: `${rulesCount} regla${rulesCount === 1 ? "" : "s"} aportaron evidencia directa al resultado final.`,
    description: "La entrada se evaluo sin ajustes previos.",
  };
}

// ── API clients ───────────────────────────────────────────────────────────────

export async function predecirRiesgoMaterno(payload: PrediccionRequest, signal?: AbortSignal) {
  return apiRequest<PrediccionResponse>("/predicciones/riesgo-materno", {
    body: JSON.stringify(payload),
    method: "POST",
    signal,
  });
}

export async function explicarPrediccion(payload: PrediccionRequest, signal?: AbortSignal) {
  return apiRequest<ExplicacionResponse>("/predicciones/riesgo-materno/explicacion", {
    body: JSON.stringify(payload),
    method: "POST",
    signal,
  });
}

export async function obtenerDefinicionesDifusas() {
  return apiRequest<FuzzyDefinicionesResponse>("/difuso/definiciones", { method: "GET" });
}

export async function obtenerReglasDifusas() {
  return apiRequest<FuzzyReglasResponse>("/difuso/reglas", { method: "GET" });
}

export async function obtenerSeleccionReglas() {
  return apiRequest<SeleccionReglasResponse>("/ga/seleccion-reglas", { method: "GET" });
}

// ── Internal ──────────────────────────────────────────────────────────────────

const riskToneConfig = {
  low: { accent: "#4ade80", label: "Riesgo bajo" },
  mid: { accent: "#f59e0b", label: "Riesgo medio" },
  high: { accent: "#fb7185", label: "Riesgo alto" },
} as const;

const numberFormatter = new Intl.NumberFormat("es-EC", { maximumFractionDigits: 2 });
const scoreFormatter = new Intl.NumberFormat("es-EC", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const categoryLabels: Record<string, string> = {
  adolescente: "adolescente",
  optima: "óptima",
  avanzada: "avanzada",
  muy_avanzada: "muy avanzada",
  hipotension: "hipotensión",
  normal: "normal",
  elevada: "elevada",
  hipertension: "hipertensión",
  hipertension_severa: "hipertensión severa",
  normoglucemia: "normoglucemia",
  hiperglucemia_gestacional: "hiperglucemia gestacional",
  diabetes_manifiesta: "diabetes manifiesta",
  febricular: "febrícula",
  fiebre: "fiebre",
  hiperpirexia: "hiperpirexia",
  bradicardia: "bradicardia",
  taquicardia: "taquicardia",
  bajo: "bajo",
  medio: "medio",
  alto: "alto",
};

function getCategoryLabel(categoria: string): string {
  return categoryLabels[categoria] ?? humanize(categoria);
}

async function apiRequest<T>(path: string, init: RequestInit): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${apiBaseUrl}${path}`, { ...init, headers });

  if (!response.ok) {
    let message = `No se pudo completar la solicitud (${response.status}).`;
    try {
      const data = (await response.json()) as { detail?: string };
      if (typeof data.detail === "string" && data.detail.trim().length > 0) {
        message = data.detail;
      }
    } catch {
      // keep default
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function humanize(value: string) {
  return value.replaceAll("_", " ");
}
