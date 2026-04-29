import type { PatientFormData } from "../data/mockData";

export type RiskTone = "low" | "mid" | "high";

// ── Prediccion ────────────────────────────────────────────────────────────────

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
  // true cuando ninguna regla RIPPER activo — puntaje=50 es neutro, no un resultado clinico real
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
  // true cuando ninguna regla RIPPER activo — puntaje=50 es neutro, no un resultado clinico real
  sin_activacion: boolean;
  origen_modelo: string;
  ajustes_entrada: AjusteEntradaResponse[];
}

// ── Algoritmo genetico ────────────────────────────────────────────────────────

export interface GeneracionHistorial {
  generacion: number;
  fitness: number;
  fitness_promedio: number;
  macro_f1: number;
  recall_alto: number;
}

export interface GAHistorialResponse {
  disponible: boolean;
  historial_generaciones: GeneracionHistorial[];
  fitness: number;
  generaciones: number;
  macro_f1: number;
  recall_alto: number;
}

export interface ComparacionRow {
  metrica: string;
  base: number;
  optimizado: number;
  delta: number;
}

export interface GAComparacionResponse {
  disponible: boolean;
  tabla_comparativa: ComparacionRow[];
  mejor_cromosoma: number[];
  membresias_decodificadas: Record<string, Record<string, number[]>>;
}

// ── Logica difusa ─────────────────────────────────────────────────────────────

export interface CategoriaDefinicion {
  puntos_base: number[];
  puntos_optimizados: number[];
}

export interface VariableDefinicion {
  limites: number[];
  epsilon: number;
  categorias: Record<string, CategoriaDefinicion>;
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
}

export interface FuzzyReglasResponse {
  reglas: ReglaSchema[];
  total: number;
}

// ── Field specs ───────────────────────────────────────────────────────────────

export const numericFieldSpecs = [
  {
    formKey: "age",
    apiKey: "edad",
    label: "Edad",
    helper: "Edad materna en anos",
    placeholder: "31",
    unit: "anos",
    min: 10,
    max: 70,
    step: 1,
  },
  {
    formKey: "systolicBP",
    apiKey: "presion_sistolica",
    label: "Presion sistolica",
    helper: "Valor superior de referencia",
    placeholder: "146",
    unit: "mmHg",
    min: 70,
    max: 160,
    step: 1,
  },
  {
    formKey: "diastolicBP",
    apiKey: "presion_diastolica",
    label: "Presion diastolica",
    helper: "Valor inferior de referencia",
    placeholder: "94",
    unit: "mmHg",
    min: 49,
    max: 100,
    step: 1,
  },
  {
    formKey: "bloodGlucose",
    apiKey: "azucar_sangre",
    label: "Glucosa / BS",
    helper: "Medicion de glucosa en sangre",
    placeholder: "7.8",
    unit: "mmol/L",
    min: 6,
    max: 19,
    step: 0.1,
  },
  {
    formKey: "bodyTemperature",
    apiKey: "temperatura_corporal",
    label: "Temperatura corporal",
    helper: "Registro termico observado",
    placeholder: "100.1",
    unit: "F",
    min: 97,
    max: 103,
    step: 0.1,
  },
  {
    formKey: "heartRate",
    apiKey: "frecuencia_cardiaca",
    label: "Frecuencia cardiaca",
    helper: "Frecuencia cardiaca observada",
    placeholder: "88",
    unit: "bpm",
    min: 60,
    max: 90,
    step: 1,
  },
] as const;

export type NumericFormField = (typeof numericFieldSpecs)[number]["formKey"];
type NumericFieldSpec = (typeof numericFieldSpecs)[number];

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "/api/v1").replace(/\/$/, "");

const fieldSpecByApiKey = Object.fromEntries(
  numericFieldSpecs.map((spec) => [spec.apiKey, spec]),
) as Record<(typeof numericFieldSpecs)[number]["apiKey"], NumericFieldSpec>;

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

// ── Builders ──────────────────────────────────────────────────────────────────

export function buildPredictionPayload(formData: PatientFormData): PrediccionRequest {
  return numericFieldSpecs.reduce((payload, spec) => {
    payload[spec.apiKey] = parseNumericField(formData[spec.formKey], spec.label);
    return payload;
  }, {} as PrediccionRequest);
}

export function getFieldLabel(variable: string) {
  return fieldSpecByApiKey[variable as keyof typeof fieldSpecByApiKey]?.label ?? humanize(variable);
}

export function getFieldUnit(variable: string) {
  return fieldSpecByApiKey[variable as keyof typeof fieldSpecByApiKey]?.unit ?? "";
}

export function getFieldRange(formKey: NumericFormField) {
  const spec = numericFieldSpecs.find((item) => item.formKey === formKey);
  return spec ? `${spec.min} - ${spec.max} ${spec.unit}` : "";
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

export function buildRuleTitle(rule: ReglaActivada) {
  return `Si ${rule.antecedentes.map(formatAntecedentLabel).join(" y ")}`;
}

const categoryLabels: Record<string, string> = {
  bajo: "bajo",
  baja: "baja",
  normal: "normal",
  alto: "alto",
  alta: "alta",
  elevado: "elevado",
  elevada: "elevada",
  joven: "joven",
  adulta: "adulta",
  avanzada: "avanzada",
  fiebre: "con fiebre",
};

function getCategoryLabel(categoria: string): string {
  return categoryLabels[categoria] ?? humanize(categoria);
}

export function buildRuleNarrative(rule: ReglaActivada): string {
  const parts = rule.antecedentes.map(
    (a) => `la ${getFieldLabel(a.variable).toLowerCase()} está ${getCategoryLabel(a.categoria)}`,
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
  // Caso sin activacion: ninguna regla se disparo — puntaje 50 es fallback neutro, no clasificacion real
  if (result.sin_activacion) {
    return {
      intro: "El perfil ingresado no coincidio con ninguna regla aprendida por el sistema.",
      details:
        "Ninguna combinacion de indicadores activo reglas del sistema difuso. El puntaje de 50 es un valor neutro de respaldo, no el resultado de una inferencia clinica.",
      conclusion:
        "Verifique que los valores ingresados sean correctos. Si los valores son validos, el caso puede requerir evaluacion medica directa ya que el sistema no tiene evidencia suficiente para clasificarlo.",
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
    if (topVal >= 0.4 && topCat !== "normal") {
      alerts.push(`${getFieldLabel(variable).toLowerCase()} ${getCategoryLabel(topCat)}`);
    }
  }

  const details =
    alerts.length > 0
      ? `Los indicadores que mas influyeron: ${alerts.join(", ")}.`
      : "Los indicadores clinicos se encuentran dentro de los rangos esperados.";

  const high = result.activaciones["alto"] ?? 0;
  const mid = result.activaciones["medio"] ?? 0;
  const low = result.activaciones["bajo"] ?? 0;
  const rulesCount = result.reglas_activadas.length;

  let conclusion = `Se evaluaron ${rulesCount} regla${rulesCount === 1 ? "" : "s"} del sistema difuso.`;

  if (high > 0.5) {
    conclusion += ` Evidencia hacia riesgo alto: ${Math.round(high * 100)}%.`;
  } else if (mid > 0.4) {
    conclusion += ` Evidencia hacia riesgo medio: ${Math.round(mid * 100)}%.`;
  } else if (low > 0.5) {
    conclusion += ` Evidencia hacia riesgo bajo: ${Math.round(low * 100)}%.`;
  }

  return { intro, details, conclusion };
}

export function buildResultSummary(
  result: Pick<ExplicacionResponse, "reglas_activadas" | "ajustes_entrada" | "sin_activacion">,
) {
  const adj = result.ajustes_entrada.length;
  const rulesCount = result.reglas_activadas.length;
  const adjustmentText =
    adj > 0
      ? `Se normalizaron ${adj} valor${adj === 1 ? "" : "es"} antes del analisis para mantener la entrada dentro del rango esperado.`
      : "La entrada se evaluo sin ajustes previos.";

  if (result.sin_activacion) {
    return {
      headline: "Perfil sin coincidencia en las reglas aprendidas.",
      description: adjustmentText,
    };
  }

  return {
    headline: `${rulesCount} regla${rulesCount === 1 ? "" : "s"} aportaron evidencia directa al resultado final.`,
    description: adjustmentText,
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

export async function obtenerEstadoGA() {
  return apiRequest<{ en_entrenamiento: boolean }>("/ga/estado", { method: "GET" });
}

export async function obtenerHistorialGA() {
  return apiRequest<GAHistorialResponse>("/ga/historial", { method: "GET" });
}

export async function obtenerComparacionGA() {
  return apiRequest<GAComparacionResponse>("/ga/comparacion", { method: "GET" });
}

export interface ReentrenarParams {
  tamano_poblacion: number;
  cantidad_hijos: number;
  maximo_generaciones: number;
  probabilidad_cruce: number;
  probabilidad_mutacion: number;
}

export const defaultReentrenarParams: ReentrenarParams = {
  tamano_poblacion: 50,
  cantidad_hijos: 50,
  maximo_generaciones: 60,
  probabilidad_cruce: 0.85,
  probabilidad_mutacion: 0.04,
};

export interface GAProgresoDone {
  tipo: "done";
  exito: boolean;
  fitness: number;
  generaciones: number;
  macro_f1: number;
  recall_alto: number;
}

export interface GAProgresoGeneracion {
  tipo: "generacion";
  generacion: number;
  fitness: number;
  fitness_promedio: number;
  macro_f1: number;
  recall_alto: number;
  membresias_decodificadas: Record<string, Record<string, number[]>>;
}

export interface GAProgresoError {
  tipo: "error";
  mensaje: string;
}

export type GAProgresoEvento = GAProgresoGeneracion | GAProgresoDone | GAProgresoError;

export async function reentrenarGA(params: ReentrenarParams = defaultReentrenarParams) {
  return apiRequest<{ exito: boolean; fitness: number; generaciones: number; macro_f1: number; recall_alto: number }>(
    "/ga/reentrenar",
    { method: "POST", body: JSON.stringify(params) },
  );
}

export async function* reentrenarGAStream(
  params: ReentrenarParams,
  signal?: AbortSignal,
): AsyncGenerator<GAProgresoEvento> {
  const response = await fetch(`${apiBaseUrl}/ga/reentrenar-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
    signal,
  });

  if (!response.ok) {
    let message = `Error ${response.status}`;
    try {
      const data = (await response.json()) as { detail?: string };
      if (typeof data.detail === "string") message = data.detail;
    } catch { /* keep default */ }
    throw new Error(message);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop()!;
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          yield JSON.parse(line.slice(6)) as GAProgresoEvento;
        }
      }
    }
  } finally {
    reader.cancel();
  }
}

// ── Internal ──────────────────────────────────────────────────────────────────

function parseNumericField(value: string, label: string) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    throw new Error(`Ingrese un valor numerico valido para ${label.toLowerCase()}.`);
  }
  return n;
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
