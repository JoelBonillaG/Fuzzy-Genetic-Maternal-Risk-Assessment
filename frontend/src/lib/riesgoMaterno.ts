// ── Prediccion ────────────────────────────────────────────────────────────────

export type RiskTone = "low" | "mid" | "high" | "none";

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
  puntaje: number | null;
  riesgo: string | null;
  sin_activacion: boolean;
  sistema: string;
  origen_modelo: string;
  fuente_reglas: string;
  fallback_ripper: boolean;
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
  puntaje: number | null;
  riesgo: string | null;
  sin_activacion: boolean;
  sistema: string;
  origen_modelo: string;
  fuente_reglas: string;
  fallback_ripper: boolean;
  ajustes_entrada: AjusteEntradaResponse[];
}

// ── Logica difusa ─────────────────────────────────────────────────────────────

export interface VariableDefinicion {
  limites: number[];
  epsilon?: number;
  categorias: Record<string, { puntos_base: number[]; puntos_optimizados: number[] }>;
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
  total_activas?: number;
  fuente_reglas: string;
}

// ── Field specs (labels / units para display) ─────────────────────────────────

const TEMPERATURE_VARIABLE = "temperatura_corporal";
export const OUTPUT_RISK_CUTS = {
  lowToMid: 39.92,
  midToHigh: 73.12,
} as const;

const fieldMetaByApiKey: Record<string, { label: string; unit: string; unitDescription: string }> = {
  edad: {
    label: "Edad",
    unit: "años",
    unitDescription: "Edad del paciente expresada en años.",
  },
  presion_sistolica: {
    label: "Presion sistolica",
    unit: "mmHg",
    unitDescription: "Milimetros de mercurio: unidad usada para medir la presion arterial.",
  },
  presion_diastolica: {
    label: "Presion diastolica",
    unit: "mmHg",
    unitDescription: "Milimetros de mercurio: unidad usada para medir la presion arterial.",
  },
  azucar_sangre: {
    label: "Glucemia",
    unit: "mmol/L",
    unitDescription: "Milimoles por litro: concentracion de glucosa en sangre.",
  },
  temperatura_corporal: {
    label: "Temperatura corporal",
    unit: "°C",
    unitDescription: "Grados Celsius: unidad de temperatura usada para el ingreso del paciente.",
  },
  frecuencia_cardiaca: {
    label: "Frecuencia cardiaca",
    unit: "bpm",
    unitDescription: "Latidos por minuto: cantidad de pulsaciones del corazon en un minuto.",
  },
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
  const payload: Record<string, number> = {};

  for (const key of VARIABLE_ORDER) {
    if (values[key] === undefined || !Number.isFinite(values[key])) {
      throw new Error(`Valor faltante o invalido para ${getFieldLabel(key)}.`);
    }
    payload[key] = toBackendValue(key, values[key]);
  }

  return payload as unknown as PrediccionRequest;
}

export function getFieldLabel(variable: string): string {
  return fieldMetaByApiKey[variable]?.label ?? humanize(variable);
}

export function getFieldUnit(variable: string): string {
  if (variable === TEMPERATURE_VARIABLE) return "°C";
  return fieldMetaByApiKey[variable]?.unit ?? "";
}

export function getFieldUnitDescription(variable: string): string {
  return fieldMetaByApiKey[variable]?.unitDescription ?? "";
}

export function getRiskTone(value: string | null | undefined): RiskTone {
  if (!value) return "none";
  const n = value.toLowerCase();
  if (n.includes("high") || n.includes("alto")) return "high";
  if (n.includes("mid") || n.includes("medio")) return "mid";
  return "low";
}

export function getRiskUi(value: string | null | undefined) {
  const tone = getRiskTone(value);
  return { tone, ...riskToneConfig[tone] };
}

export function formatScore(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "No disponible";
  return scoreFormatter.format(value);
}

export function formatPercentage(value: number) {
  const rounded = Math.round(value * 100);
  if (rounded === 0 && value > 0) return "<1%";
  return `${rounded}%`;
}

export function formatValue(variable: string, value: number) {
  const displayValue = toDisplayValue(variable, value);
  const unit = getFieldUnit(variable);
  return unit ? `${numberFormatter.format(displayValue)} ${unit}` : numberFormatter.format(displayValue);
}

export function toBackendValue(variable: string, value: number) {
  if (variable !== TEMPERATURE_VARIABLE) return value;
  return roundToTwo((value * 9) / 5 + 32);
}

export function toDisplayValue(variable: string, value: number) {
  if (variable !== TEMPERATURE_VARIABLE) return value;
  return roundToTwo(((value - 32) * 5) / 9);
}

export function toDisplayVariableDefinition(variable: string, definition: VariableDefinicion): VariableDefinicion {
  if (variable !== TEMPERATURE_VARIABLE) return definition;

  return {
    ...definition,
    limites: definition.limites.map((value) => toDisplayValue(variable, value)),
    categorias: Object.fromEntries(
      Object.entries(definition.categorias).map(([category, points]) => [
        category,
        {
          puntos_base: points.puntos_base.map((value) => toDisplayValue(variable, value)),
          puntos_optimizados: points.puntos_optimizados.map((value) => toDisplayValue(variable, value)),
        },
      ]),
    ),
  };
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
  const sinClasificacion = result.sin_activacion || result.reglas_activadas.length === 0 || !result.riesgo;

  if (sinClasificacion) {
    return {
      intro: "El sistema no pudo clasificar el riesgo de este paciente.",
      details: "No se activaron reglas suficientes para emitir una clasificacion.",
      conclusion: "Verifique los datos ingresados.",
    };
  }

  const riskLabel = getRiskUi(result.riesgo).label.toLowerCase();
  const score = Math.round(result.puntaje ?? 0);
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

  const finalTone = getRiskUi(result.riesgo).tone;
  const finalActivationKey = finalTone === "high" ? "alto" : finalTone === "mid" ? "medio" : "bajo";
  const finalActivation = result.activaciones[finalActivationKey] ?? 0;
  const rulesCount = result.reglas_activadas.length;

  let conclusion = `Se evaluaron ${rulesCount} regla${rulesCount === 1 ? "" : "s"} del sistema difuso.`;
  if (finalActivation > 0) {
    conclusion += ` Evidencia hacia ${riskLabel}: ${Math.round(finalActivation * 100)}%.`;
  }

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

export async function obtenerReglasDifusas(fuente: string = "AG") {
  return apiRequest<FuzzyReglasResponse>(`/difuso/reglas?fuente=${encodeURIComponent(fuente)}`, { method: "GET" });
}

// ── Internal ──────────────────────────────────────────────────────────────────

const riskToneConfig = {
  low: { accent: "#4ade80", label: "Riesgo bajo" },
  mid: { accent: "#f59e0b", label: "Riesgo medio" },
  high: { accent: "#fb7185", label: "Riesgo alto" },
  none: { accent: "#64748b", label: "Sin clasificacion" },
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

export function getCategoryLabel(categoria: string): string {
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

function roundToTwo(value: number) {
  return Number(value.toFixed(2));
}
