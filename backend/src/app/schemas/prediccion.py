from pydantic import BaseModel


# ── Prediccion ────────────────────────────────────────────────────────────────

class PrediccionRequest(BaseModel):
    edad: float
    presion_sistolica: float
    presion_diastolica: float
    azucar_sangre: float
    temperatura_corporal: float
    frecuencia_cardiaca: float


class AjusteEntradaResponse(BaseModel):
    variable: str
    valor_original: float
    valor_ajustado: float


class PrediccionResponse(BaseModel):
    puntaje: float
    riesgo: str
    sin_activacion: bool
    sistema: str
    origen_modelo: str
    ajustes_entrada: list[AjusteEntradaResponse]
    cantidad_reglas_activas: int


class AntecedentExplicacion(BaseModel):
    variable: str
    categoria: str
    pertenencia: float


class ReglaActivada(BaseModel):
    numero: int
    antecedentes: list[AntecedentExplicacion]
    fuerza: float
    consecuente: str


class ExplicacionResponse(BaseModel):
    entrada_validada: dict[str, float]
    pertenencias: dict[str, dict[str, float]]
    reglas_activadas: list[ReglaActivada]
    activaciones: dict[str, float]
    puntaje: float
    riesgo: str
    sin_activacion: bool
    origen_modelo: str
    ajustes_entrada: list[AjusteEntradaResponse]
    cantidad_reglas_activas: int


class CurvaMembresia(BaseModel):
    puntos_x: list[float]
    puntos_y: list[float]


class MembresiasResponse(BaseModel):
    variables: dict[str, dict[str, CurvaMembresia]]
    origen_modelo: str


# ── Algoritmo genetico (lectura de la seleccion guardada) ─────────────────────

class GeneracionHistorial(BaseModel):
    generacion: int
    mejor_fitness: float
    fitness_promedio: float
    aciertos: int
    cantidad_reglas: int


class MetricasPrueba(BaseModel):
    aciertos: int
    total: int
    accuracy: float
    fitness: float


class SeleccionReglasResponse(BaseModel):
    disponible: bool
    cromosoma: list[int]
    numeros_reglas_activas: list[int]
    cantidad_reglas: int
    fitness: float
    metricas_prueba: dict | MetricasPrueba | None = None
    historial: list[GeneracionHistorial] = []


# ── Logica difusa ─────────────────────────────────────────────────────────────

class VariableDefinicion(BaseModel):
    limites: list[float]
    categorias: dict[str, list[float]]


class SalidaDifusa(BaseModel):
    nombre: str
    universo: list[float]
    categorias: dict[str, list[float]]


class FuzzyDefinicionesResponse(BaseModel):
    variables: dict[str, VariableDefinicion]
    salida: SalidaDifusa
    origen_modelo: str


class AntecedentRegla(BaseModel):
    variable: str
    categoria: str


class ReglaSchema(BaseModel):
    numero: int
    antecedentes: list[AntecedentRegla]
    consecuente: str
    activa: bool


class FuzzyReglasResponse(BaseModel):
    reglas: list[ReglaSchema]
    total: int
    total_activas: int
