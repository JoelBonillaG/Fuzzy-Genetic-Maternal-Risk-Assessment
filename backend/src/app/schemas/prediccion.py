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
    puntaje: float | None
    riesgo: str | None
    sistema: str
    origen_modelo: str
    ajustes_entrada: list[AjusteEntradaResponse]


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
    puntaje: float | None
    riesgo: str | None
    origen_modelo: str
    ajustes_entrada: list[AjusteEntradaResponse]


class CurvaMembresia(BaseModel):
    puntos_x: list[float]
    puntos_y: list[float]


class MembresiasResponse(BaseModel):
    variables: dict[str, dict[str, CurvaMembresia]]
    origen_modelo: str


# ── Logica difusa ─────────────────────────────────────────────────────────────

class CategoriaDefinicion(BaseModel):
    puntos_base: list[float]
    puntos_optimizados: list[float]


class VariableDefinicion(BaseModel):
    limites: list[float]
    epsilon: float
    categorias: dict[str, CategoriaDefinicion]


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


class FuzzyReglasResponse(BaseModel):
    reglas: list[ReglaSchema]
    total: int
