import threading

from src.riesgo_materno.entrenamiento.entrenador import (
    cargar_modelo_optimizado,
    entrenar_con_progreso,
    obtener_membresias_optimizadas,
    obtener_resultado_entrenamiento,
)
from src.riesgo_materno.logica_difusa.reglas import REGLAS
from src.riesgo_materno.logica_difusa.variables import ESPECIFICACIONES_VARIABLES, SALIDA_DIFUSA
from src.riesgo_materno.optimizacion.cromosoma import decodificar_cromosoma
from src.riesgo_materno.prediccion import predecir_caso
from src.riesgo_materno.prediccion.predictor import obtener_curvas_membresia, predecir_caso_con_explicacion

_lock_reentrenamiento = threading.Lock()
_entrenamiento_activo = False


def esta_entrenando() -> bool:
    return _entrenamiento_activo

# ── Prediccion ────────────────────────────────────────────────────────────────

def predecir_riesgo_materno(valores_entrada: dict[str, float]) -> dict:
    return predecir_caso(valores_entrada)


def obtener_membresias() -> dict:
    return obtener_curvas_membresia()


def explicar_prediccion(valores_entrada: dict[str, float]) -> dict:
    return predecir_caso_con_explicacion(valores_entrada)


# ── Algoritmo genetico ────────────────────────────────────────────────────────

def obtener_historial_ga() -> dict:
    resultado = cargar_modelo_optimizado()
    if resultado is None or not resultado.get("historial"):
        return {
            "disponible": False,
            "historial_generaciones": [],
            "fitness": 0.0,
            "generaciones": 0,
            "macro_f1": 0.0,
            "recall_alto": 0.0,
        }
    print("aqui")
    return {
        "disponible": True,
        "historial_generaciones": resultado["historial"],
        "fitness": resultado.get("fitness", 0.0),
        "generaciones": resultado.get("generaciones", 0),
        "macro_f1": resultado.get("macro_f1", 0.0),
        "recall_alto": resultado.get("recall_alto", 0.0),
    }


def obtener_comparacion_ga() -> dict:
    resultado = cargar_modelo_optimizado()
    if resultado is None:
        return {
            "disponible": False,
            "tabla_comparativa": [],
            "mejor_cromosoma": [],
            "membresias_decodificadas": {},
        }
    cromosoma = resultado["mejor_cromosoma"]
    membresias = decodificar_cromosoma(cromosoma)
    membresias_serializables = {
        var: {cat: puntos.tolist() for cat, puntos in cats.items()}
        for var, cats in membresias.items()
    }
    print("tabla_comparativa", resultado.get("tabla_comparativa", []))
    return {
        "disponible": True,
        "tabla_comparativa": resultado.get("tabla_comparativa", []),
        "mejor_cromosoma": cromosoma.tolist(),
        "membresias_decodificadas": membresias_serializables,
    }


def reentrenar_ga_con_progreso(parametros: dict, progress_callback) -> dict:
    """Entrena el GA emitiendo cada generacion via progress_callback."""
    global _entrenamiento_activo
    with _lock_reentrenamiento:
        _entrenamiento_activo = True
        try:
            resultado = entrenar_con_progreso(parametros=parametros, progress_callback=progress_callback)
        finally:
            _entrenamiento_activo = False
    mejor = resultado["mejor_individuo"]
    return {
        "exito": True,
        "fitness": float(mejor.fitness),
        "generaciones": int(len(resultado["historial"]) - 1),
        "macro_f1": float(mejor.macro_f1_validacion),
        "recall_alto": float(mejor.recall_alto_validacion),
    }


def reentrenar_ga(parametros: dict | None = None) -> dict:
    resultado = obtener_resultado_entrenamiento(
        forzar_reentrenamiento=True,
        parametros=parametros,
    )
    mejor = resultado["mejor_individuo"]
    return {
        "exito": True,
        "fitness": float(mejor.fitness),
        "generaciones": int(len(resultado["historial"]) - 1),
        "macro_f1": float(mejor.macro_f1_validacion),
        "recall_alto": float(mejor.recall_alto_validacion),
    }


# ── Logica difusa ─────────────────────────────────────────────────────────────

def obtener_definiciones_difusas() -> dict:
    membresias, origen = obtener_membresias_optimizadas()
    variables = {}
    for nombre, espec in ESPECIFICACIONES_VARIABLES.items():
        variables[nombre] = {
            "limites": list(map(float, espec["limites"])),
            "epsilon": float(espec["epsilon"]),
            "categorias": {
                cat: {
                    "puntos_base": list(map(float, puntos)),
                    "puntos_optimizados": membresias[nombre][cat].tolist(),
                }
                for cat, puntos in espec["categorias"].items()
            },
        }
    return {
        "variables": variables,
        "salida": {
            "nombre": SALIDA_DIFUSA["nombre"],
            "universo": list(map(float, SALIDA_DIFUSA["universo"])),
            "categorias": {k: list(map(float, v)) for k, v in SALIDA_DIFUSA["categorias"].items()},
        },
        "origen_modelo": origen,
    }


def obtener_reglas_difusas() -> dict:
    reglas_formateadas = [
        {
            "numero": regla["numero"],
            "antecedentes": [
                {"variable": var, "categoria": cat}
                for var, cat in regla["antecedentes"]
            ],
            "consecuente": regla["consecuente"],
        }
        for regla in REGLAS
    ]
    return {
        "reglas": reglas_formateadas,
        "total": len(REGLAS),
    }
