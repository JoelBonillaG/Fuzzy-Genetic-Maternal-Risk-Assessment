"""Servicios HTTP: prediccion + lectura de la seleccion de reglas guardada por el CLI.

El entrenamiento del AG vive en el CLI; la API solo lee resultados.
"""

from src.riesgo_materno.entrenamiento.entrenador import cargar_seleccion_reglas
from src.riesgo_materno.logica_difusa.reglas import REGLAS
from src.riesgo_materno.logica_difusa.variables import ESPECIFICACIONES_VARIABLES, SALIDA_DIFUSA
from src.riesgo_materno.prediccion.predictor import (
    obtener_curvas_membresia,
    predecir_caso,
    predecir_caso_con_explicacion,
)


# ── Prediccion ────────────────────────────────────────────────────────────────

def predecir_riesgo_materno(valores_entrada: dict[str, float]) -> dict:
    return predecir_caso(valores_entrada)


def obtener_membresias() -> dict:
    return obtener_curvas_membresia()


def explicar_prediccion(valores_entrada: dict[str, float]) -> dict:
    return predecir_caso_con_explicacion(valores_entrada)


# ── Algoritmo genetico (solo lectura) ─────────────────────────────────────────

def obtener_seleccion_reglas_actual() -> dict:
    """Devuelve la seleccion de reglas guardada o un payload vacio si no existe."""
    seleccion = cargar_seleccion_reglas()
    if seleccion is None:
        return {
            "disponible": False,
            "cromosoma": [],
            "numeros_reglas_activas": [],
            "cantidad_reglas": 0,
            "fitness": 0.0,
            "metricas_prueba": {},
            "historial": [],
        }
    return {
        "disponible": True,
        "cromosoma": seleccion["cromosoma"].tolist(),
        "numeros_reglas_activas": seleccion["numeros_reglas_activas"],
        "cantidad_reglas": seleccion["cantidad_reglas"],
        "fitness": seleccion["fitness"],
        "metricas_prueba": seleccion["metricas_prueba"],
        "historial": seleccion["historial"],
    }


# ── Logica difusa ─────────────────────────────────────────────────────────────

def obtener_definiciones_difusas() -> dict:
    """Definiciones de las variables y la salida (las membresias no se optimizan)."""
    seleccion = cargar_seleccion_reglas()
    origen = seleccion["ruta_modelo"] if seleccion else "modelo no entrenado"

    variables = {}
    for nombre, espec in ESPECIFICACIONES_VARIABLES.items():
        variables[nombre] = {
            "limites": list(map(float, espec["limites"])),
            "epsilon": float(espec["epsilon"]),
            "categorias": {
                cat: list(map(float, puntos))
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
    """Lista todas las reglas candidatas y marca cuales estan activas en la base optimizada."""
    seleccion = cargar_seleccion_reglas()
    numeros_activos = set(seleccion["numeros_reglas_activas"]) if seleccion else set()

    reglas_formateadas = [
        {
            "numero": regla["numero"],
            "antecedentes": [
                {"variable": var, "categoria": cat}
                for var, cat in regla["antecedentes"]
            ],
            "consecuente": regla["consecuente"],
            "activa": regla["numero"] in numeros_activos,
        }
        for regla in REGLAS
    ]
    return {
        "reglas": reglas_formateadas,
        "total": len(REGLAS),
        "total_activas": len(numeros_activos),
    }
