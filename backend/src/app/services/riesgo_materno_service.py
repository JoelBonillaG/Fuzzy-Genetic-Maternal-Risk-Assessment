from src.riesgo_materno.logica_difusa.reglas import REGLAS
from src.riesgo_materno.logica_difusa.variables import ESPECIFICACIONES_VARIABLES, SALIDA_DIFUSA
from src.riesgo_materno.prediccion import predecir_caso
from src.riesgo_materno.prediccion.predictor import obtener_curvas_membresia, predecir_caso_con_explicacion


# ── Prediccion ────────────────────────────────────────────────────────────────

def predecir_riesgo_materno(valores_entrada: dict[str, float]) -> dict:
    return predecir_caso(valores_entrada)


def obtener_membresias() -> dict:
    return obtener_curvas_membresia()


def explicar_prediccion(valores_entrada: dict[str, float]) -> dict:
    return predecir_caso_con_explicacion(valores_entrada)


# ── Logica difusa ─────────────────────────────────────────────────────────────

def obtener_definiciones_difusas() -> dict:
    variables = {}
    for nombre, espec in ESPECIFICACIONES_VARIABLES.items():
        variables[nombre] = {
            "limites": list(map(float, espec["limites"])),
            "epsilon": 0.0,
            "categorias": {
                cat: {
                    "puntos_base": list(map(float, puntos)),
                    "puntos_optimizados": list(map(float, puntos)),
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
        "origen_modelo": "reglas_publicadas",
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
