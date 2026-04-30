"""Predictor: usa las membresias base + la base de reglas seleccionada por el AG Pittsburgh.

La prediccion no dispara entrenamiento. Si no existe el modelo guardado,
indica al usuario que ejecute el CLI de entrenamiento primero.
"""

from ..entrenamiento.entrenador import cargar_seleccion_reglas, construir_membresias_base
from ..logica_difusa.motor import SistemaDifusoMamdani
from .validacion_entrada import construir_entrada_lote, validar_valores_entrada


_MENSAJE_SIN_MODELO = (
    "No hay un modelo de reglas optimizado guardado. "
    "Ejecute primero el entrenamiento: "
    "python -m src.riesgo_materno.herramientas.entrenar_ag"
)


def predecir_caso(valores_entrada):
    """Predice el riesgo materno de un paciente con la base de reglas optimizada."""
    sistema, seleccion = _construir_sistema()
    entradas, ajustes = construir_entrada_lote(valores_entrada)
    inferencia = sistema.inferir_lote(entradas)

    return {
        "puntaje": float(inferencia["puntajes"][0]),
        "riesgo": str(inferencia["riesgos"][0]),
        "sin_activacion": bool(inferencia["sin_activacion"][0]),
        "sistema": "Mamdani con seleccion Pittsburgh",
        "origen_modelo": seleccion["ruta_modelo"],
        "ajustes_entrada": ajustes,
        "cantidad_reglas_activas": seleccion["cantidad_reglas"],
    }


def predecir_caso_con_explicacion(valores_entrada):
    """Predice el riesgo exponiendo pertenencias, reglas activadas y activaciones por nivel."""
    sistema, seleccion = _construir_sistema()
    entradas, ajustes = validar_valores_entrada(valores_entrada)
    resultado = sistema.inferir_con_explicacion(entradas)

    return {
        **resultado,
        "entrada_validada": entradas,
        "origen_modelo": seleccion["ruta_modelo"],
        "ajustes_entrada": ajustes,
        "sin_activacion": resultado["sin_activacion"],
        "cantidad_reglas_activas": seleccion["cantidad_reglas"],
    }


def obtener_curvas_membresia():
    """Curvas trapezoidales base para visualizar en el frontend (las membresias no se optimizan)."""
    sistema, seleccion = _construir_sistema()

    variables = {}
    for variable, universo in sistema.universos_entrada.items():
        puntos_x = universo.tolist()
        variables[variable] = {
            categoria: {
                "puntos_x": puntos_x,
                "puntos_y": curva.tolist(),
            }
            for categoria, curva in sistema.curvas_entrada[variable].items()
        }

    return {
        "variables": variables,
        "origen_modelo": seleccion["ruta_modelo"],
    }


def _construir_sistema():
    """Construye el sistema difuso con membresias base y la seleccion de reglas guardada."""
    seleccion = cargar_seleccion_reglas()
    if seleccion is None:
        raise FileNotFoundError(_MENSAJE_SIN_MODELO)

    membresias = construir_membresias_base()
    sistema = SistemaDifusoMamdani(membresias, reglas=seleccion["reglas_activas"])
    return sistema, seleccion
