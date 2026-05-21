"""Predictor: usa las membresias base y, si existe, la base de reglas seleccionada por el AG Pittsburgh.

Si no hay un modelo guardado, cae a las reglas publicadas del sistema difuso.
"""

from ..entrenamiento.entrenador import cargar_seleccion_reglas, construir_membresias_base
from ..logica_difusa.reglas import REGLAS
from ..logica_difusa.motor import SistemaDifusoMamdani
from .validacion_entrada import construir_entrada_lote, validar_valores_entrada


def predecir_caso(valores_entrada):
    """Predice el riesgo materno de un paciente con la base de reglas optimizada."""
    sistema, seleccion = _construir_sistema()
    entradas, ajustes = construir_entrada_lote(valores_entrada)
    inferencia = sistema.inferir_lote(entradas)
    puntaje = _puntaje_para_respuesta(inferencia["puntajes"][0])
    riesgo = inferencia["riesgos"][0]

    return {
        "puntaje": puntaje,
        "riesgo": str(riesgo) if riesgo is not None else None,
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
    puntaje = _puntaje_para_respuesta(resultado["puntaje"])

    return {
        **resultado,
        "puntaje": puntaje,
        "riesgo": resultado["riesgo"] if resultado["riesgo"] is not None else None,
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
    """Construye el sistema difuso con membresias base y la seleccion de reglas guardada.

    Si no existe una seleccion persistida, usa las reglas publicadas del sistema.
    """
    seleccion = cargar_seleccion_reglas()
    if seleccion is None:
        seleccion = {
            "reglas_activas": REGLAS,
            "ruta_modelo": "reglas_publicadas",
            "cantidad_reglas": len(REGLAS),
        }

    membresias = construir_membresias_base()
    sistema = SistemaDifusoMamdani(membresias, reglas=seleccion["reglas_activas"])
    return sistema, seleccion


def _puntaje_para_respuesta(puntaje):
    """Convierte NaN interno a None para emitir JSON valido y no clasificar sin reglas."""
    puntaje = float(puntaje)
    if puntaje != puntaje:
        return None
    return puntaje
