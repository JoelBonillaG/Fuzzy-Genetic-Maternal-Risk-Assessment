"""Predictor: usa las reglas publicadas del sistema y RIPPER como respaldo."""

import numpy as np

from ..logica_difusa.reglas import REGLAS, REGLAS_RIPPER
from ..logica_difusa.motor import SistemaDifusoMamdani
from ..logica_difusa.variables import ESPECIFICACIONES_VARIABLES
from .validacion_entrada import construir_entrada_lote, validar_valores_entrada


def predecir_caso(valores_entrada):
    """Predice el riesgo materno de un paciente con la base de reglas optimizada."""
    sistema, seleccion = _construir_sistema()
    entradas, ajustes = construir_entrada_lote(valores_entrada)
    inferencia = sistema.inferir_lote(entradas)
    seleccion_usada = seleccion

    if bool(inferencia["sin_activacion"][0]):
        sistema, seleccion_usada = _construir_sistema_ripper()
        inferencia = sistema.inferir_lote(entradas)

    puntaje = _puntaje_para_respuesta(inferencia["puntajes"][0])
    riesgo = inferencia["riesgos"][0]

    return {
        "puntaje": puntaje,
        "riesgo": str(riesgo) if riesgo is not None else None,
        "sin_activacion": bool(inferencia["sin_activacion"][0]),
        "sistema": seleccion_usada["sistema"],
        "origen_modelo": seleccion_usada["ruta_modelo"],
        "fuente_reglas": seleccion_usada["fuente_reglas"],
        "fallback_ripper": seleccion_usada["fuente_reglas"] == "RIPPER",
        "ajustes_entrada": ajustes,
        "cantidad_reglas_activas": seleccion_usada["cantidad_reglas"],
    }


def predecir_caso_con_explicacion(valores_entrada):
    """Predice el riesgo exponiendo pertenencias, reglas activadas y activaciones por nivel."""
    sistema, seleccion = _construir_sistema()
    entradas, ajustes = validar_valores_entrada(valores_entrada)
    resultado = sistema.inferir_con_explicacion(entradas)
    seleccion_usada = seleccion

    if resultado["sin_activacion"]:
        sistema, seleccion_usada = _construir_sistema_ripper()
        resultado = sistema.inferir_con_explicacion(entradas)

    puntaje = _puntaje_para_respuesta(resultado["puntaje"])

    return {
        **resultado,
        "puntaje": puntaje,
        "riesgo": resultado["riesgo"] if resultado["riesgo"] is not None else None,
        "entrada_validada": entradas,
        "origen_modelo": seleccion_usada["ruta_modelo"],
        "sistema": seleccion_usada["sistema"],
        "fuente_reglas": seleccion_usada["fuente_reglas"],
        "fallback_ripper": seleccion_usada["fuente_reglas"] == "RIPPER",
        "ajustes_entrada": ajustes,
        "sin_activacion": resultado["sin_activacion"],
        "cantidad_reglas_activas": seleccion_usada["cantidad_reglas"],
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
    """Construye el sistema difuso con las reglas publicadas del algoritmo genetico."""
    seleccion = {
        "reglas_activas": REGLAS,
        "ruta_modelo": "src/riesgo_materno/reglas/reglas_sistema_difuso.json",
        "cantidad_reglas": len(REGLAS),
        "fuente_reglas": "AG",
        "sistema": "Mamdani con reglas del algoritmo genetico",
    }
    membresias = construir_membresias_base()
    sistema = SistemaDifusoMamdani(membresias, reglas=seleccion["reglas_activas"])
    return sistema, seleccion


def _construir_sistema_ripper():
    """Construye el sistema alterno con reglas RIPPER para casos sin activacion AG."""
    seleccion = {
        "reglas_activas": REGLAS_RIPPER,
        "ruta_modelo": "src/riesgo_materno/reglas/reglas_sistema_difuso_ripper.json",
        "cantidad_reglas": len(REGLAS_RIPPER),
        "fuente_reglas": "RIPPER",
        "sistema": "Mamdani con reglas RIPPER",
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


def construir_membresias_base():
    return {
        variable: {
            categoria: np.asarray(puntos, dtype=float)
            for categoria, puntos in especificacion["categorias"].items()
        }
        for variable, especificacion in ESPECIFICACIONES_VARIABLES.items()
    }
