import numpy as np
import pandas as pd

from ..logica_difusa.motor import SistemaDifusoMamdani
from .metricas import calcular_recall_de_riesgo_alto, calcular_macro_f1


def evaluar_membresias_en_splits(membresias, datos_por_split):
    """Construye el sistema difuso con las membresias dadas y mide su desempeno en cada split."""
    sistema = SistemaDifusoMamdani(membresias)
    resultados = {}
    for nombre_split, datos_split in datos_por_split.items():
        # POR AHORA EN VALIDACION...
        if nombre_split not in ["validacion"]:
            continue
        resultados[nombre_split] = evaluar_sistema_en_split(sistema, datos_split)
    return resultados


def evaluar_sistema_en_split(sistema, datos_split):
    """Infiere en lote sobre un split y devuelve macro_f1, recall_riesgo_alto y reporte por clase."""
    inferencia = sistema.inferir_lote(datos_split["entradas"])
    puntajes = inferencia["puntajes"]
    riesgos = inferencia["riesgos"]

    if np.isnan(puntajes).any() or np.any(riesgos == None):
        raise ValueError("El sistema produjo puntajes invalidos o riesgos vacios.")

    return {
        "macro_f1": calcular_macro_f1(datos_split["riesgos"], riesgos),
        "recall_riesgo_alto": calcular_recall_de_riesgo_alto(datos_split["riesgos"], riesgos),
    }

def crear_tabla_comparativa_prueba(resultado_base_prueba, resultado_optimizado_prueba, fitness_base=0.0, fitness_opt=0.0):
    """Compara Macro F1, Recall alto y Fitness entre sistema base y optimizado (split de prueba)."""
    filas = [
        {
            "metrica": "Macro F1",
            "base": float(resultado_base_prueba["macro_f1"]),
            "optimizado": float(resultado_optimizado_prueba["macro_f1"]),
        },
        {
            "metrica": "Recall alto",
            "base": float(resultado_base_prueba["recall_riesgo_alto"]),
            "optimizado": float(resultado_optimizado_prueba["recall_riesgo_alto"]),
        },
        {
            "metrica": "Fitness",
            "base": float(fitness_base),
            "optimizado": float(fitness_opt),
        },
    ]
    tabla = pd.DataFrame(filas)
    tabla["delta"] = tabla["optimizado"] - tabla["base"]
    return tabla
