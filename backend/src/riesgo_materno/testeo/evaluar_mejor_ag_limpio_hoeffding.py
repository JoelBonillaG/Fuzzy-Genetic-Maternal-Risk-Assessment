"""Evalua la base limpia del AG con tamano de muestra por Hoeffding."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

RUTA_BACKEND = Path(__file__).resolve().parents[3]
if str(RUTA_BACKEND) not in sys.path:
    sys.path.insert(0, str(RUTA_BACKEND))

from src.riesgo_materno.entrenamiento.datos import (  # noqa: E402
    cargar_dataset,
    convertir_split_a_diccionario,
)
from src.riesgo_materno.entrenamiento.modelo import RUTA_CSV  # noqa: E402
from src.riesgo_materno.logica_difusa.motor import SistemaDifusoMamdani  # noqa: E402
from src.riesgo_materno.logica_difusa.variables import (  # noqa: E402
    ESPECIFICACIONES_VARIABLES,
)
from src.riesgo_materno.reportes.graficos import guardar_matriz_confusion  # noqa: E402
from src.riesgo_materno.reportes.metricas import calcular_metricas  # noqa: E402


CLASES = ["low risk", "mid risk", "high risk"]
SIN_ACTIVACION = "__sin_activacion__"
CLASE_A_CONSECUENTE = {"low risk": "bajo", "mid risk": "medio", "high risk": "alto"}
EPSILON = 0.05
CONFIANZA = 0.95
DELTA = 1.0 - CONFIANZA
TAMANO_MUESTRA = math.ceil(math.log(2.0 / DELTA) / (2.0 * EPSILON**2))
SEMILLA = 42

RUTA_PAQUETE = Path(__file__).resolve().parents[1]
RUTA_REGLAS = (
    RUTA_PAQUETE
    / "reglas"
    / "candidatas_despues_limpiar"
    / "mejor_ag_limpio.json"
)
RUTA_SALIDA = Path(__file__).resolve().parent / "resultados" / "resultados_ag_limpio"


def main():
    tabla = cargar_dataset(RUTA_CSV)
    muestra = muestrear_estratificado(tabla, TAMANO_MUESTRA, SEMILLA)
    datos = convertir_split_a_diccionario(muestra)
    reglas = cargar_reglas(RUTA_REGLAS)
    sistema = SistemaDifusoMamdani(membresias_base(), reglas=reglas)
    inferencia = sistema.inferir_lote(datos["entradas"])

    metricas = calcular_metricas(
        datos["riesgos"],
        inferencia["riesgos"],
        clases=CLASES,
        sin_activacion=inferencia["sin_activacion"],
        etiqueta_sin=SIN_ACTIVACION,
    )
    resultado = {
        "base_reglas": str(RUTA_REGLAS),
        "criterio_muestra": {
            "desigualdad": "Hoeffding",
            "epsilon": EPSILON,
            "confianza": CONFIANZA,
            "delta": DELTA,
            "n_calculado": TAMANO_MUESTRA,
            "semilla": SEMILLA,
            "muestreo": "estratificado por clase de riesgo",
        },
        "datos": {
            "total_dataset_procesado": int(len(tabla)),
            "total_muestra": int(len(muestra)),
            "distribucion_muestra": conteos_por_clase(muestra),
        },
        "reglas": {
            "total": int(len(reglas)),
        },
        "metricas": sin_matriz(metricas),
        "matriz_confusion": metricas["matriz_confusion"],
    }

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    guardar_json(RUTA_SALIDA / "resultado.json", resultado)
    guardar_predicciones(
        RUTA_SALIDA / "predicciones.csv",
        muestra,
        inferencia["riesgos"],
        inferencia["puntajes"],
        inferencia["sin_activacion"],
    )
    guardar_matriz_confusion(RUTA_SALIDA / "matriz_confusion.png", metricas, "AG limpio")

    print(f"n={TAMANO_MUESTRA} | epsilon={EPSILON} | confianza={CONFIANZA:.2f}")
    print(
        f"Accuracy={metricas['accuracy']:.4f} | "
        f"BA={metricas['balanced_accuracy']:.4f} | "
        f"Sin activacion={metricas['sin_activacion']}"
    )
    print(f"Resultados: {RUTA_SALIDA.resolve()}")


def muestrear_estratificado(tabla, n, semilla):
    if n > len(tabla):
        raise ValueError(f"n={n} supera el total disponible: {len(tabla)}")

    fracciones = tabla["riesgo"].value_counts(normalize=True)
    cuotas = (fracciones * n).round().astype(int)
    diferencia = n - int(cuotas.sum())
    if diferencia:
        clase_mayor = fracciones.idxmax()
        cuotas.loc[clase_mayor] += diferencia

    partes = []
    for clase in CLASES:
        grupo = tabla[tabla["riesgo"] == clase]
        partes.append(grupo.sample(n=int(cuotas.loc[clase]), random_state=semilla))
    indices = np.concatenate([parte.index.to_numpy() for parte in partes])
    indices = np.random.default_rng(semilla).permutation(indices)
    return tabla.loc[indices].reset_index(drop=True)


def cargar_reglas(ruta):
    contenido = json.loads(ruta.read_text(encoding="utf-8-sig"))
    reglas_crudas = contenido.get("reglas_finales", contenido)
    reglas = []
    for numero, regla in enumerate(reglas_crudas, start=1):
        reglas.append(
            {
                "numero": numero,
                "antecedentes": [
                    [
                        antecedente["variable"],
                        antecedente.get("etiqueta_linguistica", antecedente.get("categoria")),
                    ]
                    for antecedente in regla["antecedentes"]
                ],
                "consecuente": CLASE_A_CONSECUENTE.get(regla["consecuente"], regla["consecuente"]),
            }
        )
    return reglas


def membresias_base():
    return {
        variable: {
            categoria: np.asarray(puntos, dtype=float)
            for categoria, puntos in especificacion["categorias"].items()
        }
        for variable, especificacion in ESPECIFICACIONES_VARIABLES.items()
    }


def conteos_por_clase(tabla):
    conteos = tabla["riesgo"].value_counts().reindex(CLASES, fill_value=0)
    return {clase: int(conteos[clase]) for clase in CLASES}


def sin_matriz(metricas):
    salida = dict(metricas)
    salida.pop("matriz_confusion", None)
    return salida


def guardar_json(ruta, contenido):
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")


def guardar_predicciones(ruta, muestra, predichos, puntajes, sin_activacion):
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=[
                "indice_muestra",
                "riesgo_real",
                "riesgo_predicho",
                "puntaje",
                "sin_activacion",
            ],
        )
        escritor.writeheader()
        for indice, (_, fila) in enumerate(muestra.iterrows(), start=1):
            escritor.writerow(
                {
                    "indice_muestra": indice,
                    "riesgo_real": fila["riesgo"],
                    "riesgo_predicho": predichos[indice - 1],
                    "puntaje": "" if np.isnan(puntajes[indice - 1]) else round(float(puntajes[indice - 1]), 4),
                    "sin_activacion": bool(sin_activacion[indice - 1]),
                }
            )


if __name__ == "__main__":
    main()
