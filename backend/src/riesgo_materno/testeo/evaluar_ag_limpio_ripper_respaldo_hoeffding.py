"""Evalua AG limpio con RIPPER como respaldo sobre una muestra Hoeffding."""

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
RUTA_REGLAS_AG = (
    RUTA_PAQUETE
    / "reglas"
    / "candidatas_despues_limpiar"
    / "mejor_ag_limpio.json"
)
RUTA_REGLAS_RIPPER = RUTA_PAQUETE / "reglas" / "reglas_sistema_difuso_ripper.json"
RUTA_SALIDA = Path(__file__).resolve().parent / "resultados" / "resultados_ag_y_ripper"


def main():
    tabla = cargar_dataset(RUTA_CSV)
    muestra = muestrear_estratificado(tabla, TAMANO_MUESTRA, SEMILLA)
    datos = convertir_split_a_diccionario(muestra)

    reglas_ag = cargar_reglas(RUTA_REGLAS_AG)
    reglas_ripper = cargar_reglas(RUTA_REGLAS_RIPPER)

    sistema_ag = SistemaDifusoMamdani(membresias_base(), reglas=reglas_ag)
    sistema_ripper = SistemaDifusoMamdani(membresias_base(), reglas=reglas_ripper)

    inferencia_ag = sistema_ag.inferir_lote(datos["entradas"])
    inferencia_final = aplicar_respaldo_ripper(
        datos["entradas"],
        inferencia_ag,
        sistema_ripper,
    )

    metricas = calcular_metricas(
        datos["riesgos"],
        inferencia_final["riesgos"],
        clases=CLASES,
        sin_activacion=inferencia_final["sin_activacion"],
        etiqueta_sin=SIN_ACTIVACION,
    )

    resultado = {
        "base_principal": str(RUTA_REGLAS_AG),
        "base_respaldo": str(RUTA_REGLAS_RIPPER),
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
            "ag_limpio": int(len(reglas_ag)),
            "ripper_respaldo": int(len(reglas_ripper)),
            "total_disponibles": int(len(reglas_ag) + len(reglas_ripper)),
        },
        "respaldo": {
            "casos_sin_activacion_ag": int(inferencia_ag["sin_activacion"].sum()),
            "casos_resueltos_por_ripper": int(inferencia_final["resueltos_por_ripper"].sum()),
            "casos_sin_activacion_final": int(inferencia_final["sin_activacion"].sum()),
        },
        "metricas": sin_matriz(metricas),
        "matriz_confusion": metricas["matriz_confusion"],
    }

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    guardar_json(RUTA_SALIDA / "resultado.json", resultado)
    guardar_predicciones(
        RUTA_SALIDA / "predicciones.csv",
        muestra,
        inferencia_final["riesgos"],
        inferencia_final["puntajes"],
        inferencia_final["sin_activacion"],
        inferencia_final["fuente"],
    )
    guardar_matriz_confusion(RUTA_SALIDA / "matriz_confusion.png", metricas, "AG limpio + RIPPER")

    print(f"n={TAMANO_MUESTRA} | epsilon={EPSILON} | confianza={CONFIANZA:.2f}")
    print(
        f"Accuracy={metricas['accuracy']:.4f} | "
        f"BA={metricas['balanced_accuracy']:.4f} | "
        f"Sin activacion={metricas['sin_activacion']}"
    )
    print(
        "Respaldo RIPPER="
        f"{resultado['respaldo']['casos_resueltos_por_ripper']} / "
        f"{resultado['respaldo']['casos_sin_activacion_ag']} casos sin activacion AG"
    )
    print(f"Resultados: {RUTA_SALIDA.resolve()}")


def aplicar_respaldo_ripper(entradas, inferencia_ag, sistema_ripper):
    n = len(inferencia_ag["riesgos"])
    riesgos = np.array(inferencia_ag["riesgos"], dtype=object)
    puntajes = np.array(inferencia_ag["puntajes"], dtype=float)
    sin_activacion = np.array(inferencia_ag["sin_activacion"], dtype=bool)
    fuente = np.where(sin_activacion, "sin_activacion_ag", "ag_limpio").astype(object)
    resueltos_por_ripper = np.zeros(n, dtype=bool)

    indices_respaldo = np.flatnonzero(sin_activacion)
    if len(indices_respaldo) == 0:
        return {
            "riesgos": riesgos,
            "puntajes": puntajes,
            "sin_activacion": sin_activacion,
            "fuente": fuente,
            "resueltos_por_ripper": resueltos_por_ripper,
        }

    entradas_respaldo = {
        variable: np.asarray(valores)[indices_respaldo]
        for variable, valores in entradas.items()
    }
    inferencia_ripper = sistema_ripper.inferir_lote(entradas_respaldo)

    for pos_local, indice_global in enumerate(indices_respaldo):
        if not bool(inferencia_ripper["sin_activacion"][pos_local]):
            riesgos[indice_global] = inferencia_ripper["riesgos"][pos_local]
            puntajes[indice_global] = inferencia_ripper["puntajes"][pos_local]
            sin_activacion[indice_global] = False
            fuente[indice_global] = "ripper_respaldo"
            resueltos_por_ripper[indice_global] = True

    return {
        "riesgos": riesgos,
        "puntajes": puntajes,
        "sin_activacion": sin_activacion,
        "fuente": fuente,
        "resueltos_por_ripper": resueltos_por_ripper,
    }


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
    reglas_crudas = contenido.get("reglas_finales", contenido) if isinstance(contenido, dict) else contenido
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
                    if isinstance(antecedente, dict)
                    else antecedente
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


def guardar_predicciones(ruta, muestra, predichos, puntajes, sin_activacion, fuente):
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=[
                "indice_muestra",
                "riesgo_real",
                "riesgo_predicho",
                "puntaje",
                "sin_activacion",
                "fuente",
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
                    "fuente": fuente[indice - 1],
                }
            )


if __name__ == "__main__":
    main()
