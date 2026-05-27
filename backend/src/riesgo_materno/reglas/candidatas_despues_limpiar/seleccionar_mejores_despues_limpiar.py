"""Selecciona las mejores bases limpias por algoritmo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUTA_BACKEND = Path(__file__).resolve().parents[4]
if str(RUTA_BACKEND) not in sys.path:
    sys.path.insert(0, str(RUTA_BACKEND))

from src.riesgo_materno.entrenamiento.datos import (
    cargar_dataset,
    convertir_split_a_diccionario,
)
from src.riesgo_materno.entrenamiento.modelo import RUTA_CSV
from src.riesgo_materno.herramientas.pipeline_reglas.limpiar_reglas_iteraciones import (
    construir_membresias_base,
    evaluar_reglas,
    normalizar_reglas,
    quitar_duplicadas,
    reglas_a_motor,
    renumerar_reglas,
    resumir_reglas,
    sin_matriz,
)


RUTA_SALIDA = Path(__file__).resolve().parent
RUTA_RESULTADOS = (
    RUTA_SALIDA.parents[1]
    / "herramientas"
    / "comparaciones"
    / "resultados"
)

ARCHIVOS_SALIDA = {
    "AG_MICHIGAN_BINARIO": "mejor_ag_limpio.json",
    "PRISM_ESTOCASTICO": "mejor_prism_limpio.json",
    "RIPPER": "mejor_ripper_limpio.json",
}


def seleccionar_mejores_limpios():
    tabla = cargar_dataset(RUTA_CSV)
    datos = convertir_split_a_diccionario(tabla)
    membresias = construir_membresias_base()
    mejores = {}

    for ruta in sorted(RUTA_RESULTADOS.glob("iteracion_*/*.json")):
        contenido = leer_json(ruta)
        algoritmo = contenido.get("algoritmo")
        if algoritmo not in ARCHIVOS_SALIDA:
            continue

        resultado = limpiar_candidato(ruta, contenido, datos, membresias)
        actual = mejores.get(algoritmo)
        if actual is None or clave_orden(resultado) > clave_orden(actual):
            mejores[algoritmo] = resultado

    faltantes = set(ARCHIVOS_SALIDA) - set(mejores)
    if faltantes:
        raise RuntimeError(f"No se encontraron resultados para: {', '.join(sorted(faltantes))}")

    for algoritmo, archivo in ARCHIVOS_SALIDA.items():
        ruta_salida = RUTA_SALIDA / archivo
        guardar_json(ruta_salida, mejores[algoritmo])
        metricas = mejores[algoritmo]["metricas_limpias"]
        print(
            f"{archivo} <- iteracion {mejores[algoritmo]['iteracion']:02d} "
            f"(BA={metricas['balanced_accuracy']:.4f}, "
            f"Accuracy={metricas['accuracy']:.4f})"
        )


def limpiar_candidato(ruta, contenido, datos, membresias):
    reglas_originales = normalizar_reglas(contenido.get("reglas_finales", []))
    reglas_limpias, duplicadas = quitar_duplicadas(reglas_originales)
    reglas_motor = reglas_a_motor(reglas_limpias)
    metricas = evaluar_reglas(reglas_motor, datos, membresias)
    return {
        "id_experimento": contenido.get("id_experimento"),
        "iteracion": int(contenido["iteracion"]),
        "algoritmo": contenido["algoritmo"],
        "archivo_origen": str(ruta),
        "datos": contenido.get("datos", {}),
        "hiperparametros": contenido.get("hiperparametros", {}),
        "metricas_originales": contenido.get("metricas", {}),
        "matriz_confusion_original": contenido.get("matriz_confusion", {}),
        "resumen_reglas_original": contenido.get("resumen_reglas", {}),
        "resumen_reglas_limpias": resumir_reglas(reglas_limpias, duplicadas),
        "metricas_limpias": sin_matriz(metricas),
        "matriz_confusion_limpia": metricas["matriz_confusion"],
        "reglas_finales": renumerar_reglas(reglas_limpias),
    }


def clave_orden(resultado):
    metricas = resultado["metricas_limpias"]
    return metricas["balanced_accuracy"], metricas["accuracy"]


def leer_json(ruta):
    return json.loads(ruta.read_text(encoding="utf-8-sig"))


def guardar_json(ruta, contenido):
    ruta.write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    seleccionar_mejores_limpios()
