"""Selecciona las mejores bases originales antes de limpiar duplicadas."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


RUTA_REGLAS = Path(__file__).resolve().parent
RUTA_RESULTADOS = (
    RUTA_REGLAS.parents[1]
    / "herramientas"
    / "comparaciones"
    / "resultados"
)

ARCHIVOS_SALIDA = {
    "AG_MICHIGAN_BINARIO": "mejor_ag_antes_limpiar.json",
    "PRISM_ESTOCASTICO": "mejor_prism_antes_limpiar.json",
    "RIPPER": "mejor_ripper_antes_limpiar.json",
}


def seleccionar_mejores():
    mejores = {}
    for ruta in sorted(RUTA_RESULTADOS.glob("iteracion_*/*.json")):
        contenido = json.loads(ruta.read_text(encoding="utf-8-sig"))
        algoritmo = contenido.get("algoritmo")
        balanced_accuracy = contenido.get("metricas", {}).get("balanced_accuracy")
        if algoritmo not in ARCHIVOS_SALIDA or balanced_accuracy is None:
            continue

        actual = mejores.get(algoritmo)
        if actual is None or balanced_accuracy > actual["balanced_accuracy"]:
            mejores[algoritmo] = {
                "ruta": ruta,
                "balanced_accuracy": float(balanced_accuracy),
                "iteracion": int(contenido["iteracion"]),
            }

    faltantes = set(ARCHIVOS_SALIDA) - set(mejores)
    if faltantes:
        raise RuntimeError(f"No se encontraron resultados para: {', '.join(sorted(faltantes))}")

    for algoritmo, salida in ARCHIVOS_SALIDA.items():
        destino = RUTA_REGLAS / salida
        shutil.copy2(mejores[algoritmo]["ruta"], destino)
        print(
            f"{salida} <- {mejores[algoritmo]['ruta'].name} "
            f"(iteracion {mejores[algoritmo]['iteracion']:02d}, "
            f"BA={mejores[algoritmo]['balanced_accuracy']:.4f})"
        )


if __name__ == "__main__":
    seleccionar_mejores()
