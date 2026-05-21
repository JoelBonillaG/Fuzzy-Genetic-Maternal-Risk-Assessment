"""Metricas de clasificacion usadas por los experimentos."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def calcular_metricas(reales, predichos, clases: list[str], sin_activacion=None, etiqueta_sin=None) -> dict:
    reales = np.asarray(reales, dtype=object)
    predichos = np.asarray(predichos, dtype=object).copy()

    etiquetas = list(clases)
    if sin_activacion is not None and etiqueta_sin is not None:
        predichos[np.asarray(sin_activacion, dtype=bool)] = etiqueta_sin
        etiquetas.append(etiqueta_sin)

    matriz = confusion_matrix(reales, predichos, labels=etiquetas)
    aciertos = int(np.sum(reales == predichos))
    return {
        "accuracy": float(accuracy_score(reales, predichos)),
        "balanced_accuracy": balanced_accuracy(reales, predichos, clases),
        "aciertos": aciertos,
        "errores": int(len(reales) - aciertos),
        "total": int(len(reales)),
        "sin_activacion": int(np.sum(predichos == etiqueta_sin)) if etiqueta_sin else 0,
        "matriz_confusion": {"etiquetas": etiquetas, "matriz": matriz.tolist()},
    }


def balanced_accuracy(reales, predichos, clases: list[str]) -> float:
    recalls = []
    for clase in clases:
        mascara = reales == clase
        if np.any(mascara):
            recalls.append(float(np.mean(predichos[mascara] == clase)))
    return float(np.mean(recalls))

