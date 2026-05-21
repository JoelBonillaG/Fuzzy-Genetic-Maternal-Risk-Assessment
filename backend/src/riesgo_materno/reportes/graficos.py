"""Graficos reutilizables para matrices e histogramas."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def guardar_matriz_confusion(ruta: Path, metricas: dict, titulo: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ruta.parent.mkdir(parents=True, exist_ok=True)
    etiquetas = metricas["matriz_confusion"]["etiquetas"]
    matriz = np.asarray(metricas["matriz_confusion"]["matriz"], dtype=int)

    fig, ax = plt.subplots(figsize=(9, 7))
    imagen = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(len(etiquetas)))
    ax.set_yticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=35, ha="right")
    ax.set_yticklabels(etiquetas)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Clase real")
    ax.set_title(f"{titulo}\nAccuracy={metricas['accuracy']:.4f} | BA={metricas['balanced_accuracy']:.4f}")

    umbral = matriz.max() / 2 if matriz.size else 0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            color = "white" if matriz[i, j] > umbral else "black"
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center", color=color, fontweight="bold")

    fig.colorbar(imagen, ax=ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def guardar_histograma_puntajes(
    ruta: Path,
    casos: list[dict],
    campo: str,
    titulo: str,
    descripcion: str,
    grupos: list[dict],
    intervalo_confianza: tuple[float, float],
    corte: float,
    limite_y: int | None = None,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ruta.parent.mkdir(parents=True, exist_ok=True)
    valores = np.asarray([caso[campo] for caso in casos], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    limite_inferior, limite_superior = intervalo_confianza
    bins = np.linspace(limite_inferior, limite_superior, 17)
    if len(valores):
        _dibujar_series(ax, casos, campo, grupos, bins)
        ax.set_title(f"{titulo}\n{descripcion} | Casos={len(valores)}")
    else:
        ax.set_title(f"{titulo}\nCasos=0")

    ax.axvspan(
        limite_inferior,
        limite_superior,
        color="#A7C7E7",
        alpha=0.22,
        label=f"Intervalo de confianza [{limite_inferior:.2f}, {limite_superior:.2f}]",
    )
    ax.axvline(limite_inferior, color="#1F77B4", linestyle="--", linewidth=2, label="Limites IC")
    ax.axvline(limite_superior, color="#1F77B4", linestyle="--", linewidth=2)
    ax.axvline(corte, color="#D62728", linewidth=2, label=f"Corte central={corte:.2f}")
    ax.set_xlim(limite_inferior - 0.3, limite_superior + 0.3)
    if limite_y:
        ax.set_ylim(0, limite_y)
    ax.set_xlabel("Puntaje de riesgo desfusificado")
    ax.set_ylabel("Cantidad de casos")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def _dibujar_series(ax, casos: list[dict], campo: str, grupos: list[dict], bins: np.ndarray):
    series = []
    colores = []
    etiquetas = []
    for grupo in grupos:
        valores = [caso[campo] for caso in casos if caso.get(grupo["campo"]) == grupo["valor"]]
        if valores:
            series.append(valores)
            colores.append(grupo["color"])
            etiquetas.append(grupo["etiqueta"])

    if len(series) == 1:
        ax.hist(series[0], bins=bins, color=colores[0], edgecolor="white", label=etiquetas[0])
        return

    ax.hist(series, bins=bins, color=colores, edgecolor="white", label=etiquetas, stacked=False, alpha=0.78)
