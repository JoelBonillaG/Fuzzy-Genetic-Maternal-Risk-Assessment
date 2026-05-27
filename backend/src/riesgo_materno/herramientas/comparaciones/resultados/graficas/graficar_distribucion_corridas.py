from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
SALIDA_DIR = Path(__file__).resolve().parent

ARCHIVOS = {
    "Michigan GA": "genetic_algorithm.json",
    "RIPPER": "ripper.json",
    "PRISM": "prism.json",
}

METRICAS = {
    "accuracy": {
        "eje_y": "Accuracy",
        "archivo": "accuracy_distribucion",
    },
    "balanced_accuracy": {
        "eje_y": "Balanced accuracy",
        "archivo": "balanced_accuracy_distribucion",
    },
}


def leer_metricas() -> dict[str, dict[str, list[float]]]:
    datos = {metrica: {metodo: [] for metodo in ARCHIVOS} for metrica in METRICAS}

    for carpeta in sorted(BASE_DIR.glob("iteracion_*")):
        if not carpeta.is_dir():
            continue
        for metodo, nombre_archivo in ARCHIVOS.items():
            ruta = carpeta / nombre_archivo
            if not ruta.exists():
                raise FileNotFoundError(f"No existe el archivo esperado: {ruta}")
            contenido = json.loads(ruta.read_text(encoding="utf-8-sig"))
            metricas = contenido["metricas"]
            for metrica in METRICAS:
                datos[metrica][metodo].append(float(metricas[metrica]))

    return datos


def configurar_estilo() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": "0.25",
            "axes.linewidth": 0.8,
            "grid.color": "0.85",
            "grid.linewidth": 0.5,
            "savefig.dpi": 300,
        }
    )


def graficar_distribucion(nombre_metrica: str, valores_por_metodo: dict[str, list[float]]) -> None:
    info = METRICAS[nombre_metrica]
    metodos = list(ARCHIVOS.keys())
    posiciones = np.arange(1, len(metodos) + 1)

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    rng = np.random.default_rng(42)

    colores_puntos = ["0.45", "0.55", "0.65"]
    color_media = "0.10"
    color_dispersion = "0.20"

    for posicion, metodo, color_puntos in zip(posiciones, metodos, colores_puntos):
        valores = np.asarray(valores_por_metodo[metodo], dtype=float)
        jitter = rng.uniform(-0.07, 0.07, size=len(valores))

        ax.scatter(
            np.full_like(valores, posicion, dtype=float) + jitter,
            valores,
            s=16,
            color=color_puntos,
            alpha=0.82,
            linewidths=0,
            zorder=2,
        )

        media = float(np.mean(valores))
        desviacion = float(np.std(valores, ddof=1))
        ax.errorbar(
            posicion,
            media,
            yerr=desviacion,
            fmt="s",
            markersize=4.2,
            color=color_media,
            ecolor=color_dispersion,
            elinewidth=1.15,
            capsize=4,
            capthick=1.15,
            zorder=3,
        )

    ax.set_ylabel(info["eje_y"])
    ax.set_xticks(posiciones)
    ax.set_xticklabels(metodos)
    ax.set_xlim(0.5, len(metodos) + 0.5)
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    todos = np.concatenate([np.asarray(valores_por_metodo[m], dtype=float) for m in metodos])
    margen = max(0.015, float((todos.max() - todos.min()) * 0.12))
    ax.set_ylim(max(0.0, float(todos.min()) - margen), min(1.0, float(todos.max()) + margen))

    fig.tight_layout(pad=0.6)

    for extension in ("png", "pdf"):
        salida = SALIDA_DIR / f"{info['archivo']}.{extension}"
        fig.savefig(salida, bbox_inches="tight")

    plt.close(fig)


def main() -> None:
    configurar_estilo()
    datos = leer_metricas()
    for metrica, valores in datos.items():
        graficar_distribucion(metrica, valores)


if __name__ == "__main__":
    main()
