from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ....entrenamiento.datos import cargar_dataset
from ....entrenamiento.modelo import RUTA_CSV
from ....logica_difusa.variables import ETIQUETAS_RIESGO, VARIABLES_ENTRADA


RUTA_RESULTADOS = Path(__file__).resolve().parent / "resultados"
OPCIONES_GRAFICA = {
    "todo": "histogramas_variables_riesgo.png",
    "barras_lado": "histogramas_variables_riesgo_barras_lado.png",
}

NOMBRES_VARIABLES = {
    "edad": "Edad",
    "presion_sistolica": "Presion sistolica",
    "presion_diastolica": "Presion diastolica",
    "azucar_sangre": "Azucar en sangre",
    "temperatura_corporal": "Temperatura corporal",
    "frecuencia_cardiaca": "Frecuencia cardiaca",
}

COLORES_RIESGO = {
    "low risk": "#2ca25f",
    "mid risk": "#fdae61",
    "high risk": "#d7191c",
}


def generar_histogramas(tipo="todo", ruta_salida=None):
    """Genera histogramas por variable coloreados por nivel de riesgo."""
    if tipo not in OPCIONES_GRAFICA:
        opciones = ", ".join(OPCIONES_GRAFICA)
        raise ValueError(f"Tipo de grafica no reconocido: {tipo}. Opciones: {opciones}")

    datos = cargar_dataset(RUTA_CSV)

    figura, ejes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ejes = ejes.ravel()

    for eje, variable in zip(ejes, VARIABLES_ENTRADA):
        valores = datos[variable]
        bins = np.histogram_bin_edges(valores, bins="auto")

        if tipo == "todo":
            _dibujar_histograma_superpuesto(eje, datos, variable, bins)
        elif tipo == "barras_lado":
            _dibujar_histograma_barras_lado(eje, datos, variable, bins)

        eje.set_title(NOMBRES_VARIABLES[variable])
        eje.set_xlabel("Valor de la variable")
        eje.set_ylabel("Cantidad de casos")
        eje.grid(axis="y", alpha=0.25)

    figura.suptitle(
        "Distribucion de variables clinicas por nivel de riesgo materno",
        fontsize=16,
        fontweight="bold",
    )
    manejadores, etiquetas = ejes[0].get_legend_handles_labels()
    figura.legend(
        manejadores,
        etiquetas,
        title="Nivel de riesgo",
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    if ruta_salida is None:
        ruta_salida = RUTA_RESULTADOS / OPCIONES_GRAFICA[tipo]
    else:
        ruta_salida = Path(ruta_salida)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(figura)
    return ruta_salida


def _dibujar_histograma_superpuesto(eje, datos, variable, bins):
    for riesgo in ETIQUETAS_RIESGO:
        valores_riesgo = datos.loc[datos["riesgo"] == riesgo, variable]
        eje.hist(
            valores_riesgo,
            bins=bins,
            alpha=0.62,
            label=riesgo,
            color=COLORES_RIESGO[riesgo],
            edgecolor="white",
            linewidth=0.6,
        )


def _dibujar_histograma_barras_lado(eje, datos, variable, bins):
    centros = (bins[:-1] + bins[1:]) / 2
    ancho_bin = np.diff(bins)
    ancho_barra = ancho_bin / (len(ETIQUETAS_RIESGO) + 0.6)
    desplazamiento_base = -(len(ETIQUETAS_RIESGO) - 1) / 2

    for indice, riesgo in enumerate(ETIQUETAS_RIESGO):
        valores_riesgo = datos.loc[datos["riesgo"] == riesgo, variable]
        conteos, _ = np.histogram(valores_riesgo, bins=bins)
        desplazamiento = desplazamiento_base + indice
        eje.bar(
            centros + desplazamiento * ancho_barra,
            conteos,
            width=ancho_barra,
            align="center",
            label=riesgo,
            color=COLORES_RIESGO[riesgo],
            edgecolor="white",
            linewidth=0.6,
        )


def _leer_argumentos():
    parser = argparse.ArgumentParser(
        description="Genera histogramas de variables clinicas por nivel de riesgo."
    )
    parser.add_argument(
        "--tipo",
        choices=sorted(OPCIONES_GRAFICA),
        default="todo",
        help="Tipo de grafica a generar.",
    )
    parser.add_argument(
        "--salida",
        default=None,
        help="Ruta opcional del archivo PNG de salida.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    argumentos = _leer_argumentos()
    salida = generar_histogramas(tipo=argumentos.tipo, ruta_salida=argumentos.salida)
    print(f"Histogramas guardados en: {salida}")
