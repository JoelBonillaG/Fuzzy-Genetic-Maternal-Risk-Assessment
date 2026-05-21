"""Evalua matrices de confusion cambiando la salida difusa Mamdani.

No modifica archivos del sistema. Recibe por CLI los trapecios de bajo, medio
y alto, ejecuta las reglas ya guardadas y genera matrices de confusion.

Uso desde backend:
    python -m src.riesgo_materno.herramientas.pipeline_reglas.evaluar_salida_difusa ^
      --bajo 0,0,34.5,39.92 ^
      --medio 34.5,39.92,70.31,75.76 ^
      --alto 70.31,75.76,100,100 ^
      --corte-bajo-medio 36.95 ^
      --corte-medio-alto 73.12
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from ...entrenamiento.datos import cargar_dataset, convertir_split_a_diccionario
from ...entrenamiento.modelo import RUTA_CSV
from ...logica_difusa import motor as motor_mod
from ...logica_difusa import variables as variables_mod
from ...logica_difusa.motor import SistemaDifusoMamdani
from ...logica_difusa.variables import ESPECIFICACIONES_VARIABLES


CLASES = ["low risk", "mid risk", "high risk"]
SIN_ACTIVACION = "__sin_activacion__"
CLASE_A_CONSECUENTE = {"low risk": "bajo", "mid risk": "medio", "high risk": "alto"}
RUTA_PAQUETE = Path(__file__).resolve().parents[2]

RUTAS_REGLAS = [
    RUTA_PAQUETE / "reglas" / "candidatas" / "mejor_ag_limpio.json",
]
RUTA_SALIDA = RUTA_PAQUETE / "herramientas" / "pipeline_reglas" / "resultados" / "pruebas_salida_difusa"


def principal():
    args = crear_parser().parse_args()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    salida_difusa = construir_salida_difusa(args)
    aplicar_salida_difusa_temporal(salida_difusa)

    tabla = cargar_dataset(RUTA_CSV)
    resumen = []

    print("Evaluacion con salida difusa temporal")
    print(f"Bajo:  {salida_difusa['categorias']['bajo']}")
    print(f"Medio: {salida_difusa['categorias']['medio']}")
    print(f"Alto:  {salida_difusa['categorias']['alto']}")
    print(f"Cortes: low/mid={args.corte_bajo_medio} | mid/high={args.corte_medio_alto}")

    for ruta in RUTAS_REGLAS:
        resultado = evaluar_archivo(ruta, tabla, args)
        nombre = ruta.stem
        carpeta = salida / nombre
        carpeta.mkdir(parents=True, exist_ok=True)
        guardar_json(carpeta / "resultado.json", resultado)
        guardar_matriz(carpeta / "matriz_confusion.png", resultado["metricas"])
        resumen.append(fila_resumen(nombre, ruta, resultado))
        imprimir(nombre, resultado)

    guardar_json(salida / "config_salida_difusa.json", {
        "salida_difusa": salida_difusa,
        "corte_bajo_medio": args.corte_bajo_medio,
        "corte_medio_alto": args.corte_medio_alto,
    })
    guardar_csv(salida / "resumen.csv", resumen)
    guardar_json(salida / "resumen.json", resumen)
    print(f"\nResultados guardados en: {salida.resolve()}")


def crear_parser():
    parser = argparse.ArgumentParser(
        description="Genera matrices cambiando los trapecios de salida difusa."
    )
    parser.add_argument("--bajo", default="0,0,36.95,42.97")
    parser.add_argument("--medio", default="36.95,42.97,70.31,75.76")
    parser.add_argument("--alto", default="70.31,75.76,100,100")
    parser.add_argument("--corte-bajo-medio", type=float, default=39.92)
    parser.add_argument("--corte-medio-alto", type=float, default=73.12)
    parser.add_argument("--salida", default=str(RUTA_SALIDA))
    return parser


def construir_salida_difusa(args):
    return {
        "nombre": "puntaje_riesgo",
        "universo": (0.0, 100.0),
        "categorias": OrderedDict([
            ("bajo", parsear_trapecio(args.bajo)),
            ("medio", parsear_trapecio(args.medio)),
            ("alto", parsear_trapecio(args.alto)),
        ]),
    }


def parsear_trapecio(texto):
    puntos = [float(x.strip()) for x in texto.split(",")]
    if len(puntos) != 4:
        raise ValueError(f"Un trapecio necesita 4 puntos: {texto}")
    if puntos != sorted(puntos):
        raise ValueError(f"Los puntos del trapecio deben estar ordenados: {texto}")
    return puntos


def aplicar_salida_difusa_temporal(salida_difusa):
    variables_mod.SALIDA_DIFUSA.clear()
    variables_mod.SALIDA_DIFUSA.update(salida_difusa)
    motor_mod.SALIDA_DIFUSA = variables_mod.SALIDA_DIFUSA


def evaluar_archivo(ruta, tabla, args):
    reglas = cargar_reglas(ruta)
    datos = convertir_split_a_diccionario(tabla)
    sistema = SistemaDifusoMamdani(
        construir_membresias_base(),
        reglas=reglas,
        permitir_neutro=False,
    )
    inferencia = sistema.inferir_lote(datos["entradas"])

    predichos = clasificar_puntajes(
        inferencia["puntajes"],
        inferencia["sin_activacion"],
        args.corte_bajo_medio,
        args.corte_medio_alto,
    )
    metricas = construir_metricas(datos["riesgos"], predichos)

    return {
        "archivo_reglas": str(ruta),
        "total_reglas": len(reglas),
        "metricas": metricas,
    }


def cargar_reglas(ruta):
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    reglas = []
    for i, regla in enumerate(contenido["reglas_finales"], start=1):
        antecedentes = [
            (a["variable"], a.get("etiqueta_linguistica", a.get("categoria")))
            for a in regla["antecedentes"]
        ]
        consecuente = regla["consecuente"]
        reglas.append({
            "numero": i,
            "antecedentes": antecedentes,
            "consecuente": CLASE_A_CONSECUENTE.get(consecuente, consecuente),
        })
    return reglas


def clasificar_puntajes(puntajes, sin_activacion, corte_bajo_medio, corte_medio_alto):
    predichos = []
    for puntaje, sin_act in zip(puntajes, sin_activacion):
        if sin_act or np.isnan(puntaje):
            predichos.append(SIN_ACTIVACION)
        elif puntaje < corte_bajo_medio:
            predichos.append("low risk")
        elif puntaje < corte_medio_alto:
            predichos.append("mid risk")
        else:
            predichos.append("high risk")
    return np.asarray(predichos, dtype=object)


def construir_metricas(reales, predichos):
    reales = np.asarray(reales, dtype=object)
    predichos = np.asarray(predichos, dtype=object)
    etiquetas = CLASES + [SIN_ACTIVACION]
    matriz = confusion_matrix(reales, predichos, labels=etiquetas)
    accuracy = float(accuracy_score(reales, predichos))
    ba = balanced_accuracy(reales, predichos)
    aciertos = int(np.sum(reales == predichos))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": ba,
        "aciertos": aciertos,
        "errores": int(len(reales) - aciertos),
        "total": int(len(reales)),
        "sin_activacion": int(np.sum(predichos == SIN_ACTIVACION)),
        "matriz_confusion": {
            "etiquetas": etiquetas,
            "matriz": matriz.tolist(),
        },
    }


def balanced_accuracy(reales, predichos):
    recalls = []
    for clase in CLASES:
        mascara = reales == clase
        if np.any(mascara):
            recalls.append(float(np.mean(predichos[mascara] == clase)))
    return float(np.mean(recalls))


def construir_membresias_base():
    return {
        variable: {
            categoria: np.asarray(puntos, dtype=float)
            for categoria, puntos in especificacion["categorias"].items()
        }
        for variable, especificacion in ESPECIFICACIONES_VARIABLES.items()
    }


def guardar_matriz(ruta, metricas):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

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
    ax.set_title(
        f"Accuracy={metricas['accuracy']:.4f} | "
        f"BA={metricas['balanced_accuracy']:.4f} | "
        f"Sin activacion={metricas['sin_activacion']}"
    )
    umbral = matriz.max() / 2 if matriz.size else 0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            color = "white" if matriz[i, j] > umbral else "black"
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center", color=color, fontweight="bold")
    fig.colorbar(imagen, ax=ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def fila_resumen(nombre, ruta, resultado):
    metricas = resultado["metricas"]
    return {
        "conjunto": nombre,
        "archivo": str(ruta),
        "reglas": resultado["total_reglas"],
        "accuracy": metricas["accuracy"],
        "balanced_accuracy": metricas["balanced_accuracy"],
        "aciertos": metricas["aciertos"],
        "errores": metricas["errores"],
        "sin_activacion": metricas["sin_activacion"],
    }


def imprimir(nombre, resultado):
    metricas = resultado["metricas"]
    print()
    print(nombre)
    print(f"  Reglas: {resultado['total_reglas']}")
    print(f"  Accuracy={metricas['accuracy']:.4f} | BA={metricas['balanced_accuracy']:.4f}")


def guardar_json(ruta, contenido):
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")


def guardar_csv(ruta, filas):
    filas = list(filas)
    if not filas:
        ruta.write_text("", encoding="utf-8")
        return
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)


if __name__ == "__main__":
    principal()
