"""Limpia reglas duplicadas por iteracion y recalcula metricas.

Uso desde backend:
    python -m src.riesgo_materno.herramientas.comparaciones.limpiar_duplicadas_recalcular

Lee:
    src/riesgo_materno/herramientas/comparaciones/resultados/vN/iteracion_XX/*.json

Guarda:
    src/riesgo_materno/herramientas/comparaciones/resultados_sin_duplicadas/vN
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from ...entrenamiento.datos import cargar_dataset, convertir_split_a_diccionario
from ...entrenamiento.modelo import RUTA_CSV
from ...logica_difusa.motor import SistemaDifusoMamdani
from ...logica_difusa.variables import ESPECIFICACIONES_VARIABLES


CLASES = ["low risk", "mid risk", "high risk"]
SIN_ACTIVACION = "__sin_activacion__"
CLASE_A_CONSECUENTE = {"low risk": "bajo", "mid risk": "medio", "high risk": "alto"}
CONSECUENTE_A_CLASE = {v: k for k, v in CLASE_A_CONSECUENTE.items()}

RUTA_BASE = Path(__file__).resolve().parent
RUTA_ENTRADA = RUTA_BASE / "resultados"
RUTA_SALIDA = RUTA_BASE / "resultados_sin_duplicadas"


def main():
    args = crear_parser().parse_args()
    entrada = resolver_entrada(Path(args.entrada) if args.entrada else RUTA_ENTRADA)
    salida = Path(args.salida) if args.salida else resolver_salida(entrada)
    salida.mkdir(parents=True, exist_ok=True)

    tabla = cargar_dataset(RUTA_CSV)
    datos = convertir_split_a_diccionario(tabla)
    membresias = construir_membresias_base()

    filas = []
    resultados = []
    for ruta_json in archivos_de_resultados(entrada):
        resultado = procesar_archivo(ruta_json, salida, datos, membresias)
        resultados.append(resultado)
        filas.append(fila_resumen(resultado))
        imprimir_resultado(resultado)

    resumen = construir_resumen(filas)
    ruta_matriz = guardar_matriz_mejor_ba(resultados, salida)
    if ruta_matriz is not None:
        resumen["matriz_confusion_mejor_ba"] = str(ruta_matriz)
    guardar_csv(salida / "resumen_iteraciones.csv", filas)
    guardar_csv(salida / "resumen_estadistico.csv", resumen["resumen_estadistico"])
    guardar_json(salida / "resumen_metricas.json", resumen)
    print(f"\nResultados sin duplicadas guardados en: {salida.resolve()}")


def crear_parser():
    parser = argparse.ArgumentParser(description="Recalcula metricas eliminando reglas duplicadas.")
    parser.add_argument("--entrada", default=None)
    parser.add_argument("--salida", default=None)
    return parser


def resolver_entrada(ruta: Path):
    if ruta.name.startswith("iteracion_"):
        return ruta

    versiones = [
        carpeta
        for carpeta in ruta.glob("v*")
        if carpeta.is_dir() and carpeta.name[1:].isdigit() and contiene_iteraciones(carpeta)
    ]
    if versiones:
        return max(versiones, key=lambda carpeta: int(carpeta.name[1:]))
    if contiene_iteraciones(ruta):
        return ruta
    return ruta


def resolver_salida(entrada: Path):
    if entrada.name.startswith("v") and entrada.name[1:].isdigit():
        return RUTA_SALIDA / entrada.name
    return crear_carpeta_versionada(RUTA_SALIDA)


def contiene_iteraciones(ruta: Path):
    return ruta.is_dir() and any(carpeta.is_dir() for carpeta in ruta.glob("iteracion_*"))


def crear_carpeta_versionada(base: Path):
    base.mkdir(parents=True, exist_ok=True)
    versiones = [
        int(carpeta.name[1:])
        for carpeta in base.glob("v*")
        if carpeta.is_dir() and carpeta.name[1:].isdigit()
    ]
    ruta = base / f"v{max(versiones, default=0) + 1}"
    ruta.mkdir(parents=True, exist_ok=False)
    return ruta


def archivos_de_resultados(entrada: Path):
    carpetas = [entrada] if entrada.name.startswith("iteracion_") else sorted(entrada.glob("iteracion_*"))
    for carpeta_iteracion in carpetas:
        if not carpeta_iteracion.is_dir():
            continue
        for ruta_json in sorted(carpeta_iteracion.glob("*.json")):
            if ruta_json.name.startswith("resumen") or ruta_json.name.startswith("config"):
                continue
            yield ruta_json


def procesar_archivo(ruta_json: Path, salida: Path, datos: dict, membresias: dict):
    contenido = json.loads(ruta_json.read_text(encoding="utf-8-sig"))
    reglas_originales = contenido.get("reglas_finales", [])
    reglas_limpias, duplicadas = quitar_reglas_duplicadas(reglas_originales)
    reglas_motor = reglas_a_motor(reglas_limpias)
    metricas = evaluar_reglas(reglas_motor, datos, membresias)

    iteracion = int(contenido.get("iteracion", extraer_iteracion(ruta_json)))
    algoritmo = contenido.get("algoritmo", ruta_json.stem)
    resultado = {
        "id_experimento": contenido.get("id_experimento"),
        "iteracion": iteracion,
        "algoritmo": algoritmo,
        "archivo_origen": str(ruta_json),
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

    carpeta_salida = salida / f"iteracion_{iteracion:02d}"
    guardar_json(carpeta_salida / ruta_json.name, resultado)
    return resultado


def quitar_reglas_duplicadas(reglas: list[dict]):
    vistas = set()
    limpias = []
    duplicadas = 0
    for regla in reglas:
        clave = clave_regla(regla)
        if clave in vistas:
            duplicadas += 1
            continue
        vistas.add(clave)
        limpias.append(regla)
    return limpias, duplicadas


def clave_regla(regla: dict):
    antecedentes = tuple(sorted(
        (antecedente["variable"], antecedente["etiqueta_linguistica"])
        for antecedente in regla.get("antecedentes", [])
    ))
    return antecedentes, regla.get("consecuente")


def reglas_a_motor(reglas: list[dict]):
    reglas_motor = []
    for numero, regla in enumerate(reglas, start=1):
        consecuente = regla["consecuente"]
        reglas_motor.append({
            "numero": numero,
            "antecedentes": [
                (antecedente["variable"], antecedente["etiqueta_linguistica"])
                for antecedente in regla["antecedentes"]
            ],
            "consecuente": CLASE_A_CONSECUENTE.get(consecuente, consecuente),
        })
    return reglas_motor


def evaluar_reglas(reglas_motor: list[dict], datos: dict, membresias: dict):
    sistema = SistemaDifusoMamdani(membresias, reglas=reglas_motor, permitir_neutro=False)
    inferencia = sistema.inferir_lote(datos["entradas"])
    metricas = calcular_metricas_clasificacion(
        datos["riesgos"],
        inferencia["riesgos"],
        inferencia["sin_activacion"],
    )
    metricas["error_clasificacion"] = float(1.0 - metricas["accuracy"])
    metricas["error_balanceado"] = float(1.0 - metricas["balanced_accuracy"])
    return metricas


def calcular_metricas_clasificacion(reales, predichos, sin_activacion):
    reales = np.asarray(reales, dtype=object)
    predichos = np.asarray(predichos, dtype=object).copy()
    predichos[np.asarray(sin_activacion, dtype=bool)] = SIN_ACTIVACION
    etiquetas = CLASES + [SIN_ACTIVACION]
    matriz = confusion_matrix(reales, predichos, labels=etiquetas)
    aciertos = int(np.sum(reales == predichos))
    return {
        "accuracy": float(accuracy_score(reales, predichos)),
        "balanced_accuracy": balanced_accuracy(reales, predichos),
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
    return float(np.mean(recalls)) if recalls else 0.0


def construir_membresias_base():
    return {
        variable: {
            categoria: np.asarray(puntos, dtype=float)
            for categoria, puntos in especificacion["categorias"].items()
        }
        for variable, especificacion in ESPECIFICACIONES_VARIABLES.items()
    }


def resumir_reglas(reglas: list[dict], duplicadas: int):
    longitudes = [len(regla.get("antecedentes", [])) for regla in reglas]
    por_clase = {clase: 0 for clase in CLASES}
    for regla in reglas:
        clase = regla.get("consecuente")
        clase = CONSECUENTE_A_CLASE.get(clase, clase)
        if clase in por_clase:
            por_clase[clase] += 1
    return {
        "total_reglas": int(len(reglas)),
        "reglas_duplicadas_eliminadas": int(duplicadas),
        "reglas_por_clase": por_clase,
        "antecedentes_promedio": float(np.mean(longitudes)) if longitudes else 0.0,
        "antecedentes_min": int(np.min(longitudes)) if longitudes else 0,
        "antecedentes_max": int(np.max(longitudes)) if longitudes else 0,
    }


def renumerar_reglas(reglas: list[dict]):
    salida = []
    for numero, regla in enumerate(reglas, start=1):
        nueva = dict(regla)
        nueva["id"] = f"R{numero:03d}"
        salida.append(nueva)
    return salida


def sin_matriz(metricas: dict):
    salida = dict(metricas)
    salida.pop("matriz_confusion", None)
    return salida


def fila_resumen(resultado: dict):
    metricas_limpias = resultado["metricas_limpias"]
    metricas_originales = resultado["metricas_originales"]
    reglas_limpias = resultado["resumen_reglas_limpias"]
    reglas_originales = resultado["resumen_reglas_original"]
    return {
        "iteracion": resultado["iteracion"],
        "algoritmo": resultado["algoritmo"],
        "accuracy_original": metricas_originales.get("accuracy"),
        "ba_original": metricas_originales.get("balanced_accuracy"),
        "accuracy_limpia": metricas_limpias["accuracy"],
        "ba_limpia": metricas_limpias["balanced_accuracy"],
        "delta_accuracy": metricas_limpias["accuracy"] - metricas_originales.get("accuracy", 0.0),
        "delta_ba": metricas_limpias["balanced_accuracy"] - metricas_originales.get("balanced_accuracy", 0.0),
        "reglas_originales": reglas_originales.get("total_reglas", len(resultado["reglas_finales"])),
        "reglas_limpias": reglas_limpias["total_reglas"],
        "duplicadas_eliminadas": reglas_limpias["reglas_duplicadas_eliminadas"],
        "antecedentes_promedio": reglas_limpias["antecedentes_promedio"],
        "antecedentes_min": reglas_limpias["antecedentes_min"],
        "antecedentes_max": reglas_limpias["antecedentes_max"],
    }


def construir_resumen(filas: list[dict]):
    if not filas:
        return {
            "tabla_iteraciones": [],
            "resumen_estadistico": [],
            "mejor_global": None,
            "mejor_por_algoritmo": {},
        }

    tabla = pd.DataFrame(filas)
    mejor_global = tabla.sort_values(
        ["ba_limpia", "accuracy_limpia"],
        ascending=False,
    ).iloc[0].to_dict()

    mejores = {}
    for algoritmo, grupo in tabla.groupby("algoritmo"):
        mejores[algoritmo] = grupo.sort_values(
            ["ba_limpia", "accuracy_limpia"],
            ascending=False,
        ).iloc[0].to_dict()

    return {
        "criterio": "mayor balanced_accuracy recalculada sin reglas duplicadas",
        "mejor_global": mejor_global,
        "mejor_por_algoritmo": mejores,
        "resumen_estadistico": resumen_estadistico(tabla),
        "tabla_iteraciones": filas,
    }


def guardar_matriz_mejor_ba(resultados: list[dict], salida: Path):
    if not resultados:
        return None

    mejor = max(
        resultados,
        key=lambda resultado: (
            resultado["metricas_limpias"]["balanced_accuracy"],
            resultado["metricas_limpias"]["accuracy"],
        ),
    )
    carpeta = salida / "matrices_confusion"
    ruta = carpeta / "mejor_ba_sin_duplicadas.png"
    graficar_matriz_confusion(mejor, ruta)
    return ruta


def graficar_matriz_confusion(resultado: dict, ruta: Path):
    import matplotlib.pyplot as plt

    matriz_info = resultado["matriz_confusion_limpia"]
    etiquetas = matriz_info["etiquetas"]
    matriz = np.asarray(matriz_info["matriz"], dtype=int)
    metricas = resultado["metricas_limpias"]
    reglas = resultado["resumen_reglas_limpias"]

    figura, eje = plt.subplots(figsize=(9, 7), constrained_layout=True)
    imagen = eje.imshow(matriz, cmap="Blues")
    figura.colorbar(imagen, ax=eje, fraction=0.046, pad=0.04)

    eje.set_xticks(np.arange(len(etiquetas)))
    eje.set_yticks(np.arange(len(etiquetas)))
    eje.set_xticklabels(etiquetas, rotation=35, ha="right")
    eje.set_yticklabels(etiquetas)
    eje.set_xlabel("Prediccion")
    eje.set_ylabel("Clase real")
    eje.set_title(
        "Mejor matriz de confusion sin duplicadas\n"
        f"{resultado['algoritmo']} | Iteracion {resultado['iteracion']:02d} | "
        f"BA={metricas['balanced_accuracy']:.4f} | "
        f"Accuracy={metricas['accuracy']:.4f} | "
        f"Reglas={reglas['total_reglas']}"
    )

    umbral = matriz.max() / 2 if matriz.size else 0
    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            color = "white" if matriz[fila, columna] > umbral else "black"
            eje.text(
                columna,
                fila,
                str(matriz[fila, columna]),
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )

    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=160)
    plt.close(figura)


def resumen_estadistico(tabla: pd.DataFrame):
    filas = []
    for algoritmo, grupo in tabla.groupby("algoritmo"):
        filas.append({
            "algoritmo": algoritmo,
            "iteraciones": int(len(grupo)),
            "accuracy_promedio": promedio(grupo["accuracy_limpia"]),
            "accuracy_desviacion_estandar": desviacion(grupo["accuracy_limpia"]),
            "error_promedio": promedio(1.0 - grupo["accuracy_limpia"]),
            "error_desviacion_estandar": desviacion(1.0 - grupo["accuracy_limpia"]),
            "balanced_accuracy_promedio": promedio(grupo["ba_limpia"]),
            "balanced_accuracy_desviacion_estandar": desviacion(grupo["ba_limpia"]),
            "error_balanceado_promedio": promedio(1.0 - grupo["ba_limpia"]),
            "error_balanceado_desviacion_estandar": desviacion(1.0 - grupo["ba_limpia"]),
            "reglas_limpias_promedio": promedio(grupo["reglas_limpias"]),
            "reglas_limpias_desviacion_estandar": desviacion(grupo["reglas_limpias"]),
            "duplicadas_eliminadas_promedio": promedio(grupo["duplicadas_eliminadas"]),
            "antecedentes_promedio": promedio(grupo["antecedentes_promedio"]),
            "antecedentes_desviacion_estandar": desviacion(grupo["antecedentes_promedio"]),
            "mejor_iteracion": int(grupo.sort_values(["ba_limpia", "accuracy_limpia"], ascending=False).iloc[0]["iteracion"]),
            "mejor_accuracy": float(grupo["accuracy_limpia"].max()),
            "mejor_balanced_accuracy": float(grupo["ba_limpia"].max()),
        })
    return filas


def promedio(valores):
    return float(np.mean(np.asarray(valores, dtype=float))) if len(valores) else 0.0


def desviacion(valores):
    valores = np.asarray(valores, dtype=float)
    return float(np.std(valores, ddof=1)) if len(valores) > 1 else 0.0


def imprimir_resultado(resultado: dict):
    metricas = resultado["metricas_limpias"]
    reglas = resultado["resumen_reglas_limpias"]
    print(
        f"Iteracion {resultado['iteracion']:02d} | {resultado['algoritmo']} | "
        f"reglas={reglas['total_reglas']} "
        f"dup_elim={reglas['reglas_duplicadas_eliminadas']} | "
        f"acc={metricas['accuracy']:.4f} | ba={metricas['balanced_accuracy']:.4f}"
    )


def extraer_iteracion(ruta_json: Path):
    nombre = ruta_json.parent.name.replace("iteracion_", "")
    return int(nombre)


def guardar_json(ruta: Path, contenido: dict):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(serializar(contenido), indent=2, ensure_ascii=False), encoding="utf-8")


def guardar_csv(ruta: Path, filas: list[dict]):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not filas:
        ruta.write_text("", encoding="utf-8")
        return
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)


def serializar(valor):
    if isinstance(valor, dict):
        return {k: serializar(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [serializar(v) for v in valor]
    if isinstance(valor, tuple):
        return [serializar(v) for v in valor]
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if hasattr(valor, "item"):
        return valor.item()
    return valor


if __name__ == "__main__":
    main()
