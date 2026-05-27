"""Limpia reglas duplicadas por iteracion y recalcula metricas.

Uso desde backend:
    python -m src.riesgo_materno.pipeline_reglas.limpiar_reglas_iteraciones

Lee resultados de comparacion y guarda las reglas limpias en:
    src/riesgo_materno/reglas/experimentos/vN
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from ..entrenamiento.datos import cargar_dataset, convertir_split_a_diccionario
from ..entrenamiento.modelo import RUTA_CSV, RUTA_PAQUETE, RUTA_REGLAS_LIMPIAS
from ..logica_difusa.motor import SistemaDifusoMamdani
from ..logica_difusa.variables import ESPECIFICACIONES_VARIABLES


CLASES = ["low risk", "mid risk", "high risk"]
SIN_ACTIVACION = "__sin_activacion__"
CLASE_A_CONSECUENTE = {"low risk": "bajo", "mid risk": "medio", "high risk": "alto"}
CONSECUENTE_A_CLASE = {v: k for k, v in CLASE_A_CONSECUENTE.items()}

RUTA_RESULTADOS = RUTA_PAQUETE / "pipeline_reglas" / "resultados"
RUTA_BACKEND = RUTA_PAQUETE.parents[1]


def main():
    args = crear_parser().parse_args()
    entrada = resolver_entrada(Path(args.entrada) if args.entrada else RUTA_RESULTADOS)
    salida = Path(args.salida) if args.salida else resolver_salida(entrada)
    resumen = limpiar_resultados(entrada, salida, args.algoritmo)
    print(f"\nReglas limpias guardadas en: {ruta_relativa(salida)}")
    if resumen["mejor_ag"] is not None:
        mejor = resumen["mejor_ag"]
        print(
            "Mejor AG limpio | "
            f"iteracion={int(mejor['iteracion']):02d} | "
            f"reglas={int(mejor['reglas_limpias'])} | "
            f"BA={mejor['ba_limpia']:.4f} | "
            f"Accuracy={mejor['accuracy_limpia']:.4f}"
        )


def crear_parser():
    parser = argparse.ArgumentParser(description="Limpia duplicadas y recalcula metricas por iteracion.")
    parser.add_argument("--entrada", default=None, help="Carpeta resultados, vN o iteracion_XX.")
    parser.add_argument("--salida", default=None, help="Carpeta donde guardar las reglas limpias.")
    parser.add_argument("--algoritmo", default=None, help="Filtra por algoritmo, por ejemplo GENETIC_ALGORITHM.")
    return parser


def limpiar_resultados(entrada: Path | None = None, salida: Path | None = None, algoritmo: str | None = None):
    entrada = resolver_entrada(entrada or RUTA_RESULTADOS)
    salida = salida or resolver_salida(entrada)
    salida.mkdir(parents=True, exist_ok=True)

    tabla = cargar_dataset(RUTA_CSV)
    datos = convertir_split_a_diccionario(tabla)
    membresias = construir_membresias_base()

    resultados = []
    filas = []
    for ruta_json in archivos_de_resultados(entrada):
        contenido = leer_json(ruta_json)
        if not isinstance(contenido, dict) or "reglas_finales" not in contenido:
            continue
        if algoritmo and contenido.get("algoritmo") != algoritmo:
            continue
        resultado = procesar_resultado(ruta_json, contenido, salida, datos, membresias)
        resultados.append(resultado)
        filas.append(fila_resumen(resultado))
        imprimir_resultado(resultado)

    resumen = construir_resumen(filas)
    guardar_csv(salida / "resumen_iteraciones.csv", filas)
    guardar_csv(salida / "resumen_estadistico.csv", resumen["resumen_estadistico"])
    guardar_json(salida / "resumen_metricas.json", resumen)
    guardar_mejores(resultados, resumen, salida)
    return resumen


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
    return ruta


def resolver_salida(entrada: Path):
    if entrada.name.startswith("v") and entrada.name[1:].isdigit():
        return RUTA_REGLAS_LIMPIAS / entrada.name
    return crear_carpeta_versionada(RUTA_REGLAS_LIMPIAS)


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


def procesar_resultado(ruta_json: Path, contenido: dict, salida: Path, datos: dict, membresias: dict):
    reglas_originales = normalizar_reglas(contenido.get("reglas_finales", []))
    reglas_limpias, duplicadas = quitar_duplicadas(reglas_originales)
    reglas_motor = reglas_a_motor(reglas_limpias)
    metricas = evaluar_reglas(reglas_motor, datos, membresias)

    iteracion = int(contenido.get("iteracion", extraer_iteracion(ruta_json)))
    algoritmo = normalizar_algoritmo(contenido.get("algoritmo", ruta_json.stem))
    resultado = {
        "id_experimento": contenido.get("id_experimento"),
        "iteracion": iteracion,
        "algoritmo": algoritmo,
        "archivo_origen": ruta_relativa(ruta_json),
        "datos": contenido.get("datos", {}),
        "hiperparametros": contenido.get("hiperparametros", {}),
        "metricas_originales": contenido.get("metricas", {}),
        "matriz_confusion_original": contenido.get("matriz_confusion", {}),
        "resumen_reglas_original": contenido.get("resumen_reglas", {}),
        "resumen_reglas_limpias": resumir_reglas(reglas_limpias, duplicadas),
        "metricas_limpias": sin_matriz(metricas),
        "matriz_confusion_limpia": metricas["matriz_confusion"],
        "reglas_finales": renumerar_reglas(reglas_limpias),
        "reglas_motor": reglas_motor,
    }

    carpeta_salida = salida / f"iteracion_{iteracion:02d}"
    guardar_json(carpeta_salida / ruta_json.name, sin_reglas_motor(resultado))
    return resultado


def normalizar_reglas(reglas: list[dict]):
    salida = []
    for regla in reglas:
        antecedentes = []
        for antecedente in regla.get("antecedentes", []):
            if isinstance(antecedente, dict):
                antecedentes.append({
                    "variable": antecedente["variable"],
                    "etiqueta_linguistica": antecedente.get(
                        "etiqueta_linguistica",
                        antecedente.get("categoria"),
                    ),
                })
            else:
                variable, etiqueta = antecedente
                antecedentes.append({"variable": variable, "etiqueta_linguistica": etiqueta})

        consecuente = regla.get("consecuente")
        salida.append({
            "id": regla.get("id"),
            "antecedentes": antecedentes,
            "consecuente": CONSECUENTE_A_CLASE.get(consecuente, consecuente),
            "origen": regla.get("origen") or regla.get("source"),
        })
    return salida


def quitar_duplicadas(reglas: list[dict]):
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
        reglas_motor.append({
            "numero": numero,
            "antecedentes": [
                [antecedente["variable"], antecedente["etiqueta_linguistica"]]
                for antecedente in regla["antecedentes"]
            ],
            "consecuente": CLASE_A_CONSECUENTE[regla["consecuente"]],
        })
    return reglas_motor


def evaluar_reglas(reglas_motor: list[dict], datos: dict, membresias: dict):
    sistema = SistemaDifusoMamdani(membresias, reglas=reglas_motor)
    inferencia = sistema.inferir_lote(datos["entradas"])
    return calcular_metricas(datos["riesgos"], inferencia["riesgos"], inferencia["sin_activacion"])


def calcular_metricas(reales, predichos, sin_activacion):
    reales = np.asarray(reales, dtype=object)
    predichos = np.asarray(predichos, dtype=object).copy()
    predichos[np.asarray(sin_activacion, dtype=bool)] = SIN_ACTIVACION
    etiquetas = CLASES + [SIN_ACTIVACION]
    matriz = confusion_matrix(reales, predichos, labels=etiquetas)
    accuracy = float(accuracy_score(reales, predichos))
    ba = balanced_accuracy(reales, predichos)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": ba,
        "error_clasificacion": float(1.0 - accuracy),
        "error_balanceado": float(1.0 - ba),
        "aciertos": int(np.sum(reales == predichos)),
        "errores": int(np.sum(reales != predichos)),
        "total": int(len(reales)),
        "sin_activacion": int(np.sum(predichos == SIN_ACTIVACION)),
        "matriz_confusion": {"etiquetas": etiquetas, "matriz": matriz.tolist()},
    }


def balanced_accuracy(reales, predichos):
    recalls = []
    for clase in CLASES:
        mascara = reales == clase
        if np.any(mascara):
            recalls.append(float(np.mean(predichos[mascara] == clase)))
    return float(np.mean(recalls)) if recalls else 0.0


def resumir_reglas(reglas: list[dict], duplicadas: int):
    longitudes = [len(regla.get("antecedentes", [])) for regla in reglas]
    por_clase = {clase: 0 for clase in CLASES}
    for regla in reglas:
        clase = regla.get("consecuente")
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


def fila_resumen(resultado: dict):
    limpias = resultado["metricas_limpias"]
    originales = resultado["metricas_originales"]
    reglas = resultado["resumen_reglas_limpias"]
    reglas_originales = resultado["resumen_reglas_original"]
    return {
        "iteracion": resultado["iteracion"],
        "algoritmo": resultado["algoritmo"],
        "accuracy_original": originales.get("accuracy"),
        "ba_original": originales.get("balanced_accuracy"),
        "accuracy_limpia": limpias["accuracy"],
        "ba_limpia": limpias["balanced_accuracy"],
        "delta_accuracy": limpias["accuracy"] - originales.get("accuracy", 0.0),
        "delta_ba": limpias["balanced_accuracy"] - originales.get("balanced_accuracy", 0.0),
        "reglas_originales": reglas_originales.get("total_reglas", len(resultado["reglas_finales"])),
        "reglas_limpias": reglas["total_reglas"],
        "duplicadas_eliminadas": reglas["reglas_duplicadas_eliminadas"],
        "antecedentes_promedio": reglas["antecedentes_promedio"],
        "antecedentes_min": reglas["antecedentes_min"],
        "antecedentes_max": reglas["antecedentes_max"],
        "sin_activacion": limpias["sin_activacion"],
    }


def construir_resumen(filas: list[dict]):
    if not filas:
        return {
            "mejor_global": None,
            "mejor_ag": None,
            "mejor_por_algoritmo": {},
            "resumen_estadistico": [],
            "tabla_iteraciones": [],
        }

    tabla = pd.DataFrame(filas)
    mejor_global = mejor_fila(tabla)
    tabla_ag = tabla[tabla["algoritmo"] == "GENETIC_ALGORITHM"]
    mejor_ag = mejor_fila(tabla_ag) if not tabla_ag.empty else None

    mejores = {}
    for algoritmo, grupo in tabla.groupby("algoritmo"):
        mejores[algoritmo] = mejor_fila(grupo)

    return {
        "mejor_global": mejor_global,
        "mejor_ag": mejor_ag,
        "mejor_por_algoritmo": mejores,
        "resumen_estadistico": resumen_estadistico(tabla),
        "tabla_iteraciones": filas,
    }


def mejor_fila(tabla: pd.DataFrame):
    if tabla.empty:
        return None
    return tabla.sort_values(["ba_limpia", "accuracy_limpia"], ascending=False).iloc[0].to_dict()


def normalizar_algoritmo(nombre: str):
    equivalencias = {
        "AG_MICHIGAN_BINARIO": "GENETIC_ALGORITHM",
        "genetic_algorithm": "GENETIC_ALGORITHM",
        "PRISM_ESTOCASTICO": "PRISM",
        "prism": "PRISM",
        "ripper": "RIPPER",
    }
    return equivalencias.get(nombre, nombre)


def resumen_estadistico(tabla: pd.DataFrame):
    filas = []
    for algoritmo, grupo in tabla.groupby("algoritmo"):
        mejor = mejor_fila(grupo)
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
            "mejor_iteracion": int(mejor["iteracion"]),
            "mejor_accuracy": float(mejor["accuracy_limpia"]),
            "mejor_balanced_accuracy": float(mejor["ba_limpia"]),
        })
    return filas


def guardar_mejores(resultados: list[dict], resumen: dict, salida: Path):
    guardar_mejor(resultados, resumen.get("mejor_global"), salida / "mejor_global_limpio.json")
    guardar_mejor(resultados, resumen.get("mejor_ag"), salida / "mejor_ag_limpio.json")


def guardar_mejor(resultados: list[dict], fila: dict | None, ruta: Path):
    if not fila:
        return
    for resultado in resultados:
        if (
            resultado["iteracion"] == int(fila["iteracion"])
            and resultado["algoritmo"] == fila["algoritmo"]
        ):
            guardar_json(ruta, sin_reglas_motor(resultado))
            return


def construir_membresias_base():
    return {
        variable: {
            categoria: np.asarray(puntos, dtype=float)
            for categoria, puntos in especificacion["categorias"].items()
        }
        for variable, especificacion in ESPECIFICACIONES_VARIABLES.items()
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


def sin_reglas_motor(resultado: dict):
    salida = dict(resultado)
    salida.pop("reglas_motor", None)
    salida.pop("id_experimento", None)
    salida.pop("algoritmo", None)
    salida.pop("archivo_origen", None)
    salida.pop("datos", None)
    salida.pop("hiperparametros", None)
    salida.pop("metricas_originales", None)
    salida.pop("matriz_confusion_original", None)
    salida.pop("resumen_reglas_original", None)
    return salida


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
    return int(ruta_json.parent.name.replace("iteracion_", ""))


def leer_json(ruta: Path):
    return json.loads(ruta.read_text(encoding="utf-8-sig"))


def ruta_relativa(ruta: Path | str):
    ruta = Path(ruta)
    try:
        return ruta.resolve().relative_to(RUTA_BACKEND).as_posix()
    except ValueError:
        return ruta.as_posix()


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
