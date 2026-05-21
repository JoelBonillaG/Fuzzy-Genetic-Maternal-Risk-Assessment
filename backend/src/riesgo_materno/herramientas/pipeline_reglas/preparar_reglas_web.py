"""Prepara reglas limpias del AG y las publica para el motor difuso web.

Uso desde backend:
    python -m src.riesgo_materno.herramientas.pipeline_reglas.preparar_reglas_web

Por defecto toma la iteracion 12 del AG, elimina reglas duplicadas, recalcula
metricas con el dataset completo y publica el JSON que consume el motor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from ...entrenamiento.datos import cargar_dataset, convertir_split_a_diccionario
from ...entrenamiento.modelo import (
    RUTA_CSV,
    RUTA_METADATA_REGLAS_SISTEMA_DIFUSO,
    RUTA_PAQUETE,
    RUTA_REGLAS_LIMPIAS,
    RUTA_REGLAS_SISTEMA_DIFUSO,
)
from ...logica_difusa.motor import SistemaDifusoMamdani
from ...logica_difusa.variables import ESPECIFICACIONES_VARIABLES


CLASES = ["low risk", "mid risk", "high risk"]
SIN_ACTIVACION = "__sin_activacion__"
CLASE_A_CONSECUENTE = {"low risk": "bajo", "mid risk": "medio", "high risk": "alto"}
CONSECUENTE_A_CLASE = {v: k for k, v in CLASE_A_CONSECUENTE.items()}

RUTA_RESULTADOS = RUTA_PAQUETE / "herramientas" / "pipeline_reglas" / "resultados"
RUTA_BACKEND = RUTA_PAQUETE.parents[1]


def main():
    args = crear_parser().parse_args()
    entrada = Path(args.entrada) if args.entrada else resolver_entrada_por_defecto()
    iteracion = None if args.todas_iteraciones else args.iteracion
    candidatos = cargar_candidatos(entrada, args.algoritmo, iteracion)
    if not candidatos:
        raise FileNotFoundError("No se encontraron resultados de reglas para publicar.")

    tabla = cargar_dataset(RUTA_CSV)
    datos = convertir_split_a_diccionario(tabla)
    membresias = construir_membresias_base()
    evaluados = [preparar_candidato(candidato, datos, membresias) for candidato in candidatos]
    seleccionado = seleccionar_candidato(evaluados)

    guardar_json(Path(args.salida_web), seleccionado["reglas_motor"])
    guardar_json(Path(args.salida_metadata), construir_metadata(seleccionado, args))

    metricas = seleccionado["metricas_limpias"]
    print("Reglas publicadas para el sistema difuso web")
    print(f"  Origen: {seleccionado['ruta_origen']}")
    print(f"  Iteracion: {seleccionado['iteracion']}")
    print(f"  Algoritmo: {seleccionado['algoritmo']}")
    print(
        f"  Reglas limpias: {len(seleccionado['reglas_comunes'])} | "
        f"duplicadas eliminadas: {seleccionado['duplicadas_eliminadas']}"
    )
    print(
        f"  Accuracy={metricas['accuracy']:.4f} | "
        f"BA={metricas['balanced_accuracy']:.4f} | "
        f"sin_activacion={metricas['sin_activacion']}"
    )
    print(f"  Archivo web: {ruta_relativa(Path(args.salida_web))}")


def crear_parser():
    parser = argparse.ArgumentParser(description="Publica reglas limpias para el motor difuso web.")
    parser.add_argument("--entrada", default=None, help="JSON de resultado o carpeta con iteraciones.")
    parser.add_argument("--algoritmo", default="AG_MICHIGAN_BINARIO")
    parser.add_argument("--iteracion", type=int, default=12)
    parser.add_argument(
        "--todas-iteraciones",
        action="store_true",
        help="Ignora --iteracion y publica la base con mayor BA limpia.",
    )
    parser.add_argument("--salida-web", default=str(RUTA_REGLAS_SISTEMA_DIFUSO))
    parser.add_argument("--salida-metadata", default=str(RUTA_METADATA_REGLAS_SISTEMA_DIFUSO))
    return parser


def cargar_candidatos(entrada: Path, algoritmo: str, iteracion: int | None):
    rutas = [entrada] if entrada.is_file() else buscar_jsons(entrada)
    candidatos = []
    for ruta in rutas:
        contenido = leer_json(ruta)
        if not isinstance(contenido, dict) or "reglas_finales" not in contenido:
            continue
        if algoritmo and contenido.get("algoritmo") != algoritmo:
            continue
        if iteracion is not None and int(contenido.get("iteracion", -1)) != iteracion:
            continue
        candidatos.append({"ruta": ruta, "contenido": contenido})
    return candidatos


def buscar_jsons(entrada: Path):
    if entrada.name.startswith("iteracion_"):
        return sorted(entrada.glob("*.json"))

    versiones = [
        carpeta
        for carpeta in entrada.glob("v*")
        if carpeta.is_dir() and carpeta.name[1:].isdigit()
    ]
    if versiones:
        entrada = max(versiones, key=lambda carpeta: int(carpeta.name[1:]))

    return sorted(entrada.glob("iteracion_*/*.json"))


def resolver_entrada_por_defecto():
    versiones_limpias = [
        carpeta
        for carpeta in RUTA_REGLAS_LIMPIAS.glob("v*")
        if carpeta.is_dir() and carpeta.name[1:].isdigit()
    ]
    if versiones_limpias:
        return max(versiones_limpias, key=lambda carpeta: int(carpeta.name[1:]))
    return RUTA_RESULTADOS


def preparar_candidato(candidato: dict, datos: dict, membresias: dict):
    contenido = candidato["contenido"]
    reglas_comunes = normalizar_reglas_comunes(contenido["reglas_finales"])
    reglas_limpias, duplicadas = quitar_duplicadas(reglas_comunes)
    reglas_motor = reglas_a_motor(reglas_limpias)
    metricas = evaluar_reglas(reglas_motor, datos, membresias)
    return {
        "ruta_origen": ruta_relativa(candidato["ruta"]),
        "id_experimento": contenido.get("id_experimento"),
        "iteracion": int(contenido.get("iteracion", 0)),
        "algoritmo": contenido.get("algoritmo"),
        "datos": contenido.get("datos", {}),
        "hiperparametros": contenido.get("hiperparametros", {}),
        "metricas_originales": contenido.get("metricas", {}),
        "metricas_limpias": metricas,
        "reglas_comunes": renumerar_reglas(reglas_limpias),
        "reglas_motor": reglas_motor,
        "duplicadas_eliminadas": duplicadas,
    }


def seleccionar_candidato(candidatos: list[dict]):
    return max(
        candidatos,
        key=lambda candidato: (
            candidato["metricas_limpias"]["balanced_accuracy"],
            candidato["metricas_limpias"]["accuracy"],
        ),
    )


def normalizar_reglas_comunes(reglas: list[dict]):
    salida = []
    for regla in reglas:
        antecedentes = []
        for antecedente in regla.get("antecedentes", []):
            if isinstance(antecedente, dict):
                antecedentes.append({
                    "variable": antecedente["variable"],
                    "etiqueta_linguistica": antecedente["etiqueta_linguistica"],
                })
            else:
                variable, etiqueta = antecedente
                antecedentes.append({
                    "variable": variable,
                    "etiqueta_linguistica": etiqueta,
                })

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
        for antecedente in regla["antecedentes"]
    ))
    return antecedentes, regla["consecuente"]


def reglas_a_motor(reglas: list[dict]):
    salida = []
    for numero, regla in enumerate(reglas, start=1):
        salida.append({
            "numero": numero,
            "antecedentes": [
                [antecedente["variable"], antecedente["etiqueta_linguistica"]]
                for antecedente in regla["antecedentes"]
            ],
            "consecuente": CLASE_A_CONSECUENTE[regla["consecuente"]],
        })
    return salida


def renumerar_reglas(reglas: list[dict]):
    salida = []
    for numero, regla in enumerate(reglas, start=1):
        nueva = dict(regla)
        nueva["id"] = f"R{numero:03d}"
        salida.append(nueva)
    return salida


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


def construir_metadata(candidato: dict, args):
    return {
        "archivo_reglas": ruta_relativa(Path(args.salida_web)),
        "archivo_origen": candidato["ruta_origen"],
        "id_experimento": candidato["id_experimento"],
        "iteracion": candidato["iteracion"],
        "algoritmo": candidato["algoritmo"],
        "criterio": "mayor balanced_accuracy despues de limpiar duplicadas",
        "metricas_limpias": sin_matriz(candidato["metricas_limpias"]),
        "resumen_reglas_limpias": resumir_reglas(
            candidato["reglas_comunes"],
            candidato["duplicadas_eliminadas"],
        ),
    }


def resumir_reglas(reglas: list[dict], duplicadas: int):
    longitudes = [len(regla["antecedentes"]) for regla in reglas]
    por_clase = {clase: 0 for clase in CLASES}
    for regla in reglas:
        por_clase[regla["consecuente"]] += 1
    return {
        "total_reglas": int(len(reglas)),
        "duplicadas_eliminadas": int(duplicadas),
        "reglas_por_clase": por_clase,
        "antecedentes_promedio": float(np.mean(longitudes)) if longitudes else 0.0,
        "antecedentes_min": int(np.min(longitudes)) if longitudes else 0,
        "antecedentes_max": int(np.max(longitudes)) if longitudes else 0,
    }


def construir_membresias_base():
    return {
        variable: {
            categoria: np.asarray(puntos, dtype=float)
            for categoria, puntos in especificacion["categorias"].items()
        }
        for variable, especificacion in ESPECIFICACIONES_VARIABLES.items()
    }


def sin_matriz(metricas: dict):
    salida = dict(metricas)
    salida.pop("matriz_confusion", None)
    return salida


def leer_json(ruta: Path):
    return json.loads(ruta.read_text(encoding="utf-8-sig"))


def ruta_relativa(ruta: Path | str):
    ruta = Path(ruta)
    try:
        return ruta.resolve().relative_to(RUTA_BACKEND).as_posix()
    except ValueError:
        return ruta.as_posix()


def guardar_json(ruta: Path, contenido):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(serializar(contenido), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def serializar(valor):
    if isinstance(valor, dict):
        return {clave: serializar(item) for clave, item in valor.items()}
    if isinstance(valor, list):
        return [serializar(item) for item in valor]
    if isinstance(valor, tuple):
        return [serializar(item) for item in valor]
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if hasattr(valor, "item"):
        return valor.item()
    return valor


if __name__ == "__main__":
    main()
