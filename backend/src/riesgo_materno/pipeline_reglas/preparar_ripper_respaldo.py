"""Genera y publica la base RIPPER de respaldo con antecedentes variables.

Uso desde backend:
    python -m src.riesgo_materno.pipeline_reglas.preparar_ripper_respaldo

El script reproduce el respaldo usado por el sistema web:
1. Lee las corridas RIPPER incompletas versionadas.
2. Consolida todas las corridas eliminando duplicados exactos.
3. Publica el resultado en reglas_sistema_difuso_ripper.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from ..entrenamiento.datos import cargar_dataset, convertir_split_a_diccionario
from ..entrenamiento.modelo import (
    RUTA_CSV,
    RUTA_PAQUETE,
    RUTA_REGLAS_SISTEMA_DIFUSO_RIPPER,
)
from ..logica_difusa.motor import SistemaDifusoMamdani
from ..logica_difusa.variables import ESPECIFICACIONES_VARIABLES


CLASES = ["low risk", "mid risk", "high risk"]
SIN_ACTIVACION = "__sin_activacion__"
CLASE_A_CONSECUENTE = {"low risk": "bajo", "mid risk": "medio", "high risk": "alto"}
CONSECUENTE_A_CLASE = {v: k for k, v in CLASE_A_CONSECUENTE.items()}

RUTA_REGLAS_RIPPER_INCOMPLETAS = RUTA_PAQUETE / "reglas" / "respaldo_ripper"
RUTA_METADATA_RIPPER_RESPALDO = RUTA_PAQUETE / "reglas" / "metadata_reglas_sistema_difuso_ripper.json"
RUTA_BACKEND = RUTA_PAQUETE.parents[1]

CONFIG_DEFAULT = {
    "iteraciones": 20,
}


def main():
    config = CONFIG_DEFAULT
    tabla = cargar_dataset(RUTA_CSV)

    print("Generando reglas RIPPER de respaldo")
    print(f"  Dataset: {len(tabla)} instancias")
    print(f"  Iteraciones: {config['iteraciones']}")
    print("  Tipo de reglas: antecedentes variables")

    resultados = cargar_iteraciones()
    todas_reglas = [regla for resultado in resultados for regla in resultado["reglas_finales"]]

    for resultado in resultados:
        metricas = resultado["metricas"]
        print(
            f"  Iteracion {resultado['iteracion']:02d} | reglas={len(resultado['reglas_finales'])} | "
            f"BA={metricas['balanced_accuracy']:.4f} | "
            f"Accuracy={metricas['accuracy']:.4f} | "
            f"sin_activacion={metricas['sin_activacion']}"
        )

    reglas_consolidadas, duplicadas = consolidar_reglas(todas_reglas)
    reglas_motor = reglas_a_motor(reglas_consolidadas)
    metricas_consolidadas = evaluar_reglas(reglas_motor, tabla)
    metadata = construir_metadata(config, resultados, reglas_consolidadas, duplicadas, metricas_consolidadas)
    guardar_json(RUTA_REGLAS_SISTEMA_DIFUSO_RIPPER, reglas_motor)
    guardar_json(RUTA_METADATA_RIPPER_RESPALDO, metadata)

    print("\nRIPPER de respaldo publicado")
    print(f"  Reglas generadas antes de limpiar: {len(todas_reglas)}")
    print(f"  Duplicadas eliminadas: {duplicadas}")
    print(f"  Reglas finales: {len(reglas_motor)}")
    print(
        f"  Accuracy={metricas_consolidadas['accuracy']:.4f} | "
        f"BA={metricas_consolidadas['balanced_accuracy']:.4f} | "
        f"sin_activacion={metricas_consolidadas['sin_activacion']}"
    )
    print(f"  Archivo web: {ruta_relativa(RUTA_REGLAS_SISTEMA_DIFUSO_RIPPER)}")
    print(f"  Metadata: {ruta_relativa(RUTA_METADATA_RIPPER_RESPALDO)}")


def cargar_iteraciones():
    archivos = sorted(RUTA_REGLAS_RIPPER_INCOMPLETAS.glob("iteracion_*/ripper.json"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron reglas RIPPER en {RUTA_REGLAS_RIPPER_INCOMPLETAS}")
    resultados = []
    for archivo in archivos:
        resultado = leer_json(archivo)
        reglas = [
            normalizar_regla(regla, indice)
            for indice, regla in enumerate(resultado.get("reglas_finales", []), start=1)
        ]
        resultado["reglas_finales"] = reglas
        resultado["resumen_reglas"] = resumir_reglas(reglas, duplicadas=0)
        resultados.append(resultado)
    return resultados


def normalizar_regla(regla: dict, posicion: int):
    antecedentes = []
    for antecedente in regla["antecedentes"]:
        if isinstance(antecedente, dict):
            variable = antecedente["variable"]
            etiqueta = antecedente.get("etiqueta_linguistica") or antecedente.get("etiqueta")
        else:
            variable, etiqueta = antecedente
        antecedentes.append({"variable": variable, "etiqueta_linguistica": etiqueta})

    consecuente = regla["consecuente"]
    if consecuente in CONSECUENTE_A_CLASE:
        consecuente = CONSECUENTE_A_CLASE[consecuente]

    return {
        "id": regla.get("id", f"R{posicion:03d}"),
        "antecedentes": antecedentes,
        "consecuente": consecuente,
    }


def consolidar_reglas(reglas: list[dict]):
    vistas = set()
    consolidadas = []
    duplicadas = 0
    for regla in reglas:
        clave = clave_regla(regla)
        if clave in vistas:
            duplicadas += 1
            continue
        vistas.add(clave)
        nueva = dict(regla)
        nueva["id"] = f"R{len(consolidadas) + 1:03d}"
        consolidadas.append(nueva)
    return consolidadas, duplicadas


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


def evaluar_reglas(reglas_motor: list[dict], tabla: pd.DataFrame):
    datos = convertir_split_a_diccionario(tabla)
    sistema = SistemaDifusoMamdani(construir_membresias_base(), reglas=reglas_motor)
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


def construir_metadata(config: dict, resultados: list[dict], reglas_consolidadas: list[dict], duplicadas: int, metricas: dict):
    return {
        "iteraciones": len(resultados),
        "metricas": sin_matriz(metricas),
        "resumen_reglas": resumir_reglas(reglas_consolidadas, duplicadas),
        "matriz_confusion": metricas["matriz_confusion"],
        "corridas": [
            {
                "iteracion": resultado["iteracion"],
                "reglas": resultado["resumen_reglas"]["total_reglas"],
                "balanced_accuracy": resultado["metricas"]["balanced_accuracy"],
                "accuracy": resultado["metricas"]["accuracy"],
                "sin_activacion": resultado["metricas"]["sin_activacion"],
            }
            for resultado in resultados
        ],
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


def leer_json(ruta: Path):
    return json.loads(ruta.read_text(encoding="utf-8-sig"))


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
