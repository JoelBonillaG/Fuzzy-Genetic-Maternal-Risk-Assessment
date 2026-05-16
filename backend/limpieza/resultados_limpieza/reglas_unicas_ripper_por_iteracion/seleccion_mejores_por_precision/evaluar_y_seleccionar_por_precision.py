"""Evalua reglas RIPPER por precision de su propia clase y selecciona las mejores.

Uso desde backend:
    python limpieza/resultados_limpieza/reglas_unicas_ripper_por_iteracion/seleccion_mejores_por_precision/evaluar_y_seleccionar_por_precision.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import skfuzzy as fuzz
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from src.riesgo_materno.entrenamiento.datos import cargar_dataset, convertir_split_a_diccionario
from src.riesgo_materno.entrenamiento.modelo import RUTA_CSV
from src.riesgo_materno.logica_difusa.motor import SistemaDifusoMamdani
from src.riesgo_materno.logica_difusa.variables import (
    ESPECIFICACIONES_VARIABLES,
    PUNTOS_GRAFICA,
    VARIABLES_ENTRADA,
)


RAIZ_SCRIPT = Path(__file__).resolve().parent
RUTA_REGLAS_UNICAS_DEFAULT = RAIZ_SCRIPT.parent
RUTA_SALIDA_DEFAULT = RAIZ_SCRIPT / "resultados"

CLASES = ["low risk", "mid risk", "high risk"]
CLASE_A_CONSECUENTE = {"low risk": "bajo", "mid risk": "medio", "high risk": "alto"}
CONSECUENTE_A_CLASE = {v: k for k, v in CLASE_A_CONSECUENTE.items()}
REGLAS_POR_CLASE_DEFAULT = {
    "low risk": 123,
    "mid risk": 123,
    "high risk": 122,
}


def cargar_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def guardar_json(ruta: Path, contenido: dict | list) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(serializar(contenido), indent=2, ensure_ascii=False), encoding="utf-8")


def guardar_csv(ruta: Path, filas: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    columnas = list(filas[0].keys()) if filas else []
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
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


def construir_membresias_base() -> dict:
    membresias = {}
    for variable, especificacion in ESPECIFICACIONES_VARIABLES.items():
        membresias[variable] = {
            categoria: np.asarray(puntos, dtype=float)
            for categoria, puntos in especificacion["categorias"].items()
        }
    return membresias


def calcular_pertenencias_lote(entradas: dict, membresias: dict) -> dict:
    pertenencias = {}
    for variable in VARIABLES_ENTRADA:
        minimo, maximo = ESPECIFICACIONES_VARIABLES[variable]["limites"]
        universo = np.linspace(minimo, maximo, PUNTOS_GRAFICA)
        valores = np.asarray(entradas[variable], dtype=float)
        pertenencias[variable] = {}
        for categoria, puntos in membresias[variable].items():
            curva = fuzz.trapmf(universo, puntos)
            pertenencias[variable][categoria] = np.asarray(
                [fuzz.interp_membership(universo, curva, valor) for valor in valores],
                dtype=float,
            )
    return pertenencias


def iteraciones_disponibles(ruta_reglas: Path, max_iteraciones: int) -> list[Path]:
    rutas = []
    for numero in range(1, max_iteraciones + 1):
        ruta = ruta_reglas / f"iteracion_{numero:02d}" / "ripper_reglas_unicas.json"
        if ruta.exists():
            rutas.append(ruta)
    return rutas


def regla_a_motor(regla: dict, numero: int) -> dict:
    return {
        "numero": numero,
        "antecedentes": [
            (antecedente["variable"], antecedente["etiqueta_linguistica"])
            for antecedente in regla["antecedentes"]
        ],
        "consecuente": CLASE_A_CONSECUENTE[regla["consecuente"]],
        "source": regla.get("origen", "RIPPER"),
    }


def clave_regla_motor(regla: dict) -> tuple:
    return tuple(regla["antecedentes"]), regla["consecuente"]


def evaluar_regla(regla_motor: dict, pertenencias: dict, riesgos_reales: np.ndarray, totales_clase: dict) -> dict:
    activacion = np.ones(len(riesgos_reales), dtype=float)
    for variable, categoria in regla_motor["antecedentes"]:
        activacion = np.minimum(activacion, pertenencias[variable][categoria])

    clase = CONSECUENTE_A_CLASE[regla_motor["consecuente"]]
    cubiertos = activacion > 0.0
    cobertura = int(np.sum(cubiertos))
    aciertos = int(np.sum(cubiertos & (riesgos_reales == clase)))
    errores = int(cobertura - aciertos)
    precision = aciertos / cobertura if cobertura else 0.0
    recall = aciertos / totales_clase[clase] if totales_clase[clase] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "clase": clase,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "cobertura": cobertura,
        "aciertos": aciertos,
        "errores": errores,
    }


def cargar_y_evaluar_reglas(ruta_reglas: Path, pertenencias: dict, riesgos_reales: np.ndarray, max_iteraciones: int) -> list[dict]:
    totales_clase = {clase: int(np.sum(riesgos_reales == clase)) for clase in CLASES}
    evaluadas = []
    vistas_globales = set()

    print("Cargando reglas RIPPER limpias por iteracion...")
    for ruta_iteracion in iteraciones_disponibles(ruta_reglas, max_iteraciones):
        resultado = cargar_json(ruta_iteracion)
        iteracion = int(resultado.get("iteracion", ruta_iteracion.parent.name[-2:]))
        reglas = resultado.get("reglas_finales", [])
        print(f"  iteracion_{iteracion:02d}: evaluando {len(reglas)} reglas unicas")

        duplicadas_globales = 0
        for indice, regla in enumerate(reglas, start=1):
            regla_motor = regla_a_motor(regla, numero=len(evaluadas) + 1)
            clave = clave_regla_motor(regla_motor)
            if clave in vistas_globales:
                duplicadas_globales += 1
                continue
            vistas_globales.add(clave)

            metricas = evaluar_regla(regla_motor, pertenencias, riesgos_reales, totales_clase)
            evaluadas.append(
                {
                    "id_global": f"RR{len(evaluadas) + 1:04d}",
                    "iteracion_origen": iteracion,
                    "id_origen": regla.get("id", f"R{indice:03d}"),
                    "regla": regla,
                    "regla_motor": regla_motor,
                    **metricas,
                }
            )

        if duplicadas_globales:
            print(f"    omitidas por duplicado global: {duplicadas_globales}")

    print(f"Total de reglas RIPPER unicas globales evaluadas: {len(evaluadas)}")
    return evaluadas


def construir_ranking(evaluadas: list[dict]) -> dict:
    ranking = {clase: [] for clase in CLASES}
    for fila in evaluadas:
        ranking[fila["clase"]].append(fila)
    for clase in CLASES:
        ranking[clase].sort(
            key=lambda fila: (
                fila["precision"],
                fila["aciertos"],
                fila["cobertura"],
                fila["recall"],
                fila["f1"],
            ),
            reverse=True,
        )
    return ranking


def seleccionar_mejores(ranking: dict, reglas_por_clase: dict) -> list[dict]:
    seleccionadas = []
    faltantes = []
    for clase in CLASES:
        cantidad = reglas_por_clase[clase]
        disponibles = ranking[clase]
        tomar = min(cantidad, len(disponibles))
        print(f"Seleccion {clase}: {tomar}/{cantidad} reglas")
        seleccionadas.extend(disponibles[:tomar])
        if tomar < cantidad:
            faltantes.append((clase, cantidad - tomar))

    if faltantes:
        restantes = [
            fila
            for clase in CLASES
            for fila in ranking[clase]
            if fila not in seleccionadas
        ]
        restantes.sort(
            key=lambda fila: (
                fila["precision"],
                fila["aciertos"],
                fila["cobertura"],
                fila["recall"],
                fila["f1"],
            ),
            reverse=True,
        )
        deficit = sum(cantidad for _, cantidad in faltantes)
        print(f"Completando {deficit} faltantes con mejores reglas restantes de cualquier clase")
        seleccionadas.extend(restantes[:deficit])
    return seleccionadas


def reglas_finales_formato_comun(seleccionadas: list[dict]) -> list[dict]:
    reglas = []
    for indice, fila in enumerate(seleccionadas, start=1):
        regla = dict(fila["regla"])
        regla["id"] = f"R{indice:03d}"
        regla["seleccion"] = {
            "id_global": fila["id_global"],
            "iteracion_origen": fila["iteracion_origen"],
            "id_origen": fila["id_origen"],
            "precision": fila["precision"],
            "recall": fila["recall"],
            "f1": fila["f1"],
            "cobertura": fila["cobertura"],
            "aciertos": fila["aciertos"],
            "errores": fila["errores"],
        }
        reglas.append(regla)
    return reglas


def evaluar_conjunto_final(seleccionadas: list[dict], tabla, membresias: dict) -> dict:
    datos = convertir_split_a_diccionario(tabla)
    reglas_motor = []
    for indice, fila in enumerate(seleccionadas, start=1):
        regla = dict(fila["regla_motor"])
        regla["numero"] = indice
        reglas_motor.append(regla)

    sistema = SistemaDifusoMamdani(membresias, reglas=reglas_motor)
    predichos = sistema.inferir_lote(datos["entradas"])["riesgos"]
    reales = datos["riesgos"]
    accuracy = float(accuracy_score(reales, predichos))
    balanced_accuracy = float(balanced_accuracy_score(reales, predichos))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "error_clasificacion": float(1.0 - accuracy),
        "error_balanceado": float(1.0 - balanced_accuracy),
        "matriz_confusion": {
            "etiquetas": CLASES,
            "matriz": confusion_matrix(reales, predichos, labels=CLASES).tolist(),
        },
    }


def filas_ranking_csv(evaluadas: list[dict]) -> list[dict]:
    return [
        {
            "id_global": fila["id_global"],
            "iteracion_origen": fila["iteracion_origen"],
            "id_origen": fila["id_origen"],
            "clase": fila["clase"],
            "precision": fila["precision"],
            "recall": fila["recall"],
            "f1": fila["f1"],
            "cobertura": fila["cobertura"],
            "aciertos": fila["aciertos"],
            "errores": fila["errores"],
            "antecedentes": " AND ".join(
                f"{variable}={categoria}"
                for variable, categoria in fila["regla_motor"]["antecedentes"]
            ),
        }
        for fila in evaluadas
    ]


def resumen_por_clase(seleccionadas: list[dict]) -> dict:
    conteo = Counter(fila["clase"] for fila in seleccionadas)
    return {clase: int(conteo.get(clase, 0)) for clase in CLASES}


def ejecutar(ruta_reglas: Path, ruta_salida: Path, max_iteraciones: int, reglas_por_clase: dict) -> dict:
    ruta_salida.mkdir(parents=True, exist_ok=True)
    print("Cargando dataset...")
    tabla = cargar_dataset(RUTA_CSV)
    datos = convertir_split_a_diccionario(tabla)
    riesgos_reales = np.asarray(datos["riesgos"])
    membresias = construir_membresias_base()

    print("Precalculando pertenencias difusas para acelerar la evaluacion...")
    pertenencias = calcular_pertenencias_lote(datos["entradas"], membresias)
    evaluadas = cargar_y_evaluar_reglas(ruta_reglas, pertenencias, riesgos_reales, max_iteraciones)

    print("Construyendo ranking RIPPER por precision dentro de cada clase...")
    ranking = construir_ranking(evaluadas)
    for clase in CLASES:
        print(f"Top {clase}:")
        for fila in ranking[clase][:3]:
            print(
                f"  {fila['id_global']} iter={fila['iteracion_origen']:02d} "
                f"precision={fila['precision']:.4f} aciertos={fila['aciertos']} "
                f"cobertura={fila['cobertura']} recall={fila['recall']:.4f}"
            )

    seleccionadas = seleccionar_mejores(ranking, reglas_por_clase)
    print("Evaluando conjunto final RIPPER con Mamdani...")
    metricas_finales = evaluar_conjunto_final(seleccionadas, tabla, membresias)
    print(
        "Resultado final RIPPER | "
        f"reglas={len(seleccionadas)} | "
        f"accuracy={metricas_finales['accuracy']:.4f} | "
        f"balanced_accuracy={metricas_finales['balanced_accuracy']:.4f}"
    )

    evaluadas_ordenadas = sorted(
        evaluadas,
        key=lambda fila: (
            fila["clase"],
            -fila["precision"],
            -fila["aciertos"],
            -fila["cobertura"],
        ),
    )
    guardar_csv(ruta_salida / "ranking_reglas_ripper_por_precision.csv", filas_ranking_csv(evaluadas_ordenadas))

    resultado_final = {
        "descripcion": "Conjunto de reglas RIPPER seleccionado por precision dentro de cada clase.",
        "criterio_principal": "precision",
        "reglas_por_clase_objetivo": reglas_por_clase,
        "reglas_por_clase_final": resumen_por_clase(seleccionadas),
        "total_reglas": len(seleccionadas),
        "metricas_finales": metricas_finales,
        "reglas_finales": reglas_finales_formato_comun(seleccionadas),
    }
    guardar_json(ruta_salida / "mejores_ripper_por_precision.json", resultado_final)

    resumen = {
        "ruta_reglas_unicas": str(ruta_reglas),
        "ruta_salida": str(ruta_salida),
        "total_reglas_evaluadas": len(evaluadas),
        "reglas_por_clase_objetivo": reglas_por_clase,
        "reglas_por_clase_final": resumen_por_clase(seleccionadas),
        "metricas_finales": metricas_finales,
        "archivos": {
            "ranking_csv": str(ruta_salida / "ranking_reglas_ripper_por_precision.csv"),
            "conjunto_final": str(ruta_salida / "mejores_ripper_por_precision.json"),
        },
    }
    guardar_json(ruta_salida / "resumen_seleccion_ripper_por_precision.json", resumen)
    return resumen


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rankea reglas RIPPER por precision de su propia clase y arma un conjunto final."
    )
    parser.add_argument("--reglas-unicas", type=Path, default=RUTA_REGLAS_UNICAS_DEFAULT)
    parser.add_argument("--salida", type=Path, default=RUTA_SALIDA_DEFAULT)
    parser.add_argument("--iteraciones", type=int, default=20)
    parser.add_argument(
        "--low",
        type=int,
        default=REGLAS_POR_CLASE_DEFAULT["low risk"],
        help="Cantidad de reglas low risk a seleccionar.",
    )
    parser.add_argument(
        "--mid",
        type=int,
        default=REGLAS_POR_CLASE_DEFAULT["mid risk"],
        help="Cantidad de reglas mid risk a seleccionar.",
    )
    parser.add_argument(
        "--high",
        type=int,
        default=REGLAS_POR_CLASE_DEFAULT["high risk"],
        help="Cantidad de reglas high risk a seleccionar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parsear_argumentos()
    ejecutar(
        ruta_reglas=args.reglas_unicas,
        ruta_salida=args.salida,
        max_iteraciones=args.iteraciones,
        reglas_por_clase={
            "low risk": args.low,
            "mid risk": args.mid,
            "high risk": args.high,
        },
    )


if __name__ == "__main__":
    main()
