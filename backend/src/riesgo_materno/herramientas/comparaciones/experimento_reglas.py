"""Comparacion experimental de reglas RIPPER, PRISM y GA.

Uso rapido:
    python -m src.riesgo_materno.herramientas.comparaciones.experimento_reglas

Los parametros por defecto son pequenos para validar el flujo. Para corridas
finales, ajustar CONFIGURACION_EXPERIMENTO.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ...entrenamiento.datos import (
    cargar_dataset,
    convertir_split_a_diccionario,
    dividir_entrenamiento_prueba,
)
from ...entrenamiento.modelo import RUTA_CSV
from ...entrenamiento.prism import aprender_reglas_prism as aprender_prism
from ...entrenamiento.prism import indices_cubiertos
from ...entrenamiento.ripper import _discretizar
from ...entrenamiento.ripper import aprender_reglas_ripper as aprender_ripper
from ...logica_difusa.motor import SistemaDifusoMamdani
from ...logica_difusa.variables import (
    ESPECIFICACIONES_VARIABLES,
    ETIQUETAS_RIESGO,
    VARIABLES_ENTRADA,
)
from ...optimizacion.selector_reglas_pygad import ejecutar_selector_reglas_pygad


RUTA_BASE = Path(__file__).resolve().parent
RUTA_RESULTADOS = RUTA_BASE / "resultados"

CLASES = ["low risk", "mid risk", "high risk"]
CLASE_A_CONSECUENTE = {"low risk": "bajo", "mid risk": "medio", "high risk": "alto"}
CONSECUENTE_A_CLASE = {v: k for k, v in CLASE_A_CONSECUENTE.items()}

CONFIGURACION_EXPERIMENTO = {
    "experiment_id": "maternal_risk_rule_comparison",
    "iterations": 3,
    "random_seeds": [1, 2, 3],
    "train_size": 0.70,
    "test_size": 0.30,
    "classes": CLASES,
    "primary_metric": "balanced_accuracy",
    "ripper": {
        "k": 2,
        "dl_allowance": 64,
    },
    "prism": {
        "min_rule_coverage": 2,
        "class_order": CLASES,
        "max_conditions_per_rule": 3,
        "max_rules_per_class": 25,
        "remove_covered_positives": False,
    },
    "ga": {
        "candidate_rules": 30,
        "population_size": 20,
        "max_generations": 100,
        "patience": 50,
        "mutation_rate": 0.10,
        "crossover_rate": 0.90,
        "elitism_count": 2,
        "selection_method": "roulette",
        "lambda_penalty": 0.05,
        "max_conditions_per_rule": 6,
        "min_rule_coverage": 2,
    },
}


def principal():
    ejecutar_experimento(CONFIGURACION_EXPERIMENTO)


def ejecutar_experimento(config):
    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    guardar_json(RUTA_RESULTADOS / "config_experimento.json", config)

    tabla = cargar_dataset(RUTA_CSV)
    resultados = []

    for iteracion, semilla in enumerate(config["random_seeds"][: config["iterations"]], start=1):
        print(f"\nIteracion {iteracion:02d} | seed={semilla}")
        splits = dividir_entrenamiento_prueba(tabla, semilla=semilla)
        ruta_iteracion = RUTA_RESULTADOS / f"iteracion_{iteracion:02d}"
        ruta_iteracion.mkdir(parents=True, exist_ok=True)

        resultados.append(
        ejecutar_ripper(iteracion, semilla, splits, ruta_iteracion, config)
        )
        resultados.append(
            ejecutar_prism(iteracion, semilla, splits, ruta_iteracion, config)
        )
        resultados.append(
            ejecutar_ga(iteracion, semilla, splits, ruta_iteracion, config)
        )

    resumen = construir_resumen_final(resultados)
    guardar_csv_seguro(pd.DataFrame(resumen["tabla"]), RUTA_RESULTADOS / "resumen_final.csv")
    guardar_json_seguro(RUTA_RESULTADOS / "resumen_final.json", resumen)
    print(f"\nResultados guardados en: {RUTA_RESULTADOS}")
    return resumen


def ejecutar_ripper(iteracion, semilla, splits, ruta_iteracion, config):
    inicio = time.perf_counter()
    reglas = aprender_ripper(splits["entrenamiento"], orden_clases=CLASES, parametros=config["ripper"])
    asignar_source(reglas, "RIPPER")
    training_ms = medir_ms(inicio)
    resultado = evaluar_y_guardar(
        iteracion=iteracion,
        semilla=semilla,
        algoritmo="RIPPER",
        reglas=reglas,
        splits=splits,
        ruta=ruta_iteracion / "ripper.json",
        hyperparameters=config["ripper"],
        training_time_ms=training_ms,
        extra_rules={"generated_rules": reglas},
        extra=None,
        fitness_reference_rules=config["ga"]["candidate_rules"],
        lambda_penalty=config["ga"]["lambda_penalty"],
    )
    print(f"  RIPPER | reglas={len(reglas)} | BA test={resultado['metrics_test']['balanced_accuracy']:.4f}")
    return resultado


def ejecutar_prism(iteracion, semilla, splits, ruta_iteracion, config):
    inicio = time.perf_counter()
    reglas = aprender_prism(splits["entrenamiento"], config["prism"])
    asignar_source(reglas, "PRISM")
    training_ms = medir_ms(inicio)
    resultado = evaluar_y_guardar(
        iteracion=iteracion,
        semilla=semilla,
        algoritmo="PRISM",
        reglas=reglas,
        splits=splits,
        ruta=ruta_iteracion / "prism.json",
        hyperparameters=config["prism"],
        training_time_ms=training_ms,
        extra_rules={"generated_rules": reglas},
        extra=None,
        fitness_reference_rules=config["ga"]["candidate_rules"],
        lambda_penalty=config["ga"]["lambda_penalty"],
    )
    print(f"  PRISM  | reglas={len(reglas)} | BA test={resultado['metrics_test']['balanced_accuracy']:.4f}")
    return resultado


def ejecutar_ga(iteracion, semilla, splits, ruta_iteracion, config):
    ga_config = config["ga"]
    inicio = time.perf_counter()
    print("  GA     | generando y rankeando reglas candidatas...")
    candidatas = generar_reglas_candidatas_ga(splits["entrenamiento"], ga_config)
    print(f"  GA     | candidatas seleccionadas={len(candidatas)}")
    mejor, historial = ejecutar_selector_reglas_pygad(
        reglas_candidatas=candidatas,
        datos_entrenamiento=convertir_split_a_diccionario(splits["entrenamiento"]),
        membresias=construir_membresias_base(),
        parametros=ga_config,
    )
    reglas_activas = [regla for bit, regla in zip(mejor.cromosoma, candidatas) if int(bit) == 1]
    training_ms = medir_ms(inicio)

    extra = {
        "ga_history": resumir_historial_ga(historial),
    }
    resultado = evaluar_y_guardar(
        iteracion=iteracion,
        semilla=semilla,
        algoritmo="GENETIC_ALGORITHM",
        reglas=reglas_activas,
        splits=splits,
        ruta=ruta_iteracion / "genetic_algorithm.json",
        hyperparameters=ga_config,
        training_time_ms=training_ms,
        extra_rules={
            "candidate_rules": candidatas,
            "active_rules": reglas_activas,
            "best_chromosome": [int(x) for x in mejor.cromosoma.tolist()],
        },
        extra=extra,
        num_rules_initial=len(candidatas),
        fitness_reference_rules=len(candidatas),
        lambda_penalty=ga_config["lambda_penalty"],
    )
    guardar_json(ruta_iteracion / "genetic_algorithm.json", serializar_resultado(resultado))
    print(
        "  GA     | candidatas="
        f"{len(candidatas)} | activas={len(reglas_activas)} | "
        f"BA test={resultado['metrics_test']['balanced_accuracy']:.4f}"
    )
    return resultado


def generar_reglas_candidatas_ga(tabla, config):
    df = pd.DataFrame(_discretizar(tabla)).rename(columns={"clase": "riesgo"})
    candidatas = []
    total_evaluadas = 0
    numero = 1
    max_largo = config["max_conditions_per_rule"]

    terminos_por_variable = [
        [(variable, categoria) for categoria in ESPECIFICACIONES_VARIABLES[variable]["categorias"]]
        for variable in VARIABLES_ENTRADA
    ]

    for largo in range(1, max_largo + 1):
        for variables_indices in itertools.combinations(range(len(VARIABLES_ENTRADA)), largo):
            grupos = [terminos_por_variable[i] for i in variables_indices]
            for antecedentes in itertools.product(*grupos):
                antecedentes = list(antecedentes)
                cubiertos = indices_cubiertos(df, antecedentes)
                if len(cubiertos) < config["min_rule_coverage"]:
                    continue
                for clase in CLASES:
                    total_evaluadas += 1
                    metricas = calidad_regla(df, cubiertos, clase)
                    if metricas["positive_coverage"] == 0:
                        continue
                    regla = crear_regla(numero, antecedentes, clase, "GA")
                    regla["_rank"] = metricas
                    candidatas.append(regla)
                    numero += 1

    print(
        "  GA     | reglas evaluadas="
        f"{total_evaluadas} | reglas validas={len(candidatas)}"
    )
    candidatas.sort(
        key=lambda r: (
            r["_rank"]["precision"],
            r["_rank"]["positive_coverage"],
            r["_rank"]["coverage"],
            -len(r["antecedentes"]),
        ),
        reverse=True,
    )
    candidatas = balancear_candidatas_por_clase(candidatas, config["candidate_rules"])
    for i, regla in enumerate(candidatas, start=1):
        regla["numero"] = i
        regla["candidate_index"] = i
        regla.pop("_rank", None)
    return candidatas


def balancear_candidatas_por_clase(candidatas, limite):
    por_clase = {clase: [] for clase in CLASES}
    for regla in candidatas:
        por_clase[CONSECUENTE_A_CLASE[regla["consecuente"]]].append(regla)
    seleccionadas = []
    while len(seleccionadas) < limite:
        agregado = False
        for clase in CLASES:
            if por_clase[clase]:
                seleccionadas.append(por_clase[clase].pop(0))
                agregado = True
                if len(seleccionadas) == limite:
                    break
        if not agregado:
            break
    return seleccionadas


def evaluar_y_guardar(
    iteracion,
    semilla,
    algoritmo,
    reglas,
    splits,
    ruta,
    hyperparameters,
    training_time_ms,
    extra_rules,
    extra,
    num_rules_initial=None,
    fitness_reference_rules=None,
    lambda_penalty=None,
):
    inicio_train = time.perf_counter()
    metricas_train = evaluar_reglas(reglas, splits["entrenamiento"])
    infer_train_ms = medir_ms(inicio_train)
    inicio_test = time.perf_counter()
    metricas_test = evaluar_reglas(reglas, splits["prueba"])
    infer_test_ms = medir_ms(inicio_test)
    if fitness_reference_rules is not None and lambda_penalty is not None:
        metricas_train["fitness_penalized"] = fitness_penalizado(
            metricas_train["balanced_accuracy"],
            len(reglas),
            fitness_reference_rules,
            lambda_penalty,
        )
        metricas_test["fitness_penalized"] = fitness_penalizado(
            metricas_test["balanced_accuracy"],
            len(reglas),
            fitness_reference_rules,
            lambda_penalty,
        )

    resultado = {
        "experiment_id": CONFIGURACION_EXPERIMENTO["experiment_id"],
        "iteration": iteracion,
        "seed": semilla,
        "algorithm": algoritmo,
        "split": {
            "strategy": "stratified_holdout",
            "train_ratio": 0.70,
            "test_ratio": 0.30,
            "train_size": int(len(splits["entrenamiento"])),
            "test_size": int(len(splits["prueba"])),
        },
        "hyperparameters": hyperparameters,
        "rules_summary": resumir_reglas(reglas, num_rules_initial),
        "rules": serializar_reglas_payload(extra_rules),
        "metrics_train": metricas_train,
        "metrics_test": metricas_test,
        "metrics_per_class_test": metricas_test.pop("per_class"),
        "confusion_matrix_test": metricas_test.pop("confusion_matrix"),
        "runtime": {
            "training_time_ms": int(training_time_ms),
            "inference_time_train_ms": int(infer_train_ms),
            "inference_time_test_ms": int(infer_test_ms),
            "total_time_ms": int(training_time_ms + infer_train_ms + infer_test_ms),
        },
    }
    if extra:
        resultado.update(extra)
    guardar_json(ruta, serializar_resultado(resultado))
    return resultado


def evaluar_reglas(reglas, tabla):
    datos = convertir_split_a_diccionario(tabla)
    membresias = construir_membresias_base()
    if not reglas:
        predichos = np.array([clase_mayoritaria(tabla)] * len(tabla), dtype=object)
        sin_activacion = np.ones(len(tabla), dtype=bool)
    else:
        sistema = SistemaDifusoMamdani(membresias, reglas=reglas)
        inferencia = sistema.inferir_lote(datos["entradas"])
        predichos = inferencia["riesgos"]
        sin_activacion = inferencia["sin_activacion"]
    reales = datos["riesgos"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        reales, predichos, labels=CLASES, zero_division=0
    )
    per_class = {
        clase: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }
        for i, clase in enumerate(CLASES)
    }
    matriz = confusion_matrix(reales, predichos, labels=CLASES).tolist()
    return {
        "balanced_accuracy": float(balanced_accuracy_score(reales, predichos)),
        "fitness_penalized": None,
        "accuracy": float(accuracy_score(reales, predichos)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "coverage": float(1.0 - np.mean(sin_activacion)),
        "uncovered_instances": int(np.sum(sin_activacion)),
        "per_class": per_class,
        "confusion_matrix": {"labels": CLASES, "matrix": matriz},
    }


def construir_membresias_base():
    membresias = {}
    for variable, especificacion in ESPECIFICACIONES_VARIABLES.items():
        membresias[variable] = {
            categoria: np.asarray(puntos, dtype=float)
            for categoria, puntos in especificacion["categorias"].items()
        }
    return membresias


def resumir_reglas(reglas, num_rules_initial=None):
    inicial = int(num_rules_initial if num_rules_initial is not None else len(reglas))
    activas = len(reglas)
    total_antecedentes = sum(len(r["antecedentes"]) for r in reglas)
    por_clase = {clase: 0 for clase in CLASES}
    for regla in reglas:
        por_clase[CONSECUENTE_A_CLASE[regla["consecuente"]]] += 1
    return {
        "num_rules_initial": inicial,
        "num_rules_generated": inicial,
        "num_rules_active": activas,
        "num_rules_inactive": max(0, inicial - activas),
        "rule_reduction_ratio": float((inicial - activas) / inicial) if inicial else 0.0,
        "avg_rule_length": float(total_antecedentes / activas) if activas else 0.0,
        "total_antecedents": int(total_antecedentes),
        "rules_by_class": por_clase,
        "active_rules_by_class": por_clase,
    }


def construir_resumen_final(resultados):
    filas = []
    for algoritmo in sorted({r["algorithm"] for r in resultados}):
        grupo = [r for r in resultados if r["algorithm"] == algoritmo]
        ba = [r["metrics_test"]["balanced_accuracy"] for r in grupo]
        fitness = [
            r["metrics_test"]["fitness_penalized"]
            for r in grupo
            if r["metrics_test"]["fitness_penalized"] is not None
        ]
        filas.append(
            {
                "algorithm": algoritmo,
                "ba_test_mean": float(np.mean(ba)),
                "ba_test_std": float(np.std(ba, ddof=1)) if len(ba) > 1 else 0.0,
                "fitness_test_mean": float(np.mean(fitness)) if fitness else None,
                "f1_macro_mean": float(np.mean([r["metrics_test"]["f1_macro"] for r in grupo])),
                "active_rules_mean": float(np.mean([r["rules_summary"]["num_rules_active"] for r in grupo])),
                "coverage_mean": float(np.mean([r["metrics_test"]["coverage"] for r in grupo])),
                "runtime_ms_mean": float(np.mean([r["runtime"]["total_time_ms"] for r in grupo])),
            }
        )
    return {
        "tabla": filas,
        "best_iterations": mejores_iteraciones(resultados, max),
        "worst_iterations": mejores_iteraciones(resultados, min),
    }


def mejores_iteraciones(resultados, funcion):
    salida = {}
    for algoritmo in sorted({r["algorithm"] for r in resultados}):
        grupo = [r for r in resultados if r["algorithm"] == algoritmo]
        elegido = funcion(grupo, key=lambda r: r["metrics_test"]["balanced_accuracy"])
        salida[algoritmo] = {
            "iteration": elegido["iteration"],
            "seed": elegido["seed"],
            "balanced_accuracy_test": elegido["metrics_test"]["balanced_accuracy"],
        }
    return salida


def calidad_regla(df, cubiertos, clase):
    positivos = [i for i in cubiertos if df.at[i, "riesgo"] == clase]
    return {
        "coverage": len(cubiertos),
        "positive_coverage": len(positivos),
        "precision": len(positivos) / len(cubiertos) if cubiertos else 0.0,
    }


def crear_regla(numero, antecedentes, clase, source):
    return {
        "numero": int(numero),
        "antecedentes": [(str(v), str(c)) for v, c in antecedentes],
        "consecuente": CLASE_A_CONSECUENTE[clase],
        "source": source,
        "active": True,
    }


def asignar_source(reglas, source):
    for regla in reglas:
        regla["source"] = source
        regla["active"] = True


def serializar_resultado(valor):
    if isinstance(valor, dict):
        return {k: serializar_resultado(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [serializar_resultado(v) for v in valor]
    if isinstance(valor, tuple):
        return [serializar_resultado(v) for v in valor]
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if hasattr(valor, "item"):
        return valor.item()
    return valor


def serializar_reglas_payload(payload):
    salida = {}
    for clave, valor in payload.items():
        if isinstance(valor, list) and valor and isinstance(valor[0], dict) and "antecedentes" in valor[0]:
            salida[clave] = [regla_a_formato_comun(regla, i) for i, regla in enumerate(valor, start=1)]
        else:
            salida[clave] = valor
    return salida


def regla_a_formato_comun(regla, posicion):
    comun = {
        "id": f"R{posicion:03d}",
        "antecedent": [
            {"variable": variable, "linguistic_term": termino}
            for variable, termino in regla["antecedentes"]
        ],
        "consequent": CONSECUENTE_A_CLASE[regla["consecuente"]],
        "source": regla.get("source"),
        "active": bool(regla.get("active", True)),
    }
    if "candidate_index" in regla:
        comun["candidate_index"] = int(regla["candidate_index"])
    return comun


def guardar_json(ruta, contenido):
    ruta.write_text(json.dumps(serializar_resultado(contenido), indent=2, ensure_ascii=False), encoding="utf-8")


def guardar_json_seguro(ruta, contenido):
    try:
        guardar_json(ruta, contenido)
    except PermissionError:
        alternativa = ruta_con_timestamp(ruta)
        guardar_json(alternativa, contenido)
        print(f"  Aviso | {ruta.name} esta bloqueado; se guardo {alternativa.name}")


def guardar_csv_seguro(dataframe, ruta):
    try:
        dataframe.to_csv(ruta, index=False)
    except PermissionError:
        alternativa = ruta_con_timestamp(ruta)
        dataframe.to_csv(alternativa, index=False)
        print(f"  Aviso | {ruta.name} esta bloqueado; se guardo {alternativa.name}")


def ruta_con_timestamp(ruta):
    marca = time.strftime("%Y%m%d_%H%M%S")
    return ruta.with_name(f"{ruta.stem}_{marca}{ruta.suffix}")


def fitness_penalizado(ba, activas, total, lambda_penalty):
    if total == 0:
        return 0.0
    return float(ba - lambda_penalty * (activas / total))


def resumir_historial_ga(historial):
    if isinstance(historial, pd.DataFrame):
        filas = historial.to_dict(orient="records")
    else:
        filas = historial
    if not filas:
        return {
            "best_generation": 0,
            "final_generation": 0,
            "stopped_by": "unknown",
            "initial_best_fitness": 0.0,
            "final_best_fitness": 0.0,
            "history": [],
        }
    mejor = max(filas, key=lambda h: h["best_fitness"])
    return {
        "best_generation": int(mejor["generation"]),
        "final_generation": int(filas[-1]["generation"]),
        "stopped_by": "patience" if filas[-1]["generation"] < CONFIGURACION_EXPERIMENTO["ga"]["max_generations"] else "max_generations",
        "initial_best_fitness": float(filas[0]["best_fitness"]),
        "final_best_fitness": float(filas[-1]["best_fitness"]),
        "history": filas,
    }


def clase_mayoritaria(tabla):
    return str(tabla["riesgo"].mode().iloc[0])


def medir_ms(inicio):
    return int((time.perf_counter() - inicio) * 1000)


if __name__ == "__main__":
    principal()
