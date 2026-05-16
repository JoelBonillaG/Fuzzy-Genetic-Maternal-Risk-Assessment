"""Comparacion experimental de RIPPER, PRISM y AG Pittsburgh-Michigan.

El experimento usa particion 70/30 estratificada. Los algoritmos aprenden solo
con entrenamiento y las reglas finales se reportan en entrenamiento y prueba.

Uso:
    python -m riesgo_materno.herramientas.comparaciones.experimento_reglas
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from ...entrenamiento.datos import (
    cargar_dataset,
    convertir_split_a_diccionario,
    dividir_entrenamiento_prueba,
    resumir_splits,
)
from ...entrenamiento.modelo import RUTA_CSV
from ...entrenamiento.prism import aprender_reglas_prism as aprender_prism
from ...entrenamiento.ripper import aprender_reglas_ripper as aprender_ripper
from ...logica_difusa.motor import SistemaDifusoMamdani
from ...logica_difusa.variables import ESPECIFICACIONES_VARIABLES
from ...optimizacion.pittsburgh_michigan import (
    BITS_POR_CAMPO,
    BITS_POR_REGLA,
    CAMPOS_POR_REGLA,
    contar_duplicados,
    ejecutar_ag_pittsburgh_michigan,
)


RUTA_BASE = Path(__file__).resolve().parent
RUTA_RESULTADOS = RUTA_BASE / "resultados"

CLASES = ["low risk", "mid risk", "high risk"]
CLASE_SIN_ACTIVACION = "__sin_activacion__"
CLASE_A_CONSECUENTE = {"low risk": "bajo", "mid risk": "medio", "high risk": "alto"}
CONSECUENTE_A_CLASE = {v: k for k, v in CLASE_A_CONSECUENTE.items()}

CONFIGURACION_EXPERIMENTO = {
    "id_experimento": "comparacion_reglas_riesgo_materno_split_70_30",
    "iteraciones": 5,
    "clases": CLASES,
    "estrategia_datos": "split_70_30_estratificado",
    "proporcion_entrenamiento": 0.70,
    "semilla_base": 42,
    "metrica_principal": "balanced_accuracy",
    "fitness": {
        "peso_balanced_accuracy": 0.98,
        "penalizacion_duplicados": 0.02,
    },
    "ripper": {
        "k": 2,
        "tolerancia_longitud_descripcion": 64,
    },
    "prism": {
        "modo": "prism_bootstrap",
        "fraccion_bootstrap": 1.0,
        "cobertura_minima_regla": 2,
        "orden_clases": CLASES,
        "maximo_condiciones_por_regla": 6,
        "maximo_reglas_por_clase": 20,
        "eliminar_positivos_cubiertos": True,
    },
    "ag_pittsburgh_michigan": {
        "reglas_por_individuo": 30,
        "tamano_poblacion": 30,
        "cantidad_padres": 15,
        "maximo_generaciones": 180,
        "paciencia": 60,
        "probabilidad_cruce": 0.90,
        "probabilidad_mutacion": 0.12,
        "probabilidad_reemplazo": 0.85,
        "fraccion_reemplazo": 0.10,
        "elitismo": 3,
        "tamano_torneo": 3,
        "peso_balanced_accuracy": 0.98,
        "penalizacion_duplicados": 0.02,
    },
}

def principal():
    ejecutar_experimento(CONFIGURACION_EXPERIMENTO)


def ejecutar_experimento(config):
    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    guardar_json(RUTA_RESULTADOS / "config_experimento.json", config)

    tabla = cargar_dataset(RUTA_CSV)
    resultados = []

    print("Dataset con split 70/30 estratificado")
    print(f"  Instancias totales: {len(tabla)}")
    print("  Estrategia: entrenamiento 70% / prueba 30%")

    for iteracion in range(1, config["iteraciones"] + 1):
        print(f"\nIteracion {iteracion:02d}")
        ruta_iteracion = RUTA_RESULTADOS / f"iteracion_{iteracion:02d}"
        ruta_iteracion.mkdir(parents=True, exist_ok=True)
        semilla = int(config["semilla_base"]) + iteracion - 1
        splits = dividir_entrenamiento_prueba(tabla, semilla=semilla)
        resumen_split = resumir_splits(splits)
        print(f"  Split  | semilla={semilla} train={len(splits['entrenamiento'])} test={len(splits['prueba'])}")

        resultados.append(ejecutar_ripper(iteracion, splits, resumen_split, ruta_iteracion, config, semilla))
        resultados.append(ejecutar_prism(iteracion, splits, resumen_split, ruta_iteracion, config, semilla))
        resultados.append(ejecutar_ag(iteracion, splits, resumen_split, ruta_iteracion, config, semilla))

    resumen = construir_resumen_final(resultados)
    guardar_csv_seguro(pd.DataFrame(resumen["tabla_resumen"]), RUTA_RESULTADOS / "resumen_final.csv")
    guardar_json_seguro(RUTA_RESULTADOS / "resumen_final.json", resumen)
    print(f"\nResultados guardados en: {RUTA_RESULTADOS}")
    return resumen


def ejecutar_ripper(iteracion, splits, resumen_split, ruta_iteracion, config, semilla):
    entrenamiento = splits["entrenamiento"]
    prueba = splits["prueba"]
    inicio = time.perf_counter()
    reglas = aprender_ripper(
        entrenamiento,
        orden_clases=CLASES,
        parametros=traducir_parametros_ripper(config["ripper"]),
    )
    asignar_origen(reglas, "RIPPER")
    tiempo_ms = medir_ms(inicio)
    resultado = evaluar_y_guardar(
        iteracion=iteracion,
        algoritmo="RIPPER",
        reglas=reglas,
        tabla_entrenamiento=entrenamiento,
        tabla_prueba=prueba,
        resumen_split=resumen_split,
        ruta=ruta_iteracion / "ripper.json",
        hiperparametros=config["ripper"],
        tiempo_entrenamiento_ms=tiempo_ms,
        config_fitness=config["fitness"],
        semilla=semilla,
    )
    print(
        f"  RIPPER | reglas={len(reglas)} | "
        f"acc_train={resultado['metricas_entrenamiento']['accuracy']:.4f} | "
        f"ba_train={resultado['metricas_entrenamiento']['balanced_accuracy']:.4f} | "
        f"acc_test={resultado['metricas_prueba']['accuracy']:.4f} | "
        f"ba_test={resultado['metricas_prueba']['balanced_accuracy']:.4f} | "
        f"fitness_test={resultado['metricas_prueba']['fitness']:.4f}"
    )
    return resultado


def ejecutar_prism(iteracion, splits, resumen_split, ruta_iteracion, config, semilla):
    entrenamiento = splits["entrenamiento"]
    prueba = splits["prueba"]
    inicio = time.perf_counter()
    reglas = aprender_prism_estocastico(entrenamiento, config["prism"])
    asignar_origen(reglas, "PRISM_ESTOCASTICO")
    tiempo_ms = medir_ms(inicio)
    resultado = evaluar_y_guardar(
        iteracion=iteracion,
        algoritmo="PRISM_ESTOCASTICO",
        reglas=reglas,
        tabla_entrenamiento=entrenamiento,
        tabla_prueba=prueba,
        resumen_split=resumen_split,
        ruta=ruta_iteracion / "prism.json",
        hiperparametros=config["prism"],
        tiempo_entrenamiento_ms=tiempo_ms,
        config_fitness=config["fitness"],
        semilla=semilla,
    )
    print(
        f"  PRISM  | modo=bootstrap | reglas={len(reglas)} | "
        f"acc_train={resultado['metricas_entrenamiento']['accuracy']:.4f} | "
        f"ba_train={resultado['metricas_entrenamiento']['balanced_accuracy']:.4f} | "
        f"acc_test={resultado['metricas_prueba']['accuracy']:.4f} | "
        f"ba_test={resultado['metricas_prueba']['balanced_accuracy']:.4f} | "
        f"fitness_test={resultado['metricas_prueba']['fitness']:.4f}"
    )
    return resultado


def aprender_prism_estocastico(tabla, config):
    """Genera un unico conjunto PRISM desde una muestra bootstrap."""
    if config.get("modo") != "prism_bootstrap":
        return aprender_prism(tabla, config)

    fraccion = float(config["fraccion_bootstrap"])
    cantidad_filas = max(1, int(len(tabla) * fraccion))
    muestra = tabla.sample(n=cantidad_filas, replace=True).reset_index(drop=True)
    return aprender_prism(muestra, config)


def ejecutar_ag(iteracion, splits, resumen_split, ruta_iteracion, config, semilla):
    entrenamiento = splits["entrenamiento"]
    prueba = splits["prueba"]
    inicio = time.perf_counter()
    print("  AG-PM  | evolucionando base completa de reglas...")
    mejor, historial = ejecutar_ag_pittsburgh_michigan(
        tabla=entrenamiento,
        membresias=construir_membresias_base(),
        parametros=config["ag_pittsburgh_michigan"],
    )
    reglas = mejor.reglas
    asignar_origen(reglas, "AG_PITTSBURGH_MICHIGAN")
    tiempo_ms = medir_ms(inicio)
    resultado = evaluar_y_guardar(
        iteracion=iteracion,
        algoritmo="AG_PITTSBURGH_MICHIGAN",
        reglas=reglas,
        tabla_entrenamiento=entrenamiento,
        tabla_prueba=prueba,
        resumen_split=resumen_split,
        ruta=ruta_iteracion / "genetic_algorithm.json",
        hiperparametros=config["ag_pittsburgh_michigan"],
        tiempo_entrenamiento_ms=tiempo_ms,
        config_fitness=config["fitness"],
        semilla=semilla,
        extra={
            "historial_ag": resumir_historial_ag(historial),
            "mejor_individuo_ag": {
                "cromosoma": [int(gene) for gene in mejor.cromosoma.tolist()],
                "longitud_cromosoma": int(len(mejor.cromosoma)),
                "campos_por_regla": CAMPOS_POR_REGLA,
                "bits_por_campo": BITS_POR_CAMPO,
                "bits_por_regla": BITS_POR_REGLA,
                "fitness": mejor.fitness,
                "balanced_accuracy": mejor.balanced_accuracy,
                "duplicados": mejor.duplicados,
                "proporcion_duplicados": mejor.proporcion_duplicados,
            },
        },
    )
    print(
        f"  AG-PM  | reglas={len(reglas)} | "
        f"acc_train={resultado['metricas_entrenamiento']['accuracy']:.4f} | "
        f"ba_train={resultado['metricas_entrenamiento']['balanced_accuracy']:.4f} | "
        f"acc_test={resultado['metricas_prueba']['accuracy']:.4f} | "
        f"ba_test={resultado['metricas_prueba']['balanced_accuracy']:.4f} | "
        f"fitness_test={resultado['metricas_prueba']['fitness']:.4f} | "
        f"duplicados={resultado['resumen_reglas']['reglas_duplicadas']}"
    )
    return resultado


def evaluar_y_guardar(
    iteracion,
    algoritmo,
    reglas,
    tabla_entrenamiento,
    tabla_prueba,
    resumen_split,
    ruta,
    hiperparametros,
    tiempo_entrenamiento_ms,
    config_fitness,
    semilla,
    extra=None,
):
    inicio_inferencia = time.perf_counter()
    metricas_entrenamiento = evaluar_reglas(reglas, tabla_entrenamiento)
    metricas_prueba = evaluar_reglas(reglas, tabla_prueba)
    tiempo_inferencia_ms = medir_ms(inicio_inferencia)
    resumen_reglas = resumir_reglas(reglas)
    agregar_fitness(metricas_entrenamiento, resumen_reglas, config_fitness)
    agregar_fitness(metricas_prueba, resumen_reglas, config_fitness)
    matriz_entrenamiento = metricas_entrenamiento.pop("matriz_confusion")
    matriz_prueba = metricas_prueba.pop("matriz_confusion")
    resultado = {
        "id_experimento": CONFIGURACION_EXPERIMENTO["id_experimento"],
        "iteracion": int(iteracion),
        "algoritmo": algoritmo,
        "datos": {
            "estrategia": CONFIGURACION_EXPERIMENTO["estrategia_datos"],
            "semilla_split": int(semilla),
            "total_instancias": int(len(tabla_entrenamiento) + len(tabla_prueba)),
            "instancias_entrenamiento": int(len(tabla_entrenamiento)),
            "instancias_prueba": int(len(tabla_prueba)),
            "resumen_splits": resumen_split.to_dict(orient="records"),
        },
        "hiperparametros": hiperparametros,
        "resumen_reglas": resumen_reglas,
        "metricas": metricas_prueba,
        "metricas_entrenamiento": metricas_entrenamiento,
        "metricas_prueba": metricas_prueba,
        "matriz_confusion_entrenamiento": matriz_entrenamiento,
        "matriz_confusion_prueba": matriz_prueba,
        "matriz_confusion": matriz_prueba,
        "reglas_finales": [regla_a_formato_comun(regla, i) for i, regla in enumerate(reglas, start=1)],
        "tiempos": {
            "entrenamiento_ms": int(tiempo_entrenamiento_ms),
            "inferencia_ms": int(tiempo_inferencia_ms),
            "total_ms": int(tiempo_entrenamiento_ms + tiempo_inferencia_ms),
        },
    }
    if extra:
        resultado.update(extra)
    guardar_json(ruta, resultado)
    return resultado


def evaluar_reglas(reglas, tabla):
    datos = convertir_split_a_diccionario(tabla)
    sistema = SistemaDifusoMamdani(
        construir_membresias_base(),
        reglas=reglas,
        permitir_neutro=False,
    )
    inferencia = sistema.inferir_lote(datos["entradas"])
    return construir_metricas_desde_predicciones(
        reales=datos["riesgos"],
        predichos=inferencia["riesgos"],
        puntajes=inferencia["puntajes"],
        sin_activacion=inferencia["sin_activacion"],
    )


def construir_metricas_desde_predicciones(reales, predichos, puntajes, sin_activacion):
    predichos = np.asarray(predichos, dtype=object).copy()
    predichos[np.asarray(sin_activacion, dtype=bool)] = CLASE_SIN_ACTIVACION

    correctas = int(np.sum(reales == predichos))
    total = int(len(reales))
    etiquetas_matriz = CLASES + [CLASE_SIN_ACTIVACION]
    accuracy = float(accuracy_score(reales, predichos))
    matriz = confusion_matrix(reales, predichos, labels=etiquetas_matriz).tolist()
    balanced_accuracy = balanced_accuracy_con_clases_reales(reales, predichos)
    desviacion_puntajes = desviacion_estandar_sin_nan(puntajes)
    return {
        "accuracy": accuracy,
        "error_clasificacion": float(1.0 - accuracy),
        "errores": int(total - correctas),
        "aciertos": correctas,
        "total": total,
        "balanced_accuracy": balanced_accuracy,
        "error_balanceado": float(1.0 - balanced_accuracy),
        "desviacion_estandar_puntajes": desviacion_puntajes,
        "cobertura": float(1.0 - np.mean(sin_activacion)),
        "instancias_sin_activacion": int(np.sum(sin_activacion)),
        "matriz_confusion": {
            "etiquetas": etiquetas_matriz,
            "matriz": matriz,
        },
    }


def balanced_accuracy_con_clases_reales(reales, predichos):
    recalls = []
    for clase in CLASES:
        mascara_clase = reales == clase
        total_clase = int(np.sum(mascara_clase))
        if total_clase == 0:
            continue
        verdaderos_positivos = int(np.sum(predichos[mascara_clase] == clase))
        recalls.append(verdaderos_positivos / total_clase)
    return float(np.mean(recalls)) if recalls else 0.0


def desviacion_estandar_sin_nan(puntajes):
    puntajes_validos = np.asarray(puntajes, dtype=float)
    puntajes_validos = puntajes_validos[~np.isnan(puntajes_validos)]
    if len(puntajes_validos) <= 1:
        return 0.0
    return float(np.std(puntajes_validos, ddof=1))


def construir_membresias_base():
    membresias = {}
    for variable, especificacion in ESPECIFICACIONES_VARIABLES.items():
        membresias[variable] = {
            categoria: np.asarray(puntos, dtype=float)
            for categoria, puntos in especificacion["categorias"].items()
        }
    return membresias


def resumir_reglas(reglas):
    total_reglas = len(reglas)
    reglas_duplicadas = contar_duplicados(reglas)
    por_clase = {clase: 0 for clase in CLASES}
    longitudes = []
    for regla in reglas:
        por_clase[CONSECUENTE_A_CLASE[regla["consecuente"]]] += 1
        longitudes.append(len(regla["antecedentes"]))
    return {
        "total_reglas": int(total_reglas),
        "reglas_duplicadas": int(reglas_duplicadas),
        "proporcion_duplicados": float(reglas_duplicadas / total_reglas) if total_reglas else 0.0,
        "antecedentes_por_regla": {
            "minimo": int(min(longitudes)) if longitudes else 0,
            "promedio": float(np.mean(longitudes)) if longitudes else 0.0,
            "maximo": int(max(longitudes)) if longitudes else 0,
        },
        "reglas_por_clase": por_clase,
    }


def calcular_fitness(balanced_accuracy, duplicados, total_reglas, config_fitness):
    proporcion_duplicados = duplicados / total_reglas if total_reglas else 0.0
    return float(
        config_fitness["peso_balanced_accuracy"] * balanced_accuracy
        - config_fitness["penalizacion_duplicados"] * proporcion_duplicados
    )


def agregar_fitness(metricas, resumen_reglas, config_fitness):
    metricas["fitness"] = calcular_fitness(
        balanced_accuracy=metricas["balanced_accuracy"],
        duplicados=resumen_reglas["reglas_duplicadas"],
        total_reglas=resumen_reglas["total_reglas"],
        config_fitness=config_fitness,
    )


def traducir_parametros_ripper(parametros):
    return {
        "k": parametros["k"],
        "dl_allowance": parametros["tolerancia_longitud_descripcion"],
    }


def construir_resumen_final(resultados):
    filas = []
    for algoritmo in sorted({r["algoritmo"] for r in resultados}):
        grupo = [r for r in resultados if r["algoritmo"] == algoritmo]
        accuracies_test = [r["metricas_prueba"]["accuracy"] for r in grupo]
        errores_test = [r["metricas_prueba"]["error_clasificacion"] for r in grupo]
        errores_balanceados_test = [r["metricas_prueba"]["error_balanceado"] for r in grupo]
        fitness_test = [r["metricas_prueba"]["fitness"] for r in grupo]
        ba_test = [r["metricas_prueba"]["balanced_accuracy"] for r in grupo]
        accuracies_train = [r["metricas_entrenamiento"]["accuracy"] for r in grupo]
        ba_train = [r["metricas_entrenamiento"]["balanced_accuracy"] for r in grupo]
        fitness_train = [r["metricas_entrenamiento"]["fitness"] for r in grupo]
        filas.append(
            {
                "algoritmo": algoritmo,
                "accuracy_train_promedio": promedio(accuracies_train),
                "accuracy_train_desviacion_estandar": desviacion(accuracies_train),
                "balanced_accuracy_train_promedio": promedio(ba_train),
                "balanced_accuracy_train_desviacion_estandar": desviacion(ba_train),
                "fitness_train_promedio": promedio(fitness_train),
                "fitness_train_desviacion_estandar": desviacion(fitness_train),
                "accuracy_test_promedio": promedio(accuracies_test),
                "accuracy_test_desviacion_estandar": desviacion(accuracies_test),
                "error_test_promedio": promedio(errores_test),
                "error_test_desviacion_estandar": desviacion(errores_test),
                "error_balanceado_test_promedio": promedio(errores_balanceados_test),
                "error_balanceado_test_desviacion_estandar": desviacion(errores_balanceados_test),
                "balanced_accuracy_test_promedio": promedio(ba_test),
                "balanced_accuracy_test_desviacion_estandar": desviacion(ba_test),
                "fitness_test_promedio": promedio(fitness_test),
                "fitness_test_desviacion_estandar": desviacion(fitness_test),
                "reglas_promedio": promedio([r["resumen_reglas"]["total_reglas"] for r in grupo]),
                "duplicados_promedio": promedio([r["resumen_reglas"]["reglas_duplicadas"] for r in grupo]),
                "tiempo_total_ms_promedio": promedio([r["tiempos"]["total_ms"] for r in grupo]),
            }
        )
    return {
        "tabla_resumen": filas,
        "mejores_iteraciones": mejores_iteraciones(resultados, max),
        "peores_iteraciones": mejores_iteraciones(resultados, min),
    }


def mejores_iteraciones(resultados, funcion):
    salida = {}
    for algoritmo in sorted({r["algoritmo"] for r in resultados}):
        grupo = [r for r in resultados if r["algoritmo"] == algoritmo]
        elegido = funcion(grupo, key=lambda r: r["metricas_prueba"]["balanced_accuracy"])
        salida[algoritmo] = {
            "iteracion": elegido["iteracion"],
            "accuracy_train": elegido["metricas_entrenamiento"]["accuracy"],
            "balanced_accuracy_train": elegido["metricas_entrenamiento"]["balanced_accuracy"],
            "accuracy_test": elegido["metricas_prueba"]["accuracy"],
            "balanced_accuracy_test": elegido["metricas_prueba"]["balanced_accuracy"],
            "error_clasificacion_test": elegido["metricas_prueba"]["error_clasificacion"],
            "fitness_test": elegido["metricas_prueba"]["fitness"],
        }
    return salida


def promedio(valores):
    return float(np.mean(valores)) if valores else 0.0


def desviacion(valores):
    return float(np.std(valores, ddof=1)) if len(valores) > 1 else 0.0


def resumir_historial_ag(historial):
    filas = historial.to_dict(orient="records") if isinstance(historial, pd.DataFrame) else historial
    if not filas:
        return {
            "mejor_generacion": 0,
            "generacion_final": 0,
            "historial": [],
        }
    mejor = max(filas, key=lambda h: h["mejor_fitness"])
    return {
        "mejor_generacion": int(mejor["generacion"]),
        "generacion_final": int(filas[-1]["generacion"]),
        "fitness_inicial": float(filas[0]["mejor_fitness"]),
        "fitness_final": float(filas[-1]["mejor_fitness"]),
        "historial": filas,
    }


def asignar_origen(reglas, origen):
    for regla in reglas:
        regla["source"] = origen


def regla_a_formato_comun(regla, posicion):
    return {
        "id": f"R{posicion:03d}",
        "antecedentes": [
            {"variable": variable, "etiqueta_linguistica": termino}
            for variable, termino in regla["antecedentes"]
        ],
        "consecuente": CONSECUENTE_A_CLASE[regla["consecuente"]],
        "origen": regla.get("source"),
    }


def guardar_json(ruta, contenido):
    ruta.write_text(
        json.dumps(serializar_resultado(contenido), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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


def medir_ms(inicio):
    return int((time.perf_counter() - inicio) * 1000)


if __name__ == "__main__":
    principal()
