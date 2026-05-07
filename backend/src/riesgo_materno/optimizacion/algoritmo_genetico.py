"""Algoritmo genetico Pittsburgh puro para seleccion de reglas difusas.

Cada individuo es un cromosoma binario que representa una base completa de reglas.
El AG selecciona el subconjunto de reglas candidatas que maximiza:

    Fitness(S) = w_aciertos * NCP(S) - w_tamano * |S|

donde NCP(S) es el numero de patrones correctamente clasificados por la base S
y |S| es la cantidad de reglas activas.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pygad
from sklearn.metrics import balanced_accuracy_score

from ..logica_difusa.motor import SistemaDifusoMamdani
from ..entrenamiento.modelo import PARAMETROS_AG, PESOS_FITNESS
from .cromosoma import (
    CANTIDAD_REGLAS_CANDIDATAS,
    cantidad_reglas_activas,
    cromosoma_aleatorio,
    cromosoma_todas_activas,
    es_base_vacia,
    seleccion_a_base_reglas,
)


# Fitness positivo minimo entregado a PyGAD para que la ruleta funcione bien.
FITNESS_MINIMO_RULETA = 1e-6


@dataclass
class ResultadoBase:
    """Resultado de evaluar una base de reglas (un individuo Pittsburgh)."""
    cromosoma: np.ndarray
    balanced_accuracy: float  # BA(S)  — promedio del recall por clase, entre 0 y 1
    compacidad: float          # C(S)   — fraccion de reglas usadas: |S| / |Scand|, entre 0 y 1
    cantidad_reglas: int       # |S|    — numero absoluto de reglas activas, util para mostrar
    fitness: float


def ejecutar_ag_pittsburgh(
    datos_entrenamiento,
    membresias,
    parametros_override=None,
    progress_callback=None,
):
    """Ejecuta el AG Pittsburgh sobre datos_entrenamiento y devuelve el mejor individuo + historial."""
    parametros = {**PARAMETROS_AG, **(parametros_override or {})}
    cache_evaluaciones = {}
    historial = []
    mejor_resultado = None
    generaciones_sin_mejora = 0

    poblacion_inicial = inicializar_poblacion(parametros["tamano_poblacion"])

    def evaluar_con_cache(cromosoma):
        clave = np.asarray(cromosoma, dtype=int).tobytes()
        if clave not in cache_evaluaciones:
            cache_evaluaciones[clave] = evaluar_base_reglas(cromosoma, datos_entrenamiento, membresias)
        return cache_evaluaciones[clave]

    def fitness_func(instancia_ga, solucion, indice_solucion):
        # PyGAD ruleta requiere fitness positivo: aplicamos un piso minimo.
        fitness_real = evaluar_con_cache(solucion).fitness
        return max(FITNESS_MINIMO_RULETA, fitness_real)

    def on_generation(instancia_ga):
        nonlocal mejor_resultado, generaciones_sin_mejora

        evaluaciones = []
        for individuo in instancia_ga.population:
            evaluaciones.append(evaluar_con_cache(individuo))

        mejor_generacion = max(evaluaciones, key=lambda r: r.fitness)

        valores_fitness = []
        for r in evaluaciones:
            valores_fitness.append(r.fitness)
        promedio_fitness = float(np.mean(valores_fitness))

        historial.append({
            "generacion": int(instancia_ga.generations_completed),
            "mejor_fitness": mejor_generacion.fitness,
            "fitness_promedio": promedio_fitness,
            "balanced_accuracy": mejor_generacion.balanced_accuracy,
            "compacidad": mejor_generacion.compacidad,
            "cantidad_reglas": mejor_generacion.cantidad_reglas,
        })

        print(
            f"Generacion {instancia_ga.generations_completed:03d} | "
            f"fitness={mejor_generacion.fitness:.4f} | "
            f"ba={mejor_generacion.balanced_accuracy:.4f} | "
            f"reglas_activas={mejor_generacion.cantidad_reglas}"
        )

        if progress_callback is not None:
            progress_callback({
                "tipo": "generacion",
                "generacion": int(instancia_ga.generations_completed),
                "fitness": round(mejor_generacion.fitness, 4),
                "fitness_promedio": round(promedio_fitness, 4),
                "balanced_accuracy": round(mejor_generacion.balanced_accuracy, 4),
                "compacidad": round(mejor_generacion.compacidad, 4),
                "cantidad_reglas": mejor_generacion.cantidad_reglas,
            })

        if mejor_resultado is None or mejor_generacion.fitness > mejor_resultado.fitness:
            mejor_resultado = mejor_generacion
            generaciones_sin_mejora = 0
        else:
            generaciones_sin_mejora += 1

        if generaciones_sin_mejora >= parametros["paciencia"]:
            return "stop"
        return None

    # Evaluar la generacion inicial para registrarla en el historial como generacion 0.
    evaluaciones_iniciales = []
    for individuo in poblacion_inicial:
        evaluaciones_iniciales.append(evaluar_con_cache(individuo))

    mejor_inicial = max(evaluaciones_iniciales, key=lambda r: r.fitness)

    valores_fitness_iniciales = []
    for r in evaluaciones_iniciales:
        valores_fitness_iniciales.append(r.fitness)

    historial.append({
        "generacion": 0,
        "mejor_fitness": mejor_inicial.fitness,
        "fitness_promedio": float(np.mean(valores_fitness_iniciales)),
        "balanced_accuracy": mejor_inicial.balanced_accuracy,
        "compacidad": mejor_inicial.compacidad,
        "cantidad_reglas": mejor_inicial.cantidad_reglas,
    })
    mejor_resultado = mejor_inicial

    instancia_ga = pygad.GA(
        initial_population=poblacion_inicial,
        num_parents_mating=parametros["cantidad_padres"],
        fitness_func=fitness_func,
        num_generations=parametros["maximo_generaciones"],
        parent_selection_type="rws",            # ruleta
        keep_elitism=parametros["elitismo"],
        crossover_type="uniform",
        crossover_probability=parametros["probabilidad_cruce"],
        mutation_type="random",                 # con gene_space=[0,1] equivale a flip
        mutation_probability=parametros["probabilidad_mutacion"],
        gene_type=int,
        gene_space=[0, 1],
        on_generation=on_generation,
        save_solutions=False,
        suppress_warnings=True,
    )
    instancia_ga.run()

    return mejor_resultado, pd.DataFrame(historial)


def inicializar_poblacion(tamano):
    """1 cromosoma con todas las reglas activas + (tamano - 1) cromosomas aleatorios 50/50."""
    poblacion = [cromosoma_todas_activas()]
    for _ in range(tamano - 1):
        poblacion.append(cromosoma_aleatorio())
    return np.asarray(poblacion, dtype=int)


def evaluar_base_reglas(cromosoma, datos, membresias):
    """Decodifica el cromosoma a una base S, infiere sobre los datos y calcula fitness Pittsburgh.

    Entradas:
        cromosoma:  array de ints [1, 0, 1, ...] — un bit por regla candidata
        datos:      dict {"entradas": {variable: np.array}, "riesgos": np.array de strings}
        membresias: dict {variable: {categoria: np.array([a, b, c, d])}}
    Salida:
        ResultadoBase con aciertos (NCP), cantidad_reglas (|S|) y fitness.

    Si la base resulta vacia, devuelve fitness 0 (penalizacion fuerte para la ruleta).
    """
    cromosoma = np.asarray(cromosoma, dtype=int)
    cantidad_reglas = cantidad_reglas_activas(cromosoma)

    if es_base_vacia(cromosoma):
        return ResultadoBase(
            cromosoma=cromosoma,
            balanced_accuracy=0.0,
            compacidad=0.0,
            cantidad_reglas=0,
            fitness=0.0,
        )

    reglas_activas = seleccion_a_base_reglas(cromosoma)
    sistema = SistemaDifusoMamdani(membresias, reglas=reglas_activas)
    inferencia = sistema.inferir_lote(datos["entradas"])

    riesgos_predichos = inferencia["riesgos"]
    riesgos_reales = datos["riesgos"]

    # BA: promedio del recall por clase — trata todas las clases por igual sin importar su tamano
    ba = float(balanced_accuracy_score(riesgos_reales, riesgos_predichos))

    # C(S): fraccion de reglas candidatas seleccionadas — penaliza bases grandes
    compacidad = cantidad_reglas / CANTIDAD_REGLAS_CANDIDATAS

    fitness = (
        PESOS_FITNESS["balanced_accuracy"] * ba
        - PESOS_FITNESS["compacidad"] * compacidad
    )

    return ResultadoBase(
        cromosoma=cromosoma,
        balanced_accuracy=ba,
        compacidad=compacidad,
        cantidad_reglas=cantidad_reglas,
        fitness=float(fitness),
    )
