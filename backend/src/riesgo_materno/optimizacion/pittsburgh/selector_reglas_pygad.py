"""Selector genetico de reglas candidatas usando PyGAD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pygad
from sklearn.metrics import balanced_accuracy_score

from ...logica_difusa.motor import SistemaDifusoMamdani


FITNESS_MINIMO_RULETA = 1e-6


@dataclass
class ResultadoSeleccion:
    cromosoma: np.ndarray
    fitness: float
    balanced_accuracy: float
    cantidad_reglas: int


def ejecutar_selector_reglas_pygad(
    reglas_candidatas,
    datos_entrenamiento,
    membresias,
    parametros,
    progress_callback=None,
):
    """Selecciona un subconjunto de reglas candidatas con cromosomas binarios."""
    cache_evaluaciones = {}
    historial = []
    mejor_resultado = None
    generaciones_sin_mejora = 0

    poblacion_inicial = inicializar_poblacion(
        parametros["population_size"],
        len(reglas_candidatas),
    )

    def evaluar_con_cache(cromosoma):
        clave = np.asarray(cromosoma, dtype=int).tobytes()
        if clave not in cache_evaluaciones:
            cache_evaluaciones[clave] = evaluar_individuo(
                cromosoma,
                reglas_candidatas,
                datos_entrenamiento,
                membresias,
                parametros["lambda_penalty"],
            )
        return cache_evaluaciones[clave]

    def fitness_func(instancia_ga, solucion, indice_solucion):
        fitness_real = evaluar_con_cache(solucion).fitness
        return max(FITNESS_MINIMO_RULETA, fitness_real)

    def on_generation(instancia_ga):
        nonlocal mejor_resultado, generaciones_sin_mejora
        evaluaciones = [evaluar_con_cache(individuo) for individuo in instancia_ga.population]
        mejor_generacion = max(evaluaciones, key=lambda r: r.fitness)
        promedio_fitness = float(np.mean([r.fitness for r in evaluaciones]))
        promedio_ba = float(np.mean([r.balanced_accuracy for r in evaluaciones]))
        cantidades = [r.cantidad_reglas for r in evaluaciones]
        historial.append(
            {
                "generation": int(instancia_ga.generations_completed),
                "best_fitness": mejor_generacion.fitness,
                "mean_fitness": promedio_fitness,
                "best_balanced_accuracy": mejor_generacion.balanced_accuracy,
                "mean_balanced_accuracy": promedio_ba,
                "active_rules": mejor_generacion.cantidad_reglas,
                "rules_min": int(np.min(cantidades)),
                "rules_mean": float(np.mean(cantidades)),
                "rules_max": int(np.max(cantidades)),
            }
        )
        print(
            f"  GA     | gen={instancia_ga.generations_completed:04d} "
            f"fitness={mejor_generacion.fitness:.4f} "
            f"ba={mejor_generacion.balanced_accuracy:.4f} "
            f"reglas={mejor_generacion.cantidad_reglas} "
            f"fit_prom={promedio_fitness:.4f}"
        )
        if progress_callback is not None:
            progress_callback(historial[-1])
        if mejor_resultado is None or mejor_generacion.fitness > mejor_resultado.fitness:
            mejor_resultado = mejor_generacion
            generaciones_sin_mejora = 0
        else:
            generaciones_sin_mejora += 1
        if generaciones_sin_mejora >= parametros["patience"]:
            return "stop"
        return None

    evaluaciones_iniciales = [evaluar_con_cache(individuo) for individuo in poblacion_inicial]
    mejor_inicial = max(evaluaciones_iniciales, key=lambda r: r.fitness)
    historial.append(
        {
            "generation": 0,
            "best_fitness": mejor_inicial.fitness,
            "mean_fitness": float(np.mean([r.fitness for r in evaluaciones_iniciales])),
            "best_balanced_accuracy": mejor_inicial.balanced_accuracy,
            "mean_balanced_accuracy": float(np.mean([r.balanced_accuracy for r in evaluaciones_iniciales])),
            "active_rules": mejor_inicial.cantidad_reglas,
            "rules_min": int(np.min([r.cantidad_reglas for r in evaluaciones_iniciales])),
            "rules_mean": float(np.mean([r.cantidad_reglas for r in evaluaciones_iniciales])),
            "rules_max": int(np.max([r.cantidad_reglas for r in evaluaciones_iniciales])),
        }
    )
    mejor_resultado = mejor_inicial
    print(
        f"  GA     | gen=0000 fitness={mejor_inicial.fitness:.4f} "
        f"ba={mejor_inicial.balanced_accuracy:.4f} "
        f"reglas={mejor_inicial.cantidad_reglas} "
        f"fit_prom={historial[-1]['mean_fitness']:.4f}"
    )

    instancia_ga = pygad.GA(
        initial_population=poblacion_inicial,
        num_parents_mating=max(2, parametros["population_size"] // 3),
        fitness_func=fitness_func,
        num_generations=parametros["max_generations"],
        parent_selection_type="rws",
        keep_elitism=parametros["elitism_count"],
        crossover_type="single_point",
        crossover_probability=parametros["crossover_rate"],
        mutation_type="random",
        mutation_probability=parametros["mutation_rate"],
        gene_type=int,
        gene_space=[0, 1],
        on_generation=on_generation,
        save_solutions=False,
        suppress_warnings=True,
    )
    instancia_ga.run()

    return mejor_resultado, pd.DataFrame(historial)


def inicializar_poblacion(tamano_poblacion, longitud_cromosoma):
    poblacion = np.random.randint(0, 2, size=(tamano_poblacion, longitud_cromosoma)).astype(int)
    if longitud_cromosoma > 0:
        poblacion[0, :] = 1
    return poblacion


def evaluar_individuo(cromosoma, reglas_candidatas, datos, membresias, lambda_penalty):
    cromosoma = np.asarray(cromosoma, dtype=int).copy()
    indices = np.where(cromosoma == 1)[0].tolist()
    if not indices:
        return ResultadoSeleccion(cromosoma, 0.0, 0.0, 0)

    reglas_activas = [reglas_candidatas[i] for i in indices]
    sistema = SistemaDifusoMamdani(membresias, reglas=reglas_activas)
    inferencia = sistema.inferir_lote(datos["entradas"])
    predichos = predicciones_con_sin_activacion(
        inferencia["riesgos"],
        inferencia["sin_activacion"],
    )
    ba = float(balanced_accuracy_score(datos["riesgos"], predichos))
    cantidad_reglas = len(reglas_activas)
    fitness = ba - lambda_penalty * (cantidad_reglas / len(reglas_candidatas))
    return ResultadoSeleccion(
        cromosoma=cromosoma,
        fitness=float(fitness),
        balanced_accuracy=ba,
        cantidad_reglas=cantidad_reglas,
    )


def predicciones_con_sin_activacion(predichos, sin_activacion):
    predichos = np.asarray(predichos, dtype=object).copy()
    predichos[np.asarray(sin_activacion, dtype=bool)] = "__sin_activacion__"
    return predichos
