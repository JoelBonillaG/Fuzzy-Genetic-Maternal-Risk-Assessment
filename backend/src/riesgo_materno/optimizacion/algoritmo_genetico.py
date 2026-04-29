from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import pygad

from ..entrenamiento.modelo import PARAMETROS_AG, PESOS_FITNESS
from ..logica_difusa.motor import SistemaDifusoMamdani
from ..entrenamiento.metricas import calcular_macro_f1, calcular_recall_de_riesgo_alto
from .cromosoma import (
    CROMOSOMA_BASE,
    LIMITES_INFERIORES,
    LIMITES_SUPERIORES,
    RANGOS_GENES,
    decodificar_cromosoma,
    reparar_cromosoma,
)
from .penalizaciones import (
    calcular_penalizacion_desviacion,
    calcular_penalizacion_interpretabilidad,
)


FITNESS_INVALIDO = -1000.0

# Fraccion de la poblacion inicial generada como variaciones del cromosoma base.
# El resto se genera de forma completamente aleatoria para mantener diversidad.
FRACCION_POBLACION_PERTURBADA = 0.65

# Intensidad del ruido al crear variaciones del cromosoma base (como fraccion del rango de cada gen).
SIGMA_INICIALIZACION = 0.05

# Intensidad del cambio por mutacion en cada gen (como fraccion del rango de cada gen).
SIGMA_MUTACION = 0.05

# Limites del coeficiente de mezcla en el cruce aritmetico.
# Restringir a [0.25, 0.75] evita producir hijos identicos a uno solo de los padres.
LAMBDA_CRUCE_MIN = 0.25
LAMBDA_CRUCE_MAX = 0.75


@dataclass
class Individuo:
    cromosoma: np.ndarray
    fitness: float
    macro_f1_validacion: float
    recall_alto_validacion: float
    penalizacion_interpretabilidad: float
    penalizacion_desviacion: float


def ejecutar_algoritmo_genetico(
    datos_entrenamiento,
    datos_validacion,
    parametros_override=None,
    progress_callback=None,
):
    """Corre el AG optimizando sobre entrenamiento y eligiendo el mejor individuo por validacion."""
    parametros = {**PARAMETROS_AG, **(parametros_override or {})}
    evaluaciones_entrenamiento_cache = {}
    evaluaciones_validacion_cache = {}
    historial = []
    mejor_individuo = None
    generaciones_sin_mejora = 0
    poblacion_inicial = inicializar_poblacion(parametros["tamano_poblacion"])

    def obtener_individuo_entrenamiento(solucion):
        clave = crear_clave_solucion(solucion)
        if clave not in evaluaciones_entrenamiento_cache:
            evaluaciones_entrenamiento_cache[clave] = evaluar_individuo(solucion, datos_entrenamiento)
        return evaluaciones_entrenamiento_cache[clave]

    def obtener_individuo_validacion(solucion):
        clave = crear_clave_solucion(solucion)
        if clave not in evaluaciones_validacion_cache:
            evaluaciones_validacion_cache[clave] = evaluar_individuo(solucion, datos_validacion)
        return evaluaciones_validacion_cache[clave]

    def fitness_func(instancia_ga, solucion, indice_solucion):
        return obtener_individuo_entrenamiento(solucion).fitness

    def on_generation(instancia_ga):
        nonlocal mejor_individuo, generaciones_sin_mejora
        poblacion_validacion = [obtener_individuo_validacion(solucion) for solucion in instancia_ga.population]
        mejor_generacion = max(poblacion_validacion, key=lambda individuo: individuo.fitness)
        promedio_fitness_validacion = float(np.mean([individuo.fitness for individuo in poblacion_validacion]))

        historial.append(
            {
                "generacion": int(instancia_ga.generations_completed),
                "mejor_fitness": mejor_generacion.fitness,
                "fitness_promedio": promedio_fitness_validacion,
                "macro_f1_validacion": mejor_generacion.macro_f1_validacion,
                "recall_alto_validacion": mejor_generacion.recall_alto_validacion,
            }
        )

        print(
            f"Generacion {instancia_ga.generations_completed:03d} | "
            f"fitness={mejor_generacion.fitness:.4f} | "
            f"macro_f1={mejor_generacion.macro_f1_validacion:.4f} | "
            f"recall_alto={mejor_generacion.recall_alto_validacion:.4f}"
        )

        if progress_callback is not None:
            membresias = decodificar_cromosoma(mejor_generacion.cromosoma)
            membresias_serializables = {
                var: {cat: puntos.tolist() for cat, puntos in cats.items()}
                for var, cats in membresias.items()
            }
            progress_callback({
                "tipo": "generacion",
                "generacion": int(instancia_ga.generations_completed),
                "fitness": round(mejor_generacion.fitness, 4),
                "fitness_promedio": round(promedio_fitness_validacion, 4),
                "macro_f1": round(mejor_generacion.macro_f1_validacion, 4),
                "recall_alto": round(mejor_generacion.recall_alto_validacion, 4),
                "membresias_decodificadas": membresias_serializables,
            })

        if mejor_individuo is None or mejor_generacion.fitness > mejor_individuo.fitness:
            mejor_individuo = mejor_generacion
            generaciones_sin_mejora = 0
        else:
            generaciones_sin_mejora += 1

        if generaciones_sin_mejora >= parametros["paciencia"]:
            return "stop"
        return None

    for solucion in poblacion_inicial:
        obtener_individuo_entrenamiento(solucion)
        
    poblacion_inicial_validacion = [obtener_individuo_validacion(solucion) for solucion in poblacion_inicial]
    
    mejor_inicial = max(poblacion_inicial_validacion, key=lambda individuo: individuo.fitness)
    historial.append(
        {
            "generacion": 0,
            "mejor_fitness": mejor_inicial.fitness,
            "fitness_promedio": float(
                np.mean([individuo.fitness for individuo in poblacion_inicial_validacion])
            ),
            "macro_f1_validacion": mejor_inicial.macro_f1_validacion,
            "recall_alto_validacion": mejor_inicial.recall_alto_validacion,
        }
    )
    mejor_individuo = mejor_inicial

    instancia_ga = pygad.GA(
        initial_population=poblacion_inicial,
        num_parents_mating=parametros["cantidad_hijos"],
        fitness_func=fitness_func,
        num_generations=parametros["maximo_generaciones"],
        parent_selection_type="tournament",
        K_tournament=parametros["tamano_torneo"],
        keep_elitism=parametros["elitismo"],
        crossover_type=cruce_aritmetico,
        crossover_probability=parametros["probabilidad_cruce"],
        mutation_type=mutacion_gaussiana,
        mutation_probability=parametros["probabilidad_mutacion"],
        gene_type=float,
        gene_space=[
            {"low": float(limite_inferior), "high": float(limite_superior)}
            for limite_inferior, limite_superior in zip(LIMITES_INFERIORES, LIMITES_SUPERIORES)
        ],
        on_generation=on_generation,
        save_solutions=False,
        suppress_warnings=True,
    )
    instancia_ga.run()

    return mejor_individuo, pd.DataFrame(historial)


def evaluar_individuo(cromosoma, datos_split, cromosoma_base=CROMOSOMA_BASE):
    """Repara el cromosoma, infiere sobre un split y calcula fitness = macro_f1 + recall_alto - penalizaciones."""
    cromosoma_reparado = reparar_cromosoma(cromosoma)
    if np.isnan(cromosoma_reparado).any():
        return Individuo(cromosoma_reparado, FITNESS_INVALIDO, 0.0, 0.0, 1.0, 1.0)

    sistema = SistemaDifusoMamdani(decodificar_cromosoma(cromosoma_reparado))
    inferencia = sistema.inferir_lote(datos_split["entradas"])
    riesgos_reales = datos_split["riesgos"]
    riesgos_predichos = inferencia["riesgos"]

    macro_f1 = calcular_macro_f1(riesgos_reales, riesgos_predichos)
    recall_alto = calcular_recall_de_riesgo_alto(riesgos_reales, riesgos_predichos)
    penalizacion_interpretabilidad = calcular_penalizacion_interpretabilidad(cromosoma_reparado)
    penalizacion_desviacion = calcular_penalizacion_desviacion(
        cromosoma_reparado,
        cromosoma_base,
        RANGOS_GENES,
    )

    # macro_f1: promedio del F1 de las 3 clases (bajo/medio/alto) con igual peso
    # recall_alto: fraccion de casos reales de alto riesgo que el sistema detecto — tiene peso propio por asimetria de costos clinicos
    # penalizacion_interpretabilidad: penaliza trapecios con huecos o solapamiento excesivo — funciones dificiles de explicar
    # penalizacion_desviacion: penaliza alejarse demasiado del diseno clinico original (cromosoma base)
    fitness = (
        PESOS_FITNESS["macro_f1"] * macro_f1
        + PESOS_FITNESS["recall_alto"] * recall_alto
        - PESOS_FITNESS["interpretabilidad"] * penalizacion_interpretabilidad["total"]
        - PESOS_FITNESS["desviacion"] * penalizacion_desviacion
    )

    return Individuo(
        cromosoma_reparado,
        float(fitness),
        float(macro_f1),
        float(recall_alto),
        float(penalizacion_interpretabilidad["total"]),
        float(penalizacion_desviacion),
    )


def inicializar_poblacion(tamano=None):
    """Genera poblacion inicial: 1 cromosoma base + 65% perturbados con ruido gaussiano + 35% aleatorios."""
    if tamano is None:
        tamano = PARAMETROS_AG["tamano_poblacion"]

    # recordemos que sigma_inicializacion es el ruido a crear variaciones del cromosoma base
    # en el "espacio" o "rango" del cromosoma base.
    sigma = SIGMA_INICIALIZACION * (LIMITES_SUPERIORES - LIMITES_INFERIORES)
    
    # la poblacion arranca con un solo individuo.
    poblacion = [CROMOSOMA_BASE.copy()]

    # recordemos que mencionamos que el 65% de los indviduos seran una perturbacion/variacion del cromosoma base
    cantidad_perturbados = math.floor(FRACCION_POBLACION_PERTURBADA * (tamano - 1))

    # y los demas seran alteatorios.
    cantidad_aleatorios = tamano - 1 - cantidad_perturbados

    for _ in range(cantidad_perturbados):

        # se promedia cercano a cero, por que no queremos que los trapecios sean favorecidos a ninguna direccion
        ruido = np.random.normal(loc=0.0, scale=sigma)
        poblacion.append(reparar_cromosoma(CROMOSOMA_BASE + ruido))

    for _ in range(cantidad_aleatorios):
        # cualquier valor entre low y high tiene la misma probabilidad
        cromosoma_aleatorio = np.random.uniform(
            low=LIMITES_INFERIORES,
            high=LIMITES_SUPERIORES,
        )
        poblacion.append(reparar_cromosoma(cromosoma_aleatorio))

    return np.asarray(poblacion, dtype=float)


def cruce_aritmetico(padres, tamano_descendencia, instancia_ga):
    """Combina pares de padres con mezcla ponderada aleatoria por gen (lambda en [0.25, 0.75]) para generar hijos."""
    descendencia = []
    cantidad_genes = padres.shape[1]
    indice_padre = 0

    while len(descendencia) < tamano_descendencia[0]:
        # Selecciona padres de forma circular: al llegar al final de la lista vuelve al inicio
        padre_uno = padres[indice_padre % len(padres)]
        padre_dos = padres[(indice_padre + 1) % len(padres)]
        hijo_uno = padre_uno.copy()
        hijo_dos = padre_dos.copy()

        if np.random.random() < PARAMETROS_AG["probabilidad_cruce"]:
            lambdas = np.random.uniform(LAMBDA_CRUCE_MIN, LAMBDA_CRUCE_MAX, size=cantidad_genes)
            hijo_uno = lambdas * padre_uno + (1.0 - lambdas) * padre_dos
            hijo_dos = (1.0 - lambdas) * padre_uno + lambdas * padre_dos

        hijo_uno = reparar_cromosoma(hijo_uno)
        hijo_dos = reparar_cromosoma(hijo_dos)
        descendencia.append(hijo_uno)

        if len(descendencia) < tamano_descendencia[0]:
            descendencia.append(hijo_dos)

        indice_padre += 2

    return np.asarray(descendencia, dtype=float)


def mutacion_gaussiana(descendencia, instancia_ga):
    """Agrega ruido gaussiano escalado por el rango de cada gen a los genes seleccionados por la mascara de mutacion."""
    descendencia_mutada = np.asarray(descendencia, dtype=float).copy()
    sigma = SIGMA_MUTACION * RANGOS_GENES

    for indice in range(descendencia_mutada.shape[0]):
        # Esto decide en que posiciones habra cambio
        mascara = (
            np.random.random(size=descendencia_mutada.shape[1])
            < PARAMETROS_AG["probabilidad_mutacion"]
        )
        if mascara.any():
            descendencia_mutada[indice, mascara] += np.random.normal(
                loc=0.0,
                scale=sigma[mascara],
                size=mascara.sum(),
            )
        descendencia_mutada[indice] = reparar_cromosoma(descendencia_mutada[indice])

    return descendencia_mutada


# simplemente se utiliza para la cache
# si se genera dos veces el mismo cromosoma, se devuelve el resultado cacheado.
def crear_clave_solucion(solucion):
    """Convierte un cromosoma a bytes para usarlo como clave en el cache de evaluaciones."""
    return np.asarray(solucion, dtype=np.float64).round(8).tobytes()
