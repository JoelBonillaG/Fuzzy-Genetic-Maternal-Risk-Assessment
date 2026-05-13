"""AG Pittsburgh-Michigan con cromosoma numerico usando PyGAD.

Genotipo:
    Cromosoma plano de longitud reglas_por_individuo * 7.
    Cada bloque de 7 genes representa una regla:
    [edad, sistolica, diastolica, glucosa, temperatura, frecuencia, clase]

Fenotipo:
    Base completa de reglas difusas consumida por el motor Mamdani.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pygad
from sklearn.metrics import balanced_accuracy_score

from ...entrenamiento.datos import convertir_split_a_diccionario
from ...entrenamiento.ripper import _discretizar
from ...logica_difusa.motor import SistemaDifusoMamdani
from ...logica_difusa.variables import ESPECIFICACIONES_VARIABLES, ETIQUETAS_RIESGO, VARIABLES_ENTRADA


GENES_POR_REGLA = 7
CLASE_A_CONSECUENTE = {"low risk": "bajo", "mid risk": "medio", "high risk": "alto"}
CONSECUENTE_A_CLASE = {v: k for k, v in CLASE_A_CONSECUENTE.items()}


@dataclass
class ResultadoPittsburghMichigan:
    cromosoma: np.ndarray
    reglas: list[dict]
    fitness: float
    balanced_accuracy: float
    duplicados: int
    proporcion_duplicados: float


def ejecutar_ag_pittsburgh_michigan(
    tabla,
    membresias,
    parametros,
    progress_callback=None,
):
    """Evoluciona una base fija de reglas completas mediante PyGAD."""
    df_discretizado = pd.DataFrame(_discretizar(tabla)).rename(columns={"clase": "riesgo"})
    codificacion = construir_codificacion()
    poblacion_inicial = inicializar_poblacion(df_discretizado, parametros, codificacion)
    gene_space = construir_gene_space(parametros["reglas_por_individuo"], codificacion)
    cache_evaluaciones = {}
    historial = []
    mejor_resultado = None
    generaciones_sin_mejora = 0

    def evaluar_con_cache(cromosoma):
        clave = np.asarray(cromosoma, dtype=int).tobytes()
        if clave not in cache_evaluaciones:
            cache_evaluaciones[clave] = evaluar_cromosoma(
                cromosoma=cromosoma,
                tabla=tabla,
                membresias=membresias,
                parametros=parametros,
                codificacion=codificacion,
            )
        return cache_evaluaciones[clave]

    def fitness_func(instancia_ga, solucion, indice_solucion):
        return evaluar_con_cache(solucion).fitness

    def crossover_por_bloques_regla(padres, tamano_descendencia, instancia_ga):
        return cruzar_por_bloques_regla(
            padres=padres,
            tamano_descendencia=tamano_descendencia,
            probabilidad_cruce=parametros["probabilidad_cruce"],
            reglas_por_individuo=parametros["reglas_por_individuo"],
        )

    def mutacion_michigan(descendencia, instancia_ga):
        return mutar_descendencia_michigan(
            descendencia=descendencia,
            df_discretizado=df_discretizado,
            parametros=parametros,
            codificacion=codificacion,
        )

    def registrar_generacion(generacion, poblacion):
        nonlocal mejor_resultado, generaciones_sin_mejora

        evaluaciones = [evaluar_con_cache(individuo) for individuo in poblacion]
        mejor_generacion = max(evaluaciones, key=lambda r: r.fitness)
        if mejor_resultado is None or mejor_generacion.fitness > mejor_resultado.fitness:
            mejor_resultado = mejor_generacion
            generaciones_sin_mejora = 0
        else:
            generaciones_sin_mejora += 1

        fila = construir_fila_historial(generacion, evaluaciones, mejor_generacion)
        historial.append(fila)
        print(
            f"  AG-PM  | gen={generacion:04d} "
            f"fitness={mejor_generacion.fitness:.4f} "
            f"ba={mejor_generacion.balanced_accuracy:.4f} "
            f"dup={mejor_generacion.duplicados}"
        )
        if progress_callback is not None:
            progress_callback(fila)

    def on_generation(instancia_ga):
        registrar_generacion(
            instancia_ga.generations_completed,
            instancia_ga.population,
        )
        if generaciones_sin_mejora >= parametros["paciencia"]:
            return "stop"
        return None

    registrar_generacion(0, poblacion_inicial)
    cantidad_padres = normalizar_cantidad_padres(parametros)

    instancia_ga = pygad.GA(
        initial_population=poblacion_inicial,
        num_parents_mating=cantidad_padres,
        fitness_func=fitness_func,
        num_generations=parametros["maximo_generaciones"],
        parent_selection_type="tournament",
        K_tournament=parametros["tamano_torneo"],
        keep_elitism=parametros["elitismo"],
        crossover_type=crossover_por_bloques_regla,
        mutation_type=mutacion_michigan,
        gene_type=int,
        gene_space=gene_space,
        on_generation=on_generation,
        save_solutions=False,
        suppress_warnings=True,
    )
    instancia_ga.run()

    mejor_cromosoma = np.asarray(mejor_resultado.cromosoma, dtype=int).copy()
    mejor_reglas = decodificar_cromosoma(mejor_cromosoma, codificacion)
    mejor_resultado = ResultadoPittsburghMichigan(
        cromosoma=mejor_cromosoma,
        reglas=mejor_reglas,
        fitness=mejor_resultado.fitness,
        balanced_accuracy=mejor_resultado.balanced_accuracy,
        duplicados=mejor_resultado.duplicados,
        proporcion_duplicados=mejor_resultado.proporcion_duplicados,
    )
    return mejor_resultado, pd.DataFrame(historial)


def normalizar_cantidad_padres(parametros):
    """Devuelve una cantidad valida y explicita de padres para PyGAD."""
    cantidad = int(parametros["cantidad_padres"])
    poblacion = int(parametros["tamano_poblacion"])
    if cantidad < 2:
        raise ValueError("cantidad_padres debe ser al menos 2.")
    if cantidad > poblacion:
        raise ValueError("cantidad_padres no puede superar tamano_poblacion.")
    return cantidad


def construir_codificacion():
    categorias_por_variable = {
        variable: list(ESPECIFICACIONES_VARIABLES[variable]["categorias"].keys())
        for variable in VARIABLES_ENTRADA
    }
    indice_por_categoria = {
        variable: {categoria: indice for indice, categoria in enumerate(categorias)}
        for variable, categorias in categorias_por_variable.items()
    }
    return {
        "categorias_por_variable": categorias_por_variable,
        "indice_por_categoria": indice_por_categoria,
        "clases": list(ETIQUETAS_RIESGO),
        "indice_por_clase": {clase: indice for indice, clase in enumerate(ETIQUETAS_RIESGO)},
    }


def construir_gene_space(reglas_por_individuo, codificacion):
    espacios_regla = []
    for variable in VARIABLES_ENTRADA:
        espacios_regla.append(
            list(range(len(codificacion["categorias_por_variable"][variable])))
        )
    espacios_regla.append(list(range(len(codificacion["clases"]))))
    return espacios_regla * reglas_por_individuo


def inicializar_poblacion(df_discretizado, parametros, codificacion):
    poblacion = []
    for _ in range(parametros["tamano_poblacion"]):
        poblacion.append(generar_cromosoma_inicial(parametros, codificacion))
    return np.asarray(poblacion, dtype=int)


def generar_cromosoma_inicial(parametros, codificacion):
    """Genera un cromosoma 100% aleatorio dentro de rangos linguisticos validos."""
    genes = []
    cantidad_reglas = parametros["reglas_por_individuo"]

    for _ in range(cantidad_reglas):
        for variable in VARIABLES_ENTRADA:
            cantidad_categorias = len(codificacion["categorias_por_variable"][variable])
            genes.append(int(np.random.randint(0, cantidad_categorias)))
        genes.append(int(np.random.randint(0, len(codificacion["clases"]))))

    return np.asarray(genes, dtype=int)


def cruzar_por_bloques_regla(padres, tamano_descendencia, probabilidad_cruce, reglas_por_individuo):
    descendencia = np.empty(tamano_descendencia, dtype=int)
    for indice_hijo in range(tamano_descendencia[0]):
        padre_a = padres[indice_hijo % padres.shape[0]]
        padre_b = padres[(indice_hijo + 1) % padres.shape[0]]

        if np.random.random() >= probabilidad_cruce or reglas_por_individuo < 2:
            descendencia[indice_hijo] = padre_a.copy()
            continue

        corte_1, corte_2 = sorted(
            np.random.choice(np.arange(1, reglas_por_individuo), size=2, replace=False)
        )
        gen_1 = corte_1 * GENES_POR_REGLA
        gen_2 = corte_2 * GENES_POR_REGLA
        descendencia[indice_hijo] = np.concatenate([
            padre_a[:gen_1],
            padre_b[gen_1:gen_2],
            padre_a[gen_2:],
        ])
    return descendencia


def mutar_descendencia_michigan(descendencia, df_discretizado, parametros, codificacion):
    descendencia = np.asarray(descendencia, dtype=int).copy()
    cantidad_reglas = parametros["reglas_por_individuo"]

    for indice_hijo in range(descendencia.shape[0]):
        matriz = descendencia[indice_hijo].reshape(cantidad_reglas, GENES_POR_REGLA).copy()

        for indice_regla in range(cantidad_reglas):
            if np.random.random() < parametros["probabilidad_mutacion"]:
                mutar_gen_regla(matriz[indice_regla], codificacion, forzar=False)

        if np.random.random() < parametros["probabilidad_reemplazo"]:
            reemplazar_reglas_malas(
                matriz=matriz,
                df_discretizado=df_discretizado,
                parametros=parametros,
                codificacion=codificacion,
            )

        descendencia[indice_hijo] = matriz.reshape(-1)

    return descendencia


def reemplazar_reglas_malas(matriz, df_discretizado, parametros, codificacion):
    calidades = evaluar_calidad_reglas_codificadas(matriz, df_discretizado, codificacion)
    cantidad_reemplazos = max(1, int(len(matriz) * parametros["fraccion_reemplazo"]))
    peores = np.argsort(calidades)[:cantidad_reemplazos]
    mejores = np.argsort(calidades)[::-1][: max(1, cantidad_reemplazos * 2)]

    for indice_malo in peores:
        indice_bueno = int(np.random.choice(mejores))
        matriz[indice_malo] = matriz[indice_bueno].copy()
        mutar_gen_regla(matriz[indice_malo], codificacion, forzar=True)


def mutar_gen_regla(regla_codificada, codificacion, forzar=False):
    mutar_consecuente = np.random.random() < 0.25
    if forzar and np.random.random() < 0.50:
        mutar_consecuente = True

    if mutar_consecuente:
        opciones = list(range(len(codificacion["clases"])))
        regla_codificada[-1] = elegir_distinto(opciones, int(regla_codificada[-1]))
        return

    indice_variable = int(np.random.randint(0, len(VARIABLES_ENTRADA)))
    variable = VARIABLES_ENTRADA[indice_variable]
    opciones = list(range(len(codificacion["categorias_por_variable"][variable])))
    regla_codificada[indice_variable] = elegir_distinto(
        opciones,
        int(regla_codificada[indice_variable]),
    )


def elegir_distinto(opciones, actual):
    opciones = [opcion for opcion in opciones if opcion != actual]
    return int(np.random.choice(opciones))


def evaluar_calidad_reglas_codificadas(matriz, df_discretizado, codificacion):
    calidades = []
    for regla_codificada in matriz:
        cubiertos = df_discretizado
        for indice_variable, variable in enumerate(VARIABLES_ENTRADA):
            categoria = decodificar_categoria(variable, regla_codificada[indice_variable], codificacion)
            cubiertos = cubiertos[cubiertos[variable] == categoria]

        clase = codificacion["clases"][int(regla_codificada[-1])]
        cobertura = int(len(cubiertos))
        if cobertura == 0:
            precision = 0.0
        else:
            precision = float((cubiertos["riesgo"] == clase).sum() / cobertura)
        calidades.append(precision * np.log1p(cobertura))
    return np.asarray(calidades, dtype=float)


def evaluar_cromosoma(cromosoma, tabla, membresias, parametros, codificacion):
    cromosoma = np.asarray(cromosoma, dtype=int).copy()
    reglas = decodificar_cromosoma(cromosoma, codificacion)
    datos = convertir_split_a_diccionario(tabla)
    sistema = SistemaDifusoMamdani(membresias, reglas=reglas)
    inferencia = sistema.inferir_lote(datos["entradas"])
    ba = float(balanced_accuracy_score(datos["riesgos"], inferencia["riesgos"]))
    duplicados = contar_duplicados(reglas)
    proporcion_duplicados = duplicados / len(reglas) if reglas else 0.0
    fitness = (
        parametros["peso_balanced_accuracy"] * ba
        - parametros["penalizacion_duplicados"] * proporcion_duplicados
    )
    return ResultadoPittsburghMichigan(
        cromosoma=cromosoma,
        reglas=reglas,
        fitness=float(fitness),
        balanced_accuracy=ba,
        duplicados=duplicados,
        proporcion_duplicados=float(proporcion_duplicados),
    )


def decodificar_cromosoma(cromosoma, codificacion):
    matriz = np.asarray(cromosoma, dtype=int).reshape(-1, GENES_POR_REGLA)
    reglas = []
    for numero, regla_codificada in enumerate(matriz, start=1):
        antecedentes = []
        for indice_variable, variable in enumerate(VARIABLES_ENTRADA):
            antecedentes.append((
                variable,
                decodificar_categoria(variable, regla_codificada[indice_variable], codificacion),
            ))
        clase = codificacion["clases"][int(regla_codificada[-1])]
        reglas.append({
            "numero": numero,
            "antecedentes": antecedentes,
            "consecuente": CLASE_A_CONSECUENTE[clase],
            "source": "AG_PITTSBURGH_MICHIGAN",
        })
    return reglas


def decodificar_categoria(variable, indice_categoria, codificacion):
    categorias = codificacion["categorias_por_variable"][variable]
    indice = int(indice_categoria)
    return categorias[indice]


def contar_duplicados(reglas):
    claves = []
    for regla in reglas:
        claves.append((
            tuple(regla["antecedentes"]),
            regla["consecuente"],
        ))
    return len(claves) - len(set(claves))


def construir_fila_historial(generacion, evaluaciones, mejor):
    fitness = [r.fitness for r in evaluaciones]
    ba = [r.balanced_accuracy for r in evaluaciones]
    duplicados = [r.duplicados for r in evaluaciones]
    return {
        "generacion": int(generacion),
        "mejor_fitness": float(mejor.fitness),
        "fitness_promedio": float(np.mean(fitness)),
        "mejor_balanced_accuracy": float(mejor.balanced_accuracy),
        "balanced_accuracy_promedio": float(np.mean(ba)),
        "duplicados_mejor": int(mejor.duplicados),
        "duplicados_promedio": float(np.mean(duplicados)),
    }
