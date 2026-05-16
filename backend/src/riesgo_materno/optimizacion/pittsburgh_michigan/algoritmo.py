"""AG Pittsburgh-Michigan con cromosoma binario usando PyGAD.

Genotipo:
    Cromosoma plano de longitud reglas_por_individuo * 21.
    Cada bloque de 21 bits representa una regla:
    6 antecedentes * 3 bits + clase * 3 bits.

Fenotipo:
    Base completa de reglas difusas consumida por el motor Mamdani.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pygad
import skfuzzy as fuzz

from ...entrenamiento.datos import convertir_split_a_diccionario
from ...entrenamiento.ripper import _discretizar
from ...logica_difusa.motor import SistemaDifusoMamdani
from ...logica_difusa.variables import (
    ESPECIFICACIONES_VARIABLES,
    ETIQUETAS_RIESGO,
    PUNTOS_GRAFICA,
    VARIABLES_ENTRADA,
)


BITS_POR_CAMPO = 3
CAMPOS_POR_REGLA = 7
BITS_POR_REGLA = CAMPOS_POR_REGLA * BITS_POR_CAMPO
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
    pertenencias_fuzzy = construir_pertenencias_fuzzy(tabla, membresias)
    poblacion_inicial = inicializar_poblacion(
        df_discretizado,
        pertenencias_fuzzy,
        parametros,
        codificacion,
    )
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
                df_discretizado=df_discretizado,
                pertenencias_fuzzy=pertenencias_fuzzy,
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
            pertenencias_fuzzy=pertenencias_fuzzy,
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
        gene_space=[0, 1],
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


def inicializar_poblacion(df_discretizado, pertenencias_fuzzy, parametros, codificacion):
    poblacion = []
    for _ in range(parametros["tamano_poblacion"]):
        poblacion.append(generar_cromosoma_inicial(parametros, codificacion, df_discretizado, pertenencias_fuzzy))
    return np.asarray(poblacion, dtype=int)


def generar_cromosoma_inicial(parametros, codificacion, df_discretizado, pertenencias_fuzzy):
    """Genera un cromosoma binario aleatorio con reglas validas y cobertura."""
    genes = []
    cantidad_reglas = parametros["reglas_por_individuo"]

    if parametros.get("balancear_consecuentes_por_clase", True):
        cuotas = calcular_cuotas_por_clase(cantidad_reglas, codificacion["clases"])
        clases_objetivo = [
            clase
            for clase, cantidad_clase in cuotas.items()
            for _ in range(cantidad_clase)
        ]
        np.random.shuffle(clases_objetivo)
    else:
        clases_objetivo = [None] * cantidad_reglas

    for clase_objetivo in clases_objetivo:
        genes.extend(
            generar_regla_binaria_aleatoria_con_cobertura(
                codificacion=codificacion,
                df_discretizado=df_discretizado,
                pertenencias_fuzzy=pertenencias_fuzzy,
                clase_objetivo=clase_objetivo,
            ).tolist()
        )

    return np.asarray(genes, dtype=int)


def calcular_cuotas_por_clase(cantidad, clases):
    cantidad = int(cantidad)
    clases = list(clases)
    base = cantidad // len(clases)
    sobrante = cantidad % len(clases)
    return {
        clase: base + (1 if indice < sobrante else 0)
        for indice, clase in enumerate(clases)
    }


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
        gen_1 = corte_1 * BITS_POR_REGLA
        gen_2 = corte_2 * BITS_POR_REGLA
        descendencia[indice_hijo] = np.concatenate([
            padre_a[:gen_1],
            padre_b[gen_1:gen_2],
            padre_a[gen_2:],
        ])
    return descendencia


def mutar_descendencia_michigan(descendencia, df_discretizado, pertenencias_fuzzy, parametros, codificacion):
    descendencia = np.asarray(descendencia, dtype=int).copy()
    cantidad_reglas = parametros["reglas_por_individuo"]
    balancear_clases = parametros.get("balancear_consecuentes_por_clase", True)

    for indice_hijo in range(descendencia.shape[0]):
        matriz = descendencia[indice_hijo].reshape(cantidad_reglas, BITS_POR_REGLA).copy()

        for indice_regla in range(cantidad_reglas):
            if np.random.random() < parametros["probabilidad_mutacion"]:
                mutar_gen_regla(matriz[indice_regla], codificacion, forzar=False)
                if not regla_tiene_cobertura(matriz[indice_regla], codificacion, df_discretizado, pertenencias_fuzzy):
                    matriz[indice_regla] = generar_regla_binaria_aleatoria_con_cobertura(
                        codificacion=codificacion,
                        df_discretizado=df_discretizado,
                        pertenencias_fuzzy=pertenencias_fuzzy,
                        clase_objetivo=decodificar_clase_regla(matriz[indice_regla], codificacion),
                    )

        if np.random.random() < parametros["probabilidad_reemplazo"]:
            reemplazar_reglas_malas(
                matriz=matriz,
                df_discretizado=df_discretizado,
                pertenencias_fuzzy=pertenencias_fuzzy,
                parametros=parametros,
                codificacion=codificacion,
            )

        reparar_reglas_sin_cobertura(
            matriz=matriz,
            codificacion=codificacion,
            df_discretizado=df_discretizado,
            pertenencias_fuzzy=pertenencias_fuzzy,
        )
        if balancear_clases:
            reparar_balance_clases(
                matriz=matriz,
                codificacion=codificacion,
                df_discretizado=df_discretizado,
                pertenencias_fuzzy=pertenencias_fuzzy,
            )

        descendencia[indice_hijo] = matriz.reshape(-1)

    return descendencia


def reemplazar_reglas_malas(matriz, df_discretizado, pertenencias_fuzzy, parametros, codificacion):
    calidades = evaluar_calidad_reglas_codificadas(matriz, df_discretizado, codificacion)
    cantidad_reemplazos = max(1, int(len(matriz) * parametros["fraccion_reemplazo"]))
    peores = np.argsort(calidades)[:cantidad_reemplazos]
    mejores = np.argsort(calidades)[::-1][: max(1, cantidad_reemplazos * 2)]

    for indice_malo in peores:
        indice_bueno = int(np.random.choice(mejores))
        matriz[indice_malo] = matriz[indice_bueno].copy()
        mutar_gen_regla(matriz[indice_malo], codificacion, forzar=True)
        if not regla_tiene_cobertura(matriz[indice_malo], codificacion, df_discretizado, pertenencias_fuzzy):
            matriz[indice_malo] = generar_regla_binaria_aleatoria_con_cobertura(
                codificacion=codificacion,
                df_discretizado=df_discretizado,
                pertenencias_fuzzy=pertenencias_fuzzy,
                clase_objetivo=decodificar_clase_regla(matriz[indice_malo], codificacion),
            )


def mutar_gen_regla(regla_codificada, codificacion, forzar=False):
    """Muta solo antecedentes cambiando una categoria completa valida."""
    indice_variable = int(np.random.randint(0, len(VARIABLES_ENTRADA)))
    inicio = indice_variable * BITS_POR_CAMPO
    fin = inicio + BITS_POR_CAMPO
    variable = VARIABLES_ENTRADA[indice_variable]
    categorias = codificacion["categorias_por_variable"][variable]
    indice_actual = bits_a_indice(regla_codificada[inicio:fin])
    indices_validos = list(range(len(categorias)))
    if indice_actual in indices_validos and len(indices_validos) > 1:
        indices_validos.remove(indice_actual)
    if not indices_validos:
        return

    nuevo_indice = int(np.random.choice(indices_validos))
    regla_codificada[inicio:fin] = indice_a_bits(nuevo_indice)

def evaluar_calidad_reglas_codificadas(matriz, df_discretizado, codificacion):
    calidades = []
    for regla_codificada in matriz:
        regla_decodificada = decodificar_regla_binaria(regla_codificada, codificacion)
        if regla_decodificada is None:
            calidades.append(0.0)
            continue
        cubiertos = df_discretizado
        antecedentes, clase = regla_decodificada
        for variable, categoria in antecedentes:
            cubiertos = cubiertos[cubiertos[variable] == categoria]

        cobertura = int(len(cubiertos))
        if cobertura == 0:
            precision = 0.0
        else:
            precision = float((cubiertos["riesgo"] == clase).sum() / cobertura)
        calidades.append(precision * np.log1p(cobertura))
    return np.asarray(calidades, dtype=float)


def evaluar_cromosoma(
    cromosoma,
    tabla,
    membresias,
    parametros,
    codificacion,
    df_discretizado,
    pertenencias_fuzzy,
):
    cromosoma = np.asarray(cromosoma, dtype=int).copy()
    cromosoma = reparar_cromosoma_sin_cobertura(
        cromosoma,
        parametros,
        codificacion,
        df_discretizado,
        pertenencias_fuzzy,
    )
    reglas = decodificar_cromosoma(cromosoma, codificacion)
    datos = convertir_split_a_diccionario(tabla)
    sistema = SistemaDifusoMamdani(membresias, reglas=reglas, permitir_neutro=False)
    inferencia = sistema.inferir_lote(datos["entradas"])
    ba = balanced_accuracy_con_sin_activacion_invalida(
        reales=datos["riesgos"],
        predichos=inferencia["riesgos"],
        sin_activacion=inferencia["sin_activacion"],
    )
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
    matriz = np.asarray(cromosoma, dtype=int).reshape(-1, BITS_POR_REGLA)
    reglas = []
    for numero, regla_codificada in enumerate(matriz, start=1):
        regla_decodificada = decodificar_regla_binaria(regla_codificada, codificacion)
        if regla_decodificada is None:
            continue
        antecedentes, clase = regla_decodificada
        reglas.append({
            "numero": numero,
            "antecedentes": antecedentes,
            "consecuente": CLASE_A_CONSECUENTE[clase],
            "source": "AG_PITTSBURGH_MICHIGAN",
        })
    return reglas


def generar_regla_binaria_valida(codificacion, clase_objetivo=None):
    bits = []
    for variable in VARIABLES_ENTRADA:
        indice = int(np.random.randint(0, len(codificacion["categorias_por_variable"][variable])))
        bits.extend(indice_a_bits(indice))
    if clase_objetivo is None:
        indice_clase = int(np.random.randint(0, len(codificacion["clases"])))
    else:
        indice_clase = int(codificacion["indice_por_clase"][clase_objetivo])
    bits.extend(indice_a_bits(indice_clase))
    return np.asarray(bits, dtype=int)


def generar_regla_binaria_aleatoria_con_cobertura(
    codificacion,
    df_discretizado,
    pertenencias_fuzzy,
    clase_objetivo=None,
    max_intentos=10000,
):
    for _ in range(max_intentos):
        regla = generar_regla_binaria_valida(codificacion, clase_objetivo=clase_objetivo)
        if regla_tiene_cobertura(regla, codificacion, df_discretizado, pertenencias_fuzzy):
            return regla
    raise RuntimeError(
        "No se pudo generar una regla aleatoria con cobertura fuzzy y discreta "
        f"despues de {max_intentos} intentos."
    )


def reparar_cromosoma_sin_cobertura(cromosoma, parametros, codificacion, df_discretizado, pertenencias_fuzzy):
    matriz = np.asarray(cromosoma, dtype=int).reshape(parametros["reglas_por_individuo"], BITS_POR_REGLA).copy()
    reparar_reglas_sin_cobertura(matriz, codificacion, df_discretizado, pertenencias_fuzzy)
    if parametros.get("balancear_consecuentes_por_clase", True):
        reparar_balance_clases(matriz, codificacion, df_discretizado, pertenencias_fuzzy)
    return matriz.reshape(-1)


def reparar_reglas_sin_cobertura(matriz, codificacion, df_discretizado, pertenencias_fuzzy):
    for indice_regla, regla_codificada in enumerate(matriz):
        if regla_tiene_cobertura(regla_codificada, codificacion, df_discretizado, pertenencias_fuzzy):
            continue
        matriz[indice_regla] = generar_regla_binaria_aleatoria_con_cobertura(
            codificacion=codificacion,
            df_discretizado=df_discretizado,
            pertenencias_fuzzy=pertenencias_fuzzy,
            clase_objetivo=decodificar_clase_regla(regla_codificada, codificacion),
        )


def reparar_balance_clases(matriz, codificacion, df_discretizado, pertenencias_fuzzy):
    cuotas = calcular_cuotas_por_clase(len(matriz), codificacion["clases"])
    indices_por_clase = {clase: [] for clase in codificacion["clases"]}
    indices_invalidos = []

    for indice, regla_codificada in enumerate(matriz):
        clase = decodificar_clase_regla(regla_codificada, codificacion)
        if clase not in indices_por_clase:
            indices_invalidos.append(indice)
            continue
        indices_por_clase[clase].append(indice)

    indices_reemplazables = list(indices_invalidos)
    faltantes = []
    for clase in codificacion["clases"]:
        indices = indices_por_clase[clase]
        exceso = max(0, len(indices) - cuotas[clase])
        if exceso:
            indices_reemplazables.extend(indices[-exceso:])
        faltantes.extend([clase] * max(0, cuotas[clase] - len(indices)))

    for indice, clase_objetivo in zip(indices_reemplazables, faltantes):
        matriz[indice] = generar_regla_binaria_aleatoria_con_cobertura(
            codificacion=codificacion,
            df_discretizado=df_discretizado,
            pertenencias_fuzzy=pertenencias_fuzzy,
            clase_objetivo=clase_objetivo,
        )


def regla_tiene_cobertura(regla_codificada, codificacion, df_discretizado, pertenencias_fuzzy):
    return (
        regla_tiene_cobertura_discreta(regla_codificada, codificacion, df_discretizado)
        and regla_tiene_cobertura_fuzzy(regla_codificada, codificacion, pertenencias_fuzzy)
        and regla_tiene_cobertura_de_su_clase(regla_codificada, codificacion, df_discretizado, pertenencias_fuzzy)
    )


def regla_tiene_cobertura_discreta(regla_codificada, codificacion, df_discretizado):
    regla_decodificada = decodificar_regla_binaria(regla_codificada, codificacion)
    if regla_decodificada is None:
        return False

    antecedentes, _ = regla_decodificada
    cubiertos = df_discretizado
    for variable, categoria in antecedentes:
        cubiertos = cubiertos[cubiertos[variable] == categoria]
        if cubiertos.empty:
            return False
    return True


def regla_tiene_cobertura_fuzzy(regla_codificada, codificacion, pertenencias_fuzzy):
    regla_decodificada = decodificar_regla_binaria(regla_codificada, codificacion)
    if regla_decodificada is None:
        return False

    activacion = calcular_activacion_fuzzy_regla(regla_codificada, codificacion, pertenencias_fuzzy)
    return bool(np.any(activacion > 0.0))


def regla_tiene_cobertura_de_su_clase(regla_codificada, codificacion, df_discretizado, pertenencias_fuzzy):
    regla_decodificada = decodificar_regla_binaria(regla_codificada, codificacion)
    if regla_decodificada is None:
        return False

    _, clase = regla_decodificada
    activacion = calcular_activacion_fuzzy_regla(regla_codificada, codificacion, pertenencias_fuzzy)
    clases_reales = df_discretizado["riesgo"].to_numpy(dtype=object)
    mascara_clase = clases_reales == clase
    if not np.any(mascara_clase):
        return False
    return bool(np.any(activacion[mascara_clase] > 0.0))


def calcular_activacion_fuzzy_regla(regla_codificada, codificacion, pertenencias_fuzzy):
    regla_decodificada = decodificar_regla_binaria(regla_codificada, codificacion)
    if regla_decodificada is None:
        cantidad = len(next(iter(next(iter(pertenencias_fuzzy.values())).values())))
        return np.zeros(cantidad, dtype=float)

    antecedentes, _ = regla_decodificada
    cantidad = len(next(iter(next(iter(pertenencias_fuzzy.values())).values())))
    activacion = np.ones(cantidad, dtype=float)
    for variable, categoria in antecedentes:
        activacion = np.minimum(activacion, pertenencias_fuzzy[variable][categoria])
    return activacion


def construir_pertenencias_fuzzy(tabla, membresias):
    datos = convertir_split_a_diccionario(tabla)
    pertenencias = {}
    for variable in VARIABLES_ENTRADA:
        minimo, maximo = ESPECIFICACIONES_VARIABLES[variable]["limites"]
        universo = np.linspace(minimo, maximo, PUNTOS_GRAFICA)
        valores = np.asarray(datos["entradas"][variable], dtype=float)
        pertenencias[variable] = {}
        for categoria, puntos in membresias[variable].items():
            curva = fuzz.trapmf(universo, puntos)
            pertenencias[variable][categoria] = np.asarray(
                [fuzz.interp_membership(universo, curva, valor) for valor in valores],
                dtype=float,
            )
    return pertenencias


def decodificar_regla_binaria(regla_codificada, codificacion):
    regla_codificada = np.asarray(regla_codificada, dtype=int)
    antecedentes = []
    for indice_variable, variable in enumerate(VARIABLES_ENTRADA):
        inicio = indice_variable * BITS_POR_CAMPO
        fin = inicio + BITS_POR_CAMPO
        indice_categoria = bits_a_indice(regla_codificada[inicio:fin])
        categorias = codificacion["categorias_por_variable"][variable]
        if indice_categoria >= len(categorias):
            return None
        antecedentes.append((variable, categorias[indice_categoria]))

    inicio_clase = len(VARIABLES_ENTRADA) * BITS_POR_CAMPO
    indice_clase = bits_a_indice(regla_codificada[inicio_clase : inicio_clase + BITS_POR_CAMPO])
    if indice_clase >= len(codificacion["clases"]):
        return None
    return antecedentes, codificacion["clases"][indice_clase]


def decodificar_clase_regla(regla_codificada, codificacion):
    regla_codificada = np.asarray(regla_codificada, dtype=int)
    inicio_clase = len(VARIABLES_ENTRADA) * BITS_POR_CAMPO
    indice_clase = bits_a_indice(regla_codificada[inicio_clase : inicio_clase + BITS_POR_CAMPO])
    if indice_clase >= len(codificacion["clases"]):
        return None
    return codificacion["clases"][indice_clase]


def indice_a_bits(indice):
    return [int(bit) for bit in f"{int(indice):0{BITS_POR_CAMPO}b}"]


def bits_a_indice(bits):
    return int("".join(str(int(bit)) for bit in bits), 2)


def balanced_accuracy_con_sin_activacion_invalida(reales, predichos, sin_activacion):
    reales = np.asarray(reales)
    predichos = np.asarray(predichos, dtype=object).copy()
    predichos[np.asarray(sin_activacion, dtype=bool)] = "__sin_activacion__"

    recalls = []
    for clase in ETIQUETAS_RIESGO:
        mascara_clase = reales == clase
        total_clase = int(np.sum(mascara_clase))
        if total_clase == 0:
            continue
        verdaderos_positivos = int(np.sum(predichos[mascara_clase] == clase))
        recalls.append(verdaderos_positivos / total_clase)
    return float(np.mean(recalls)) if recalls else 0.0

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
