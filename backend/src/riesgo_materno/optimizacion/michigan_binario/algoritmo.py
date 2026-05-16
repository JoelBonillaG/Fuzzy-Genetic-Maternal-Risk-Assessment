"""AG Michigan binario con PyGAD.

Cada individuo de PyGAD representa una regla difusa:
    6 antecedentes * 3 bits fijos + 3 bits de consecuente = 21 bits.

La poblacion completa funciona como base de reglas. Cada regla conserva sus
antecedentes y su consecuente dentro del cromosoma. La inicializacion es
aleatoria y se reparan cromosomas invalidos o sin cobertura fuzzy.
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


BITS_POR_GEN = 3
BITS_POR_CONSECUENTE = 3
BITS_ANTECEDENTES = len(VARIABLES_ENTRADA) * BITS_POR_GEN
BITS_POR_REGLA = BITS_ANTECEDENTES + BITS_POR_CONSECUENTE
FITNESS_MINIMO_RULETA = 1e-6
CLASE_A_CONSECUENTE = {"low risk": "bajo", "mid risk": "medio", "high risk": "alto"}


@dataclass
class IndividuoRegla:
    cromosoma: np.ndarray
    clase: str


@dataclass
class EvaluacionRegla:
    fitness: float
    aciertos: int
    errores: int
    cobertura: int
    calidad_local: float = 0.0
    aporte_clase: float = 0.0
    confusion_otras_clases: float = 0.0
    penalizacion_duplicado: float = 0.0


@dataclass
class ResultadoMichiganBinario:
    individuos: list[IndividuoRegla]
    reglas: list[dict]
    balanced_accuracy: float
    fitness_promedio_reglas: float
    fitness_mejor_regla: float
    intentos_invalidos_descartados_generacion: int
    generacion_mejor: int


def ejecutar_ag_michigan_binario(tabla, membresias, parametros, progress_callback=None):
    """Evoluciona una poblacion de reglas binarias validas usando PyGAD."""
    df_discretizado = pd.DataFrame(_discretizar(tabla)).rename(columns={"clase": "riesgo"})
    codificacion = construir_codificacion()
    pertenencias_fuzzy = construir_pertenencias_fuzzy(tabla, membresias)
    cantidad_reglas = int(parametros["reglas_por_poblacion"])
    cantidad_padres = normalizar_cantidad_padres(parametros)

    poblacion_inicial, descartes_iniciales = inicializar_poblacion_pygad(
        cantidad=cantidad_reglas,
        codificacion=codificacion,
    )
    poblacion_inicial, reparadas_iniciales = reparar_reglas_sin_cobertura(
        poblacion=poblacion_inicial,
        codificacion=codificacion,
        pertenencias_fuzzy=pertenencias_fuzzy,
        df_discretizado=df_discretizado,
    )

    estado = {
        "descartes_generacion": int(descartes_iniciales),
        "reparadas_sin_cobertura_generacion": int(reparadas_iniciales),
        "mejor_resultado": None,
        "generaciones_sin_mejora": 0,
        "historial": [],
        "cache_fitness": {},
    }

    def fitness_func(instancia_ga, solucion, indice_solucion):
        poblacion_actual = obtener_poblacion_actual(instancia_ga, poblacion_inicial)
        clave = poblacion_actual.astype(int).tobytes()
        if clave not in estado["cache_fitness"]:
            estado["cache_fitness"][clave] = evaluar_poblacion_por_recall_precision(
                poblacion=poblacion_actual,
                codificacion=codificacion,
                df_discretizado=df_discretizado,
                pertenencias_fuzzy=pertenencias_fuzzy,
                parametros=parametros,
            )
        return estado["cache_fitness"][clave][int(indice_solucion)].fitness

    def obtener_evaluaciones_generacion(poblacion):
        clave = np.asarray(poblacion, dtype=int).tobytes()
        if clave not in estado["cache_fitness"]:
            estado["cache_fitness"][clave] = evaluar_poblacion_por_recall_precision(
                poblacion=poblacion,
                codificacion=codificacion,
                df_discretizado=df_discretizado,
                pertenencias_fuzzy=pertenencias_fuzzy,
                parametros=parametros,
            )
        return estado["cache_fitness"][clave]

    def obtener_poblacion_actual(instancia_ga, poblacion_respaldo):
        poblacion = getattr(instancia_ga, "population", None)
        if poblacion is None:
            return np.asarray(poblacion_respaldo, dtype=int)
        return np.asarray(poblacion, dtype=int)

    def limitar_cache_fitness(maximo=3):
        while len(estado["cache_fitness"]) > maximo:
            estado["cache_fitness"].pop(next(iter(estado["cache_fitness"])))

    def crossover_single_point(padres, tamano_descendencia, instancia_ga):
        descendencia = np.empty(tamano_descendencia, dtype=int)
        for indice_hijo in range(tamano_descendencia[0]):
            padre_a = padres[indice_hijo % padres.shape[0]]
            padre_b = padres[(indice_hijo + 1) % padres.shape[0]]
            if np.random.random() >= float(parametros["probabilidad_cruce"]):
                descendencia[indice_hijo] = padre_a.copy()
                continue
            punto = int(np.random.randint(1, BITS_POR_REGLA))
            descendencia[indice_hijo] = np.concatenate([padre_a[:punto], padre_b[punto:]])
        return descendencia

    def mutacion_bit_flip(descendencia, instancia_ga):
        descendencia = np.asarray(descendencia, dtype=int).copy()
        descartes = 0
        for indice_hijo in range(descendencia.shape[0]):
            mascara = np.zeros(BITS_POR_REGLA, dtype=bool)
            mascara[:BITS_ANTECEDENTES] = (
                np.random.random(size=BITS_ANTECEDENTES) < float(parametros["probabilidad_mutacion"])
            )
            descendencia[indice_hijo, mascara] = 1 - descendencia[indice_hijo, mascara]
            if decodificar_antecedentes(descendencia[indice_hijo], codificacion) is None:
                descartes += 1
                descendencia[indice_hijo] = generar_cromosoma_valido_directo(
                    codificacion,
                )
                continue
            if decodificar_clase(descendencia[indice_hijo], codificacion) is None:
                descartes += 1
                descendencia[indice_hijo] = reparar_consecuente(
                    descendencia[indice_hijo],
                    codificacion,
                )
        estado["descartes_generacion"] = int(descartes)
        descendencia, reparadas = reparar_reglas_sin_cobertura(
            poblacion=descendencia,
            codificacion=codificacion,
            pertenencias_fuzzy=pertenencias_fuzzy,
            df_discretizado=df_discretizado,
        )
        estado["reparadas_sin_cobertura_generacion"] = int(reparadas)
        return descendencia

    def registrar_generacion(generacion, poblacion):
        poblacion = np.asarray(poblacion, dtype=int)
        evaluaciones = obtener_evaluaciones_generacion(poblacion)
        individuos = construir_individuos(poblacion, codificacion)
        reglas = decodificar_poblacion(individuos, codificacion)
        balanced_accuracy = evaluar_balanced_accuracy_global(reglas, tabla, membresias)
        fila = construir_fila_historial(
            generacion=generacion,
            evaluaciones=evaluaciones,
            balanced_accuracy=balanced_accuracy,
            invalidas_descartadas=estado["descartes_generacion"],
            reparadas_sin_cobertura=estado["reparadas_sin_cobertura_generacion"],
        )
        estado["historial"].append(fila)
        print(
            f"  AG-MB  | gen={generacion:04d} "
            f"ba={balanced_accuracy:.4f} "
            f"fitness_promedio_regla={fila['fitness_promedio_reglas']:.4f} "
            f"descartes_invalidos_gen={estado['descartes_generacion']} "
            f"reparadas_sin_cobertura={estado['reparadas_sin_cobertura_generacion']}"
        )
        if progress_callback is not None:
            progress_callback(fila)

        mejor_resultado = estado["mejor_resultado"]
        if mejor_resultado is None or balanced_accuracy > mejor_resultado.balanced_accuracy:
            estado["mejor_resultado"] = ResultadoMichiganBinario(
                individuos=clonar_poblacion(individuos),
                reglas=reglas,
                balanced_accuracy=float(balanced_accuracy),
                fitness_promedio_reglas=float(fila["fitness_promedio_reglas"]),
                fitness_mejor_regla=float(fila["fitness_mejor_regla"]),
                intentos_invalidos_descartados_generacion=int(estado["descartes_generacion"]),
                generacion_mejor=int(generacion),
            )
            estado["generaciones_sin_mejora"] = 0
        else:
            estado["generaciones_sin_mejora"] += 1
        limitar_cache_fitness()

    def on_generation(instancia_ga):
        poblacion_reparada, reparadas = reparar_reglas_sin_cobertura(
            poblacion=instancia_ga.population,
            codificacion=codificacion,
            pertenencias_fuzzy=pertenencias_fuzzy,
            df_discretizado=df_discretizado,
        )
        instancia_ga.population = poblacion_reparada
        estado["reparadas_sin_cobertura_generacion"] = int(reparadas)
        registrar_generacion(instancia_ga.generations_completed, instancia_ga.population)
        if estado["generaciones_sin_mejora"] >= int(parametros["paciencia"]):
            return "stop"
        return None

    registrar_generacion(0, poblacion_inicial)
    estado["descartes_generacion"] = 0
    estado["reparadas_sin_cobertura_generacion"] = 0

    instancia_ga = pygad.GA(
        initial_population=poblacion_inicial,
        num_parents_mating=cantidad_padres,
        fitness_func=fitness_func,
        num_generations=int(parametros["maximo_generaciones"]),
        parent_selection_type="rws",
        keep_elitism=int(parametros["elitismo"]),
        crossover_type=crossover_single_point,
        mutation_type=mutacion_bit_flip,
        gene_type=int,
        gene_space=[0, 1],
        on_generation=on_generation,
        save_solutions=False,
        suppress_warnings=True,
    )
    instancia_ga.run()

    return estado["mejor_resultado"], pd.DataFrame(estado["historial"])


def construir_codificacion():
    categorias_por_variable = {
        variable: list(ESPECIFICACIONES_VARIABLES[variable]["categorias"].keys())
        for variable in VARIABLES_ENTRADA
    }
    return {
        "categorias_por_variable": categorias_por_variable,
        "clases": list(ETIQUETAS_RIESGO),
    }


def inicializar_poblacion_pygad(cantidad, codificacion):
    poblacion = []
    descartes = 0
    while len(poblacion) < cantidad:
        cromosoma = generar_cromosoma_valido_directo(codificacion)
        poblacion.append(cromosoma)
    np.random.shuffle(poblacion)
    return np.asarray(poblacion, dtype=int), descartes


def generar_cromosoma_valido_directo(codificacion):
    bits = []
    for variable in VARIABLES_ENTRADA:
        categorias = codificacion["categorias_por_variable"][variable]
        indice = int(np.random.randint(0, len(categorias)))
        bits.extend(indice_a_bits(indice))
    clase = str(np.random.choice(codificacion["clases"]))
    bits.extend(indice_a_bits_clase(codificacion["clases"].index(clase)))
    return np.asarray(bits, dtype=int)


def indice_a_bits(indice):
    return [int(bit) for bit in f"{indice:0{BITS_POR_GEN}b}"]


def indice_a_bits_clase(indice):
    return [int(bit) for bit in f"{indice:0{BITS_POR_CONSECUENTE}b}"]


def decodificar_antecedentes(cromosoma, codificacion):
    antecedentes = []
    for indice_variable, variable in enumerate(VARIABLES_ENTRADA):
        inicio = indice_variable * BITS_POR_GEN
        bloque = cromosoma[inicio : inicio + BITS_POR_GEN]
        indice_categoria = bits_a_indice(bloque)
        categorias = codificacion["categorias_por_variable"][variable]
        if indice_categoria >= len(categorias):
            return None
        antecedentes.append((variable, categorias[indice_categoria]))
    return antecedentes


def decodificar_clase(cromosoma, codificacion):
    bloque = cromosoma[BITS_ANTECEDENTES:BITS_POR_REGLA]
    indice_clase = bits_a_indice(bloque)
    clases = codificacion["clases"]
    if indice_clase >= len(clases):
        return None
    return clases[indice_clase]


def bits_a_indice(bits):
    return int("".join(str(int(bit)) for bit in bits), 2)


def construir_individuos(poblacion, codificacion):
    individuos = []
    for cromosoma in poblacion:
        cromosoma = np.asarray(cromosoma, dtype=int).copy()
        clase = decodificar_clase(cromosoma, codificacion)
        if clase is None:
            continue
        individuos.append(IndividuoRegla(cromosoma=cromosoma, clase=clase))
    return individuos


def decodificar_poblacion(poblacion, codificacion):
    reglas = []
    for numero, individuo in enumerate(poblacion, start=1):
        antecedentes = decodificar_antecedentes(individuo.cromosoma, codificacion)
        if antecedentes is None:
            continue
        reglas.append(
            {
                "numero": numero,
                "antecedentes": antecedentes,
                "consecuente": CLASE_A_CONSECUENTE[individuo.clase],
                "source": "AG_MICHIGAN_BINARIO",
            }
        )
    return reglas


def evaluar_poblacion_por_recall_precision(
    poblacion,
    codificacion,
    df_discretizado,
    pertenencias_fuzzy=None,
    parametros=None,
):
    """Evalua cada regla con fitness local o compuesto, segun parametros."""
    evaluaciones = evaluar_poblacion_local(
        poblacion=poblacion,
        df_discretizado=df_discretizado,
        codificacion=codificacion,
    )
    parametros = parametros or {}
    if not parametros.get("usar_fitness_compuesto", False):
        return evaluaciones
    if pertenencias_fuzzy is None:
        return evaluaciones

    return aplicar_fitness_compuesto(
        poblacion=poblacion,
        evaluaciones=evaluaciones,
        codificacion=codificacion,
        df_discretizado=df_discretizado,
        pertenencias_fuzzy=pertenencias_fuzzy,
        parametros=parametros,
    )


def evaluar_poblacion_local(poblacion, df_discretizado, codificacion):
    evaluaciones = []
    for cromosoma in poblacion:
        clase = decodificar_clase(cromosoma, codificacion)
        if clase is None:
            evaluaciones.append(EvaluacionRegla(FITNESS_MINIMO_RULETA, 0, 0, 0))
            continue
        individuo = IndividuoRegla(
            cromosoma=np.asarray(cromosoma, dtype=int),
            clase=clase,
        )
        evaluaciones.append(
            evaluar_regla_local(individuo, df_discretizado, codificacion)
        )
    return evaluaciones


def construir_pertenencias_fuzzy(tabla, membresias):
    """Precalcula pertenencias difusas para detectar reglas sin cobertura."""
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


def regla_tiene_cobertura_fuzzy(cromosoma, codificacion, pertenencias_fuzzy):
    antecedentes = decodificar_antecedentes(cromosoma, codificacion)
    if antecedentes is None:
        return False

    cantidad = len(next(iter(next(iter(pertenencias_fuzzy.values())).values())))
    activacion = np.ones(cantidad, dtype=float)
    for variable, categoria in antecedentes:
        activacion = np.minimum(activacion, pertenencias_fuzzy[variable][categoria])
    return bool(np.any(activacion > 0.0))


def regla_tiene_cobertura_discreta(cromosoma, codificacion, df_discretizado):
    antecedentes = decodificar_antecedentes(cromosoma, codificacion)
    if antecedentes is None:
        return False

    cubiertos = df_discretizado
    for variable, categoria in antecedentes:
        cubiertos = cubiertos[cubiertos[variable] == categoria]
        if cubiertos.empty:
            return False
    return True


def regla_tiene_cobertura(cromosoma, codificacion, pertenencias_fuzzy, df_discretizado):
    return (
        regla_tiene_cobertura_fuzzy(cromosoma, codificacion, pertenencias_fuzzy)
        and regla_tiene_cobertura_discreta(cromosoma, codificacion, df_discretizado)
    )


def calcular_activacion_fuzzy_regla(cromosoma, codificacion, pertenencias_fuzzy):
    antecedentes = decodificar_antecedentes(cromosoma, codificacion)
    if antecedentes is None:
        cantidad = len(next(iter(next(iter(pertenencias_fuzzy.values())).values())))
        return np.zeros(cantidad, dtype=float)

    cantidad = len(next(iter(next(iter(pertenencias_fuzzy.values())).values())))
    activacion = np.ones(cantidad, dtype=float)
    for variable, categoria in antecedentes:
        activacion = np.minimum(activacion, pertenencias_fuzzy[variable][categoria])
    return activacion


def aplicar_fitness_compuesto(
    poblacion,
    evaluaciones,
    codificacion,
    df_discretizado,
    pertenencias_fuzzy,
    parametros,
):
    """Alinea el fitness individual con la separacion fuzzy global por clase."""
    peso_local = float(parametros.get("peso_calidad_local", 0.45))
    peso_aporte = float(parametros.get("peso_aporte_clase", 0.35))
    peso_confusion = float(parametros.get("peso_confusion_otras_clases", 0.15))
    peso_duplicado = float(parametros.get("peso_penalizacion_duplicado", 0.005))

    clases_reales = df_discretizado["riesgo"].to_numpy(dtype=object)
    conteo_duplicados = contar_claves_cromosomas(poblacion, codificacion)
    evaluaciones_compuestas = []

    for cromosoma, evaluacion_local in zip(poblacion, evaluaciones):
        clase = decodificar_clase(cromosoma, codificacion)
        if clase is None:
            evaluaciones_compuestas.append(evaluacion_local)
            continue

        activacion = calcular_activacion_fuzzy_regla(cromosoma, codificacion, pertenencias_fuzzy)
        mascara_clase = clases_reales == clase
        mascara_otras = clases_reales != clase

        aporte_clase = float(np.mean(activacion[mascara_clase])) if np.any(mascara_clase) else 0.0
        confusion_otras = float(np.mean(activacion[mascara_otras])) if np.any(mascara_otras) else 0.0

        clave = clave_cromosoma(cromosoma, codificacion)
        repeticiones = conteo_duplicados.get(clave, 1)
        penalizacion_duplicado = float((repeticiones - 1) / repeticiones) if repeticiones > 1 else 0.0

        calidad_local = float(evaluacion_local.fitness)
        fitness = (
            peso_local * calidad_local
            + peso_aporte * aporte_clase
            - peso_confusion * confusion_otras
            - peso_duplicado * penalizacion_duplicado
        )

        evaluaciones_compuestas.append(
            EvaluacionRegla(
                fitness=max(FITNESS_MINIMO_RULETA, float(fitness)),
                aciertos=evaluacion_local.aciertos,
                errores=evaluacion_local.errores,
                cobertura=evaluacion_local.cobertura,
                calidad_local=calidad_local,
                aporte_clase=aporte_clase,
                confusion_otras_clases=confusion_otras,
                penalizacion_duplicado=penalizacion_duplicado,
            )
        )

    return evaluaciones_compuestas


def contar_claves_cromosomas(poblacion, codificacion):
    conteo = {}
    for cromosoma in poblacion:
        clave = clave_cromosoma(cromosoma, codificacion)
        conteo[clave] = conteo.get(clave, 0) + 1
    return conteo


def clave_cromosoma(cromosoma, codificacion):
    antecedentes = decodificar_antecedentes(cromosoma, codificacion)
    clase = decodificar_clase(cromosoma, codificacion)
    return (tuple(antecedentes) if antecedentes is not None else None, clase)


def generar_cromosoma_aleatorio_con_cobertura(
    codificacion,
    pertenencias_fuzzy,
    df_discretizado,
    max_intentos=10000,
):
    """Genera una regla aleatoria valida con cobertura fuzzy y discreta."""
    for _ in range(max_intentos):
        cromosoma = generar_cromosoma_valido_directo(codificacion)
        if regla_tiene_cobertura(cromosoma, codificacion, pertenencias_fuzzy, df_discretizado):
            return cromosoma
    raise RuntimeError(
        "No se pudo generar una regla aleatoria con cobertura fuzzy y discreta "
        f"despues de {max_intentos} intentos."
    )


def reparar_reglas_sin_cobertura(poblacion, codificacion, pertenencias_fuzzy, df_discretizado):
    """Reemplaza reglas sin cobertura por nuevas reglas aleatorias con cobertura."""
    poblacion = np.asarray(poblacion, dtype=int).copy()
    reparadas = 0
    for indice, cromosoma in enumerate(poblacion):
        if regla_tiene_cobertura(cromosoma, codificacion, pertenencias_fuzzy, df_discretizado):
            continue
        poblacion[indice] = generar_cromosoma_aleatorio_con_cobertura(
            codificacion,
            pertenencias_fuzzy,
            df_discretizado,
        )
        reparadas += 1
    return poblacion, reparadas


def evaluar_regla_local(individuo, df_discretizado, codificacion):
    antecedentes = decodificar_antecedentes(individuo.cromosoma, codificacion)
    if antecedentes is None:
        return EvaluacionRegla(FITNESS_MINIMO_RULETA, 0, 0, 0)

    cubiertos = df_discretizado
    for variable, categoria in antecedentes:
        cubiertos = cubiertos[cubiertos[variable] == categoria]

    cobertura = int(len(cubiertos))
    aciertos = int((cubiertos["riesgo"] == individuo.clase).sum()) if cobertura else 0
    errores = int(cobertura - aciertos)
    total_clase = int((df_discretizado["riesgo"] == individuo.clase).sum())
    recall = aciertos / total_clase if total_clase else 0.0
    precision = aciertos / cobertura if cobertura else 0.0
    fitness = max(FITNESS_MINIMO_RULETA, float(recall * precision))
    return EvaluacionRegla(
        fitness=fitness,
        aciertos=aciertos,
        errores=errores,
        cobertura=cobertura,
        calidad_local=fitness,
    )


def evaluar_balanced_accuracy_global(reglas, tabla, membresias):
    datos = convertir_split_a_diccionario(tabla)
    sistema = SistemaDifusoMamdani(membresias, reglas=reglas, permitir_neutro=False)
    inferencia = sistema.inferir_lote(datos["entradas"])
    return balanced_accuracy_con_sin_activacion_invalida(
        reales=datos["riesgos"],
        predichos=inferencia["riesgos"],
        sin_activacion=inferencia["sin_activacion"],
    )


def balanced_accuracy_con_sin_activacion_invalida(reales, predichos, sin_activacion):
    """Calcula BA contando los casos sin activacion como error, no como riesgo medio."""
    reales = np.asarray(reales)
    predichos = np.asarray(predichos, dtype=object).copy()
    sin_activacion = np.asarray(sin_activacion, dtype=bool)
    predichos[sin_activacion] = "__sin_activacion__"

    recalls = []
    for clase in ETIQUETAS_RIESGO:
        mascara_clase = reales == clase
        total_clase = int(np.sum(mascara_clase))
        if total_clase == 0:
            continue
        verdaderos_positivos = int(np.sum(predichos[mascara_clase] == clase))
        recalls.append(verdaderos_positivos / total_clase)

    return float(np.mean(recalls)) if recalls else 0.0


def normalizar_cantidad_padres(parametros):
    cantidad = int(parametros["cantidad_padres"])
    poblacion = int(parametros["reglas_por_poblacion"])
    if cantidad < 2:
        raise ValueError("cantidad_padres debe ser al menos 2.")
    if cantidad > poblacion:
        raise ValueError("cantidad_padres no puede superar reglas_por_poblacion.")
    return cantidad


def construir_fila_historial(
    generacion,
    evaluaciones,
    balanced_accuracy,
    invalidas_descartadas,
    reparadas_sin_cobertura,
):
    fitness = [evaluacion.fitness for evaluacion in evaluaciones]
    aciertos = [evaluacion.aciertos for evaluacion in evaluaciones]
    errores = [evaluacion.errores for evaluacion in evaluaciones]
    calidad_local = [evaluacion.calidad_local for evaluacion in evaluaciones]
    aporte_clase = [evaluacion.aporte_clase for evaluacion in evaluaciones]
    confusion = [evaluacion.confusion_otras_clases for evaluacion in evaluaciones]
    duplicados = [evaluacion.penalizacion_duplicado for evaluacion in evaluaciones]
    return {
        "generacion": int(generacion),
        "balanced_accuracy_global": float(balanced_accuracy),
        "fitness_promedio_reglas": float(np.mean(fitness)) if fitness else 0.0,
        "fitness_mejor_regla": float(np.max(fitness)) if fitness else 0.0,
        "calidad_local_promedio": float(np.mean(calidad_local)) if calidad_local else 0.0,
        "aporte_clase_promedio": float(np.mean(aporte_clase)) if aporte_clase else 0.0,
        "confusion_otras_clases_promedio": float(np.mean(confusion)) if confusion else 0.0,
        "penalizacion_duplicado_promedio": float(np.mean(duplicados)) if duplicados else 0.0,
        "aciertos_promedio_regla": float(np.mean(aciertos)) if aciertos else 0.0,
        "errores_promedio_regla": float(np.mean(errores)) if errores else 0.0,
        "intentos_invalidos_descartados_generacion": int(invalidas_descartadas),
        "reglas_reparadas_sin_cobertura_generacion": int(reparadas_sin_cobertura),
    }


def clonar_individuo(individuo):
    return IndividuoRegla(
        cromosoma=np.asarray(individuo.cromosoma, dtype=int).copy(),
        clase=individuo.clase,
    )


def clonar_poblacion(poblacion):
    return [clonar_individuo(individuo) for individuo in poblacion]


def reparar_consecuente(cromosoma, codificacion):
    cromosoma = np.asarray(cromosoma, dtype=int).copy()
    clase = str(np.random.choice(codificacion["clases"]))
    cromosoma[BITS_ANTECEDENTES:BITS_POR_REGLA] = indice_a_bits_clase(codificacion["clases"].index(clase))
    return cromosoma


def contar_duplicados(reglas):
    claves = []
    for regla in reglas:
        claves.append((tuple(regla["antecedentes"]), regla["consecuente"]))
    return len(claves) - len(set(claves))
