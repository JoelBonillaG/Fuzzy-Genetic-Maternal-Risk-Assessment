"""AG Michigan binario con PyGAD.

Cada individuo de PyGAD representa una regla difusa:
    6 antecedentes * 3 bits fijos = 18 bits.

El consecuente se asigna aleatoriamente por posicion de la poblacion y no forma
parte del cromosoma; por tanto, cruce y mutacion solo modifican antecedentes.
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


BITS_POR_GEN = 3
BITS_POR_REGLA = len(VARIABLES_ENTRADA) * BITS_POR_GEN
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
    cantidad_reglas = int(parametros["reglas_por_poblacion"])
    cantidad_padres = normalizar_cantidad_padres(parametros)

    poblacion_inicial, clases_fijas, descartes_iniciales = inicializar_poblacion_pygad(
        cantidad=cantidad_reglas,
        codificacion=codificacion,
    )

    estado = {
        "descartes_generacion": int(descartes_iniciales),
        "mejor_resultado": None,
        "generaciones_sin_mejora": 0,
        "historial": [],
    }

    def fitness_func(instancia_ga, solucion, indice_solucion):
        individuo = IndividuoRegla(
            cromosoma=np.asarray(solucion, dtype=int),
            clase=clases_fijas[int(indice_solucion)],
        )
        evaluacion = evaluar_regla_local(
            individuo=individuo,
            df_discretizado=df_discretizado,
            codificacion=codificacion,
            penalizacion_error=float(parametros["penalizacion_error_regla"]),
        )
        return evaluacion.fitness

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
            mascara = np.random.random(size=BITS_POR_REGLA) < float(parametros["probabilidad_mutacion"])
            descendencia[indice_hijo, mascara] = 1 - descendencia[indice_hijo, mascara]
            if decodificar_antecedentes(descendencia[indice_hijo], codificacion) is None:
                descartes += 1
                descendencia[indice_hijo], extra_descartes = generar_cromosoma_valido(codificacion)
                descartes += extra_descartes
        estado["descartes_generacion"] = int(descartes)
        return descendencia

    def registrar_generacion(generacion, poblacion):
        evaluaciones = evaluar_poblacion_local(
            poblacion=poblacion,
            clases_fijas=clases_fijas,
            df_discretizado=df_discretizado,
            codificacion=codificacion,
            penalizacion_error=float(parametros["penalizacion_error_regla"]),
        )
        individuos = construir_individuos(poblacion, clases_fijas)
        reglas = decodificar_poblacion(individuos, codificacion)
        balanced_accuracy = evaluar_balanced_accuracy_global(reglas, tabla, membresias)
        fila = construir_fila_historial(
            generacion=generacion,
            evaluaciones=evaluaciones,
            balanced_accuracy=balanced_accuracy,
            invalidas_descartadas=estado["descartes_generacion"],
        )
        estado["historial"].append(fila)
        print(
            f"  AG-MB  | gen={generacion:04d} "
            f"ba={balanced_accuracy:.4f} "
            f"fitness_promedio_regla={fila['fitness_promedio_reglas']:.4f} "
            f"descartes_invalidos_gen={estado['descartes_generacion']}"
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

    def on_generation(instancia_ga):
        registrar_generacion(instancia_ga.generations_completed, instancia_ga.population)
        if estado["generaciones_sin_mejora"] >= int(parametros["paciencia"]):
            return "stop"
        return None

    registrar_generacion(0, poblacion_inicial)
    estado["descartes_generacion"] = 0

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
    clases = generar_clases_balanceadas(cantidad, codificacion["clases"])
    descartes = 0
    while len(poblacion) < cantidad:
        cromosoma, intentos_invalidos = generar_cromosoma_valido(codificacion)
        poblacion.append(cromosoma)
        descartes += intentos_invalidos
    return np.asarray(poblacion, dtype=int), clases, descartes


def generar_clases_balanceadas(cantidad, clases):
    clases = list(clases)
    base = cantidad // len(clases)
    sobrantes = cantidad % len(clases)
    salida = []
    for indice, clase in enumerate(clases):
        repeticiones = base + (1 if indice < sobrantes else 0)
        salida.extend([str(clase)] * repeticiones)
    np.random.shuffle(salida)
    return salida


def generar_cromosoma_valido(codificacion, max_intentos=10000):
    descartes = 0
    for _ in range(max_intentos):
        cromosoma = np.random.randint(0, 2, size=BITS_POR_REGLA, dtype=int)
        if decodificar_antecedentes(cromosoma, codificacion) is not None:
            return cromosoma, descartes
        descartes += 1
    return generar_cromosoma_valido_directo(codificacion), descartes


def generar_cromosoma_valido_directo(codificacion):
    bits = []
    for variable in VARIABLES_ENTRADA:
        categorias = codificacion["categorias_por_variable"][variable]
        indice = int(np.random.randint(0, len(categorias)))
        bits.extend(indice_a_bits(indice))
    return np.asarray(bits, dtype=int)


def indice_a_bits(indice):
    return [int(bit) for bit in f"{indice:0{BITS_POR_GEN}b}"]


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


def bits_a_indice(bits):
    return int("".join(str(int(bit)) for bit in bits), 2)


def construir_individuos(poblacion, clases_fijas):
    return [
        IndividuoRegla(cromosoma=np.asarray(cromosoma, dtype=int).copy(), clase=clases_fijas[indice])
        for indice, cromosoma in enumerate(poblacion)
    ]


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


def evaluar_poblacion_local(poblacion, clases_fijas, df_discretizado, codificacion, penalizacion_error):
    evaluaciones = []
    for indice, cromosoma in enumerate(poblacion):
        individuo = IndividuoRegla(
            cromosoma=np.asarray(cromosoma, dtype=int),
            clase=clases_fijas[indice],
        )
        evaluaciones.append(
            evaluar_regla_local(individuo, df_discretizado, codificacion, penalizacion_error)
        )
    return evaluaciones


def evaluar_regla_local(individuo, df_discretizado, codificacion, penalizacion_error):
    antecedentes = decodificar_antecedentes(individuo.cromosoma, codificacion)
    if antecedentes is None:
        return EvaluacionRegla(FITNESS_MINIMO_RULETA, 0, 0, 0)

    cubiertos = df_discretizado
    for variable, categoria in antecedentes:
        cubiertos = cubiertos[cubiertos[variable] == categoria]

    cobertura = int(len(cubiertos))
    aciertos = int((cubiertos["riesgo"] == individuo.clase).sum()) if cobertura else 0
    errores = int(cobertura - aciertos)
    fitness_crudo = aciertos - penalizacion_error * errores
    fitness = max(FITNESS_MINIMO_RULETA, float(fitness_crudo))
    return EvaluacionRegla(
        fitness=fitness,
        aciertos=aciertos,
        errores=errores,
        cobertura=cobertura,
    )


def evaluar_balanced_accuracy_global(reglas, tabla, membresias):
    datos = convertir_split_a_diccionario(tabla)
    sistema = SistemaDifusoMamdani(membresias, reglas=reglas)
    inferencia = sistema.inferir_lote(datos["entradas"])
    return float(balanced_accuracy_score(datos["riesgos"], inferencia["riesgos"]))


def normalizar_cantidad_padres(parametros):
    cantidad = int(parametros["cantidad_padres"])
    poblacion = int(parametros["reglas_por_poblacion"])
    if cantidad < 2:
        raise ValueError("cantidad_padres debe ser al menos 2.")
    if cantidad > poblacion:
        raise ValueError("cantidad_padres no puede superar reglas_por_poblacion.")
    return cantidad


def construir_fila_historial(generacion, evaluaciones, balanced_accuracy, invalidas_descartadas):
    fitness = [evaluacion.fitness for evaluacion in evaluaciones]
    aciertos = [evaluacion.aciertos for evaluacion in evaluaciones]
    errores = [evaluacion.errores for evaluacion in evaluaciones]
    return {
        "generacion": int(generacion),
        "balanced_accuracy_global": float(balanced_accuracy),
        "fitness_promedio_reglas": float(np.mean(fitness)) if fitness else 0.0,
        "fitness_mejor_regla": float(np.max(fitness)) if fitness else 0.0,
        "aciertos_promedio_regla": float(np.mean(aciertos)) if aciertos else 0.0,
        "errores_promedio_regla": float(np.mean(errores)) if errores else 0.0,
        "intentos_invalidos_descartados_generacion": int(invalidas_descartadas),
    }


def clonar_individuo(individuo):
    return IndividuoRegla(
        cromosoma=np.asarray(individuo.cromosoma, dtype=int).copy(),
        clase=individuo.clase,
    )


def clonar_poblacion(poblacion):
    return [clonar_individuo(individuo) for individuo in poblacion]


def contar_duplicados(reglas):
    claves = []
    for regla in reglas:
        claves.append((tuple(regla["antecedentes"]), regla["consecuente"]))
    return len(claves) - len(set(claves))
