"""AG Michigan binario: un individuo representa una regla difusa.

Cada cromosoma contiene solo antecedentes:
    6 variables * 3 bits fijos = 18 bits.

El consecuente se asigna aleatoriamente al crear la regla y no es mutado ni
cruzado por los operadores geneticos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
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
    """Evoluciona una poblacion de reglas binarias validas."""
    df_discretizado = pd.DataFrame(_discretizar(tabla)).rename(columns={"clase": "riesgo"})
    codificacion = construir_codificacion()
    generaciones_sin_mejora = 0
    historial = []

    poblacion, descartadas = generar_poblacion_valida(
        cantidad=parametros["reglas_por_poblacion"],
        codificacion=codificacion,
    )
    descartadas_generacion = descartadas

    mejor_resultado = None

    for generacion in range(0, parametros["maximo_generaciones"] + 1):
        evaluaciones = evaluar_poblacion_local(
            poblacion=poblacion,
            df_discretizado=df_discretizado,
            codificacion=codificacion,
            penalizacion_error=float(parametros["penalizacion_error_regla"]),
        )
        reglas = decodificar_poblacion(poblacion, codificacion)
        balanced_accuracy = evaluar_balanced_accuracy_global(reglas, tabla, membresias)
        fila = construir_fila_historial(
            generacion=generacion,
            evaluaciones=evaluaciones,
            balanced_accuracy=balanced_accuracy,
            invalidas_descartadas=descartadas_generacion,
        )
        historial.append(fila)
        print(
            f"  AG-MB  | gen={generacion:04d} "
            f"ba={balanced_accuracy:.4f} "
            f"fitness_promedio_regla={fila['fitness_promedio_reglas']:.4f} "
            f"descartes_invalidos_gen={descartadas_generacion}"
        )
        if progress_callback is not None:
            progress_callback(fila)

        if mejor_resultado is None or balanced_accuracy > mejor_resultado.balanced_accuracy:
            mejor_resultado = ResultadoMichiganBinario(
                individuos=clonar_poblacion(poblacion),
                reglas=reglas,
                balanced_accuracy=float(balanced_accuracy),
                fitness_promedio_reglas=float(fila["fitness_promedio_reglas"]),
                fitness_mejor_regla=float(fila["fitness_mejor_regla"]),
                intentos_invalidos_descartados_generacion=int(descartadas_generacion),
                generacion_mejor=int(generacion),
            )
            generaciones_sin_mejora = 0
        else:
            generaciones_sin_mejora += 1

        if generacion >= parametros["maximo_generaciones"]:
            break
        if generaciones_sin_mejora >= parametros["paciencia"]:
            break

        poblacion, descartadas = reproducir_poblacion(
            poblacion=poblacion,
            evaluaciones=evaluaciones,
            parametros=parametros,
            codificacion=codificacion,
        )
        descartadas_generacion = descartadas

    return mejor_resultado, pd.DataFrame(historial)


def construir_codificacion():
    categorias_por_variable = {
        variable: list(ESPECIFICACIONES_VARIABLES[variable]["categorias"].keys())
        for variable in VARIABLES_ENTRADA
    }
    return {
        "categorias_por_variable": categorias_por_variable,
        "clases": list(ETIQUETAS_RIESGO),
    }


def generar_poblacion_valida(cantidad, codificacion):
    poblacion = []
    descartadas = 0
    while len(poblacion) < cantidad:
        individuo, invalidas = generar_individuo_valido(codificacion)
        poblacion.append(individuo)
        descartadas += invalidas
    return poblacion, descartadas


def generar_individuo_valido(codificacion, max_intentos=10000):
    descartadas = 0
    for _ in range(max_intentos):
        cromosoma = np.random.randint(0, 2, size=BITS_POR_REGLA, dtype=int)
        clase = str(np.random.choice(codificacion["clases"]))
        individuo = IndividuoRegla(cromosoma=cromosoma, clase=clase)
        if decodificar_antecedentes(individuo.cromosoma, codificacion) is not None:
            return individuo, descartadas
        descartadas += 1

    return generar_individuo_valido_directo(codificacion), descartadas


def generar_individuo_valido_directo(codificacion):
    bits = []
    for variable in VARIABLES_ENTRADA:
        categorias = codificacion["categorias_por_variable"][variable]
        indice = int(np.random.randint(0, len(categorias)))
        bits.extend(indice_a_bits(indice))
    clase = str(np.random.choice(codificacion["clases"]))
    return IndividuoRegla(cromosoma=np.asarray(bits, dtype=int), clase=clase)


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


def evaluar_poblacion_local(poblacion, df_discretizado, codificacion, penalizacion_error):
    evaluaciones = []
    for individuo in poblacion:
        antecedentes = decodificar_antecedentes(individuo.cromosoma, codificacion)
        if antecedentes is None:
            evaluaciones.append(EvaluacionRegla(FITNESS_MINIMO_RULETA, 0, 0, 0))
            continue

        cubiertos = df_discretizado
        for variable, categoria in antecedentes:
            cubiertos = cubiertos[cubiertos[variable] == categoria]

        cobertura = int(len(cubiertos))
        aciertos = int((cubiertos["riesgo"] == individuo.clase).sum()) if cobertura else 0
        errores = int(cobertura - aciertos)
        fitness_crudo = aciertos - penalizacion_error * errores
        fitness = max(FITNESS_MINIMO_RULETA, float(fitness_crudo))
        evaluaciones.append(
            EvaluacionRegla(
                fitness=fitness,
                aciertos=aciertos,
                errores=errores,
                cobertura=cobertura,
            )
        )
    return evaluaciones


def evaluar_balanced_accuracy_global(reglas, tabla, membresias):
    datos = convertir_split_a_diccionario(tabla)
    sistema = SistemaDifusoMamdani(membresias, reglas=reglas)
    inferencia = sistema.inferir_lote(datos["entradas"])
    return float(balanced_accuracy_score(datos["riesgos"], inferencia["riesgos"]))


def reproducir_poblacion(poblacion, evaluaciones, parametros, codificacion):
    elitismo = int(parametros["elitismo"])
    cantidad = int(parametros["reglas_por_poblacion"])
    cantidad_padres = normalizar_cantidad_padres(parametros)
    padres = [seleccionar_ruleta(poblacion, evaluaciones) for _ in range(cantidad_padres)]
    nueva = seleccionar_elite(poblacion, evaluaciones, elitismo)
    descartadas = 0

    while len(nueva) < cantidad:
        padre_a = padres[int(np.random.randint(0, len(padres)))]
        padre_b = padres[int(np.random.randint(0, len(padres)))]
        hijo_bits = cruzar_single_point(
            padre_a.cromosoma,
            padre_b.cromosoma,
            probabilidad_cruce=float(parametros["probabilidad_cruce"]),
        )
        hijo_bits = mutar_bit_flip(
            hijo_bits,
            probabilidad_mutacion=float(parametros["probabilidad_mutacion"]),
        )
        clase = padre_a.clase if np.random.random() < 0.5 else padre_b.clase
        hijo = IndividuoRegla(cromosoma=hijo_bits, clase=clase)
        if decodificar_antecedentes(hijo.cromosoma, codificacion) is None:
            descartadas += 1
            hijo, extra_descartadas = generar_individuo_valido(codificacion)
            descartadas += extra_descartadas
        nueva.append(hijo)

    return nueva, descartadas


def normalizar_cantidad_padres(parametros):
    cantidad = int(parametros["cantidad_padres"])
    poblacion = int(parametros["reglas_por_poblacion"])
    if cantidad < 2:
        raise ValueError("cantidad_padres debe ser al menos 2.")
    if cantidad > poblacion:
        raise ValueError("cantidad_padres no puede superar reglas_por_poblacion.")
    return cantidad


def seleccionar_elite(poblacion, evaluaciones, elitismo):
    if elitismo <= 0:
        return []
    indices = np.argsort([evaluacion.fitness for evaluacion in evaluaciones])[::-1]
    return [clonar_individuo(poblacion[int(indice)]) for indice in indices[:elitismo]]


def seleccionar_ruleta(poblacion, evaluaciones):
    fitness = np.asarray([evaluacion.fitness for evaluacion in evaluaciones], dtype=float)
    total = float(np.sum(fitness))
    if total <= 0:
        indice = int(np.random.randint(0, len(poblacion)))
    else:
        probabilidades = fitness / total
        indice = int(np.random.choice(np.arange(len(poblacion)), p=probabilidades))
    return poblacion[indice]


def cruzar_single_point(bits_a, bits_b, probabilidad_cruce):
    if np.random.random() >= probabilidad_cruce:
        return np.asarray(bits_a, dtype=int).copy()
    punto = int(np.random.randint(1, BITS_POR_REGLA))
    return np.concatenate([bits_a[:punto], bits_b[punto:]]).astype(int)


def mutar_bit_flip(cromosoma, probabilidad_mutacion):
    cromosoma = np.asarray(cromosoma, dtype=int).copy()
    mascara = np.random.random(size=cromosoma.shape[0]) < probabilidad_mutacion
    cromosoma[mascara] = 1 - cromosoma[mascara]
    return cromosoma


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


def cromosoma_a_texto(cromosoma):
    return "".join(str(int(bit)) for bit in cromosoma)


def clonar_individuo(individuo):
    return IndividuoRegla(cromosoma=np.asarray(individuo.cromosoma, dtype=int).copy(), clase=individuo.clase)


def clonar_poblacion(poblacion):
    return [clonar_individuo(individuo) for individuo in poblacion]


def contar_duplicados(reglas):
    claves = []
    for regla in reglas:
        claves.append((tuple(regla["antecedentes"]), regla["consecuente"]))
    return len(claves) - len(set(claves))
