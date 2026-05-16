"""Cromosoma Pittsburgh: vector binario de inclusion/exclusion de reglas candidatas.

Cada bit ci indica si la regla candidata Ri esta activa (1) o no (0).
Un cromosoma representa una base completa de reglas.
"""

import numpy as np

from ...logica_difusa.reglas import REGLAS


CANTIDAD_REGLAS_CANDIDATAS = len(REGLAS)


def cromosoma_todas_activas():
    """Cromosoma con todas las reglas candidatas activas (Cfull = [1, 1, ..., 1])."""
    return np.ones(CANTIDAD_REGLAS_CANDIDATAS, dtype=int)


def cromosoma_aleatorio():
    """Cromosoma binario aleatorio: cada bit es 1 con probabilidad 0.5."""
    return np.random.randint(0, 2, size=CANTIDAD_REGLAS_CANDIDATAS).astype(int)


def seleccion_a_base_reglas(cromosoma, reglas_candidatas=REGLAS):
    """Decodifica un cromosoma binario en la lista de reglas activas (la base S).

    Entrada:
        cromosoma: array de ints [1, 0, 1, 1, 0, ...]  — un bit por regla candidata
    Salida:
        lista de dicts con formato:
        {"numero": int, "antecedentes": [(variable, categoria), ...], "consecuente": str}
        Solo se incluyen las reglas donde el bit es 1.
    """
    cromosoma = np.asarray(cromosoma, dtype=int)
    base = []
    for i, bit in enumerate(cromosoma):
        if bit == 1:
            base.append(reglas_candidatas[i])
    return base


def es_base_vacia(cromosoma):
    """True si ningun bit esta activo — la base es vacia."""
    return cantidad_reglas_activas(cromosoma) == 0


def cantidad_reglas_activas(cromosoma):
    """Numero de reglas activas |S| en el cromosoma."""
    return int(np.sum(np.asarray(cromosoma, dtype=int)))


def numeros_reglas_activas(cromosoma, reglas_candidatas=REGLAS):
    """Numeros legibles ('numero' del JSON) de las reglas activas."""
    numeros = []
    for i in indices_reglas_activas(cromosoma):
        numeros.append(reglas_candidatas[i]["numero"])
    return numeros


def indices_reglas_activas(cromosoma):
    """Indices (base 0) de las reglas activas en el cromosoma."""
    cromosoma = np.asarray(cromosoma, dtype=int)
    return np.where(cromosoma == 1)[0].tolist()
