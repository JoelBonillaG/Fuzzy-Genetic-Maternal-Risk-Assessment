"""Algoritmo genetico binario para reglas difusas."""

from .algoritmo import (
    BITS_POR_GEN,
    BITS_POR_CONSECUENTE,
    BITS_POR_REGLA,
    ResultadoAlgoritmoGenetico,
    contar_duplicados,
    ejecutar_algoritmo_genetico,
)

__all__ = [
    "BITS_POR_GEN",
    "BITS_POR_CONSECUENTE",
    "BITS_POR_REGLA",
    "ResultadoAlgoritmoGenetico",
    "contar_duplicados",
    "ejecutar_algoritmo_genetico",
]
