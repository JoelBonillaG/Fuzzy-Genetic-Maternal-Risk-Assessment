"""Algoritmo genetico Michigan binario para reglas difusas."""

from .algoritmo import (
    BITS_POR_GEN,
    BITS_POR_CONSECUENTE,
    BITS_POR_REGLA,
    ResultadoMichiganBinario,
    contar_duplicados,
    ejecutar_ag_michigan_binario,
)

__all__ = [
    "BITS_POR_GEN",
    "BITS_POR_CONSECUENTE",
    "BITS_POR_REGLA",
    "ResultadoMichiganBinario",
    "contar_duplicados",
    "ejecutar_ag_michigan_binario",
]
