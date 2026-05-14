"""Optimizacion Pittsburgh-Michigan con cromosoma numerico PyGAD."""

from .algoritmo import (
    BITS_POR_CAMPO,
    BITS_POR_REGLA,
    CAMPOS_POR_REGLA,
    contar_duplicados,
    ejecutar_ag_pittsburgh_michigan,
)

__all__ = [
    "BITS_POR_CAMPO",
    "BITS_POR_REGLA",
    "CAMPOS_POR_REGLA",
    "contar_duplicados",
    "ejecutar_ag_pittsburgh_michigan",
]
