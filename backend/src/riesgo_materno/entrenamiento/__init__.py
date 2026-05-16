__all__ = [
    "cargar_seleccion_reglas",
    "construir_membresias_base",
    "entrenar_seleccion_reglas",
]


def __getattr__(nombre):
    if nombre in __all__:
        from . import entrenador

        return getattr(entrenador, nombre)
    raise AttributeError(f"module {__name__!r} has no attribute {nombre!r}")
