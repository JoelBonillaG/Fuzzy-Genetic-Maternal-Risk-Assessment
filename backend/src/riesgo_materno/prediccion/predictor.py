from ..entrenamiento.entrenador import obtener_membresias_optimizadas
from ..logica_difusa.motor import SistemaDifusoMamdani
from .validacion_entrada import construir_entrada_lote, validar_valores_entrada

def predecir_caso(valores_entrada):
    """Predice el riesgo materno de un paciente usando el sistema difuso con membresias optimizadas por el AG."""
    membresias_optimizadas, origen_modelo = obtener_membresias_optimizadas()
    sistema = SistemaDifusoMamdani(membresias_optimizadas)

    entradas, ajustes = construir_entrada_lote(valores_entrada)

    inferencia = sistema.inferir_lote(entradas)

    return {
        "puntaje": float(inferencia["puntajes"][0]),
        "riesgo": str(inferencia["riesgos"][0]),
        "sin_activacion": bool(inferencia["sin_activacion"][0]),
        "sistema": "optimizado entrenado desde el CSV original",
        "origen_modelo": origen_modelo,
        "ajustes_entrada": ajustes,
    }

def obtener_curvas_membresia():
    """Devuelve las curvas trapezoidales optimizadas de cada variable para visualizacion en el frontend."""
    membresias_optimizadas, origen_modelo = obtener_membresias_optimizadas()
    sistema = SistemaDifusoMamdani(membresias_optimizadas)

    variables = {}
    for variable, universo in sistema.universos_entrada.items():
        puntos_x = universo.tolist()
        variables[variable] = {
            categoria: {
                "puntos_x": puntos_x,
                "puntos_y": curva.tolist(),
            }
            for categoria, curva in sistema.curvas_entrada[variable].items()
        }

    return {
        "variables": variables,
        "origen_modelo": origen_modelo,
    }

def predecir_caso_con_explicacion(valores_entrada):
    """Predice el riesgo exponiendo pertenencias, reglas activadas y activaciones de cada nivel de riesgo."""
    membresias_optimizadas, origen_modelo = obtener_membresias_optimizadas()
    sistema = SistemaDifusoMamdani(membresias_optimizadas)

    entradas, ajustes = validar_valores_entrada(valores_entrada)
    resultado = sistema.inferir_con_explicacion(entradas)

    return {
        **resultado,
        "entrada_validada": entradas,
        "origen_modelo": origen_modelo,
        "ajustes_entrada": ajustes,
        "sin_activacion": resultado["sin_activacion"],
    }