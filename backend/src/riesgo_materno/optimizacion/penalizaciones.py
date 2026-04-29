import numpy as np

from ..logica_difusa.variables import ESPECIFICACIONES_VARIABLES
from .cromosoma import decodificar_cromosoma

# Maximo solapamiento permitido entre dos trapecios vecinos.
# El soporte es la base completa del trapecio, desde a hasta d.
# Este limite se calcula usando el soporte del trapecio mas pequeno.
# Debe ser la misma que se usa en cromosoma.py.
MAX_FRACCION_SOLAPAMIENTO = 0.30



# ---------------------------------------------------------------------------
# Penalizaciones individuales
# ---------------------------------------------------------------------------

def _penalizacion_hueco(izquierda, derecha, rango_variable):
    """Mide el hueco entre dos trapecios vecinos normalizado por el rango — zona sin pertenencia activa.

    Hay hueco cuando el trapecio izquierdo termina antes de que empiece el derecho.
    Si se tocan o se solapan, la penalizacion es 0.
    Mientras mas grande sea el hueco, mas se acerca el valor a 1.
    """
    _, _, _, d_izq = izquierda
    a_der, _, _, _ = derecha
    hueco = max(0.0, a_der - d_izq)
    return float(np.clip(hueco / max(rango_variable, 1e-9), 0.0, 1.0))


def _penalizacion_solapamiento_excesivo(izquierda, derecha, rango_variable):
    """Mide el exceso de solapamiento sobre el maximo permitido (30%) normalizado por el rango.

    Un poco de solapamiento es util porque evita cortes bruscos entre categorias.
    El problema aparece cuando se superponen demasiado y dejan de distinguirse bien.
    Si el solapamiento real no supera el limite, la penalizacion es 0.
    """
    a_izq, _, _, d_izq = izquierda
    a_der, _, _, d_der = derecha
    solapamiento = max(0.0, d_izq - a_der)
    maximo_permitido = MAX_FRACCION_SOLAPAMIENTO * min(d_izq - a_izq, d_der - a_der)
    exceso = max(0.0, solapamiento - maximo_permitido)
    return float(np.clip(exceso / max(rango_variable, 1e-9), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Funciones principales de penalizacion
# Interpretabilidad: revisa si las funciones de membresia quedaron bien formadas.
# ---------------------------------------------------------------------------

def calcular_penalizacion_interpretabilidad(cromosoma):
    """Calcula penalizacion por huecos y solapamiento excesivo entre categorias vecinas — penaliza trapecios poco interpretables.

    Para cada variable revisa pares de trapecios vecinos y calcula dos cosas:
    si hay huecos y si hay demasiado solapamiento.

    Devuelve un dict con valores entre 0 y 1:
      huecos: promedio de espacios vacios entre categorias vecinas.
      solapamiento: promedio del exceso de solapamiento entre vecinas.
      total: promedio de ambos criterios.
    """
    membresias = decodificar_cromosoma(cromosoma)

    pen_huecos = []
    pen_solapamiento = []

    for variable, especificacion in ESPECIFICACIONES_VARIABLES.items():
        limite_inferior, limite_superior = especificacion["limites"]
        rango_variable = limite_superior - limite_inferior

        trapecios = list(membresias[variable].values())

        for i in range(len(trapecios) - 1):
            pen_huecos.append(
                _penalizacion_hueco(trapecios[i], trapecios[i + 1], rango_variable)
            )
            pen_solapamiento.append(
                _penalizacion_solapamiento_excesivo(
                    trapecios[i], trapecios[i + 1], rango_variable
                )
            )

    def promedio(valores):
        return float(np.mean(valores)) if valores else 0.0

    p_huecos = promedio(pen_huecos)
    p_solapamiento = promedio(pen_solapamiento)

    return {
        "huecos": p_huecos,
        "solapamiento": p_solapamiento,
        "total": promedio([p_huecos, p_solapamiento]),
    }


# Desviacion: cuanto se alejo del diseno experto original.
def calcular_penalizacion_desviacion(cromosoma, cromosoma_base, rangos_genes):
    """Mide desviacion promedio gen a gen respecto al cromosoma base — penaliza alejarse del diseno clinico original.

    Compara gen por gen, toma la diferencia absoluta y la divide por el rango
    de cada gen para que todos aporten en la misma escala.

    Devuelve un valor entre 0 y 1:
    0 significa igual al experto; valores mayores significan mas diferencia.
    """
    desviaciones = np.abs(np.asarray(cromosoma) - np.asarray(cromosoma_base)) / np.asarray(
        rangos_genes
    )
    return float(np.mean(desviaciones))
