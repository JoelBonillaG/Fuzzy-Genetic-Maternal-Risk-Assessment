from collections import OrderedDict

import numpy as np

from ..logica_difusa.variables import ESPECIFICACIONES_VARIABLES

# Maximo solapamiento permitido entre dos trapecios vecinos.
# El soporte es la base completa del trapecio, desde a hasta d.
# Este limite se calcula usando el soporte del trapecio mas pequeno.
MAX_FRACCION_SOLAPAMIENTO = 0.30

# Numero de veces que se repite el ajuste entre categorias vecinas.
# Se hacen varias pasadas porque al corregir un par se puede afectar al siguiente.
ITERACIONES_REPARACION = 5


# ---------------------------------------------------------------------------
# Funciones basicas de trapecios
# Un trapecio difuso se guarda como [a, b, c, d]:
#   a, d -> inicio y fin del soporte
#   b, c -> inicio y fin del nucleo
# ---------------------------------------------------------------------------

def centro_de_trapecio(trapecio):
    """Devuelve el centro del nucleo del trapecio."""
    return float((trapecio[1] + trapecio[2]) / 2.0)


def ancho_de_soporte(trapecio):
    """Devuelve el ancho del soporte, es decir, d - a."""
    return float(trapecio[3] - trapecio[0])


def reparar_trapecio(trapecio, limite_inferior, limite_superior, epsilon):
    """Corrige un trapecio para que no quede fuera de forma.

    Esta funcion se asegura de tres cosas:
    1. que los puntos queden dentro del dominio de la variable,
    2. que esten ordenados como a <= b <= c <= d,
    3. que el trapecio no se vuelva demasiado pequeno.
    """
    trapecio = np.clip(
        np.sort(np.asarray(trapecio, dtype=float)),
        limite_inferior,
        limite_superior,
    )
    a, b, c, d = trapecio

    ancho_minimo = max(epsilon * 0.5, 0.000001)

    if d - a < ancho_minimo:
        # Si el soporte es muy pequeno, se abre un poco desde el centro.
        punto_medio = np.clip((a + d) / 2.0, limite_inferior, limite_superior)
        mitad = ancho_minimo / 2.0
        a = max(limite_inferior, punto_medio - mitad)
        d = min(limite_superior, punto_medio + mitad)

        if d - a < ancho_minimo:
            # Si esta pegado a un borde, se extiende hacia adentro del dominio.
            if a <= limite_inferior:
                d = min(limite_superior, limite_inferior + ancho_minimo)
            else:
                a = max(limite_inferior, limite_superior - ancho_minimo)
                d = limite_superior

    b = np.clip(b, a, d)
    c = np.clip(c, b, d)
    return np.asarray([a, b, c, d], dtype=float)


# ---------------------------------------------------------------------------
# Orden y ajuste entre categorias
# ---------------------------------------------------------------------------

def ordenar_categorias_por_centro(bloque):
    """Ordena los trapecios de una variable de izquierda a derecha."""
    centros = np.asarray([centro_de_trapecio(fila) for fila in bloque], dtype=float)
    orden = np.argsort(centros, kind="stable")
    return bloque[orden]


# bloque -> edad: joven, adulta, avanzada
def ajustar_categorias_adyacentes(bloque, limite_inferior, limite_superior, epsilon):
    """Ajusta trapecios vecinos para que queden mejor organizados.

    En cada par revisa tres cosas:
    1. que no haya huecos,
    2. que no haya demasiado solapamiento,
    3. que los nucleos no se crucen.
    """
    bloque = bloque.copy()

    for _ in range(ITERACIONES_REPARACION):
        bloque = np.asarray(
            [reparar_trapecio(fila, limite_inferior, limite_superior, epsilon) for fila in bloque],
            dtype=float,
        )
        bloque = ordenar_categorias_por_centro(bloque)

        for indice in range(len(bloque) - 1):
            izquierda = bloque[indice].copy()
            derecha = bloque[indice + 1].copy()

            # Si hay hueco entre soportes, ambos se unen en el punto medio.
            # d del trapecio izquierdo y a del trapecio derecho.
            if izquierda[3] < derecha[0]:
                punto_medio = (izquierda[3] + derecha[0]) / 2.0
                izquierda[3] = punto_medio
                derecha[0] = punto_medio

            izquierda = reparar_trapecio(izquierda, limite_inferior, limite_superior, epsilon)
            derecha = reparar_trapecio(derecha, limite_inferior, limite_superior, epsilon)

            # Si se montan demasiado, se reduce el exceso en ambos lados.
            solapamiento = izquierda[3] - derecha[0]
            maximo_solapamiento = MAX_FRACCION_SOLAPAMIENTO * min(
                ancho_de_soporte(izquierda),
                ancho_de_soporte(derecha),
            )
            if solapamiento > maximo_solapamiento:
                exceso = (solapamiento - maximo_solapamiento) / 2.0
                izquierda[3] -= exceso
                derecha[0] += exceso

            izquierda = reparar_trapecio(izquierda, limite_inferior, limite_superior, epsilon)
            derecha = reparar_trapecio(derecha, limite_inferior, limite_superior, epsilon)

            # Si un nucleo invade al otro, se separan un poco.
            # c del trapecio izquierdo y b del trapecio derecho.
            if izquierda[2] > derecha[1]:
                punto_medio = (izquierda[2] + derecha[1]) / 2.0
                izquierda[2] = punto_medio - epsilon / 2.0
                derecha[1] = punto_medio + epsilon / 2.0

            izquierda = reparar_trapecio(izquierda, limite_inferior, limite_superior, epsilon)
            derecha = reparar_trapecio(derecha, limite_inferior, limite_superior, epsilon)
            bloque[indice] = izquierda
            bloque[indice + 1] = derecha

    return bloque


# ---------------------------------------------------------------------------
# Reparacion del cromosoma completo
# ---------------------------------------------------------------------------

def reparar_bloque_variable(bloque, limite_inferior, limite_superior, epsilon):
    """Repara todos los trapecios de una variable.

    El proceso es:
    recortar al dominio, ordenar puntos, reparar trapecios,
    ordenar categorias, ajustar vecinas y reparar otra vez.
    """
    bloque = np.clip(np.asarray(bloque, dtype=float), limite_inferior, limite_superior)
    
    bloque = np.sort(bloque, axis=1)
    
    bloque = np.asarray(
        [reparar_trapecio(fila, limite_inferior, limite_superior, epsilon) for fila in bloque],
        dtype=float,
    )
    bloque = ordenar_categorias_por_centro(bloque)
    bloque = ajustar_categorias_adyacentes(bloque, limite_inferior, limite_superior, epsilon)
    bloque = ordenar_categorias_por_centro(bloque)
    bloque = np.asarray(
        [reparar_trapecio(fila, limite_inferior, limite_superior, epsilon) for fila in bloque],
        dtype=float,
    )
    return bloque


def reparar_cromosoma(cromosoma):
    """Repara todo el cromosoma, variable por variable."""
    cromosoma = np.asarray(cromosoma, dtype=float).copy()
    genes_reparados = []
    cursor = 0

    for variable, especificacion in ESPECIFICACIONES_VARIABLES.items():
        cantidad_categorias = len(especificacion["categorias"])
        bloque = cromosoma[cursor : cursor + cantidad_categorias * 4].reshape(cantidad_categorias, 4)
        bloque_reparado = reparar_bloque_variable(
            bloque,
            limite_inferior=especificacion["limites"][0],
            limite_superior=especificacion["limites"][1],
            epsilon=especificacion["epsilon"],
        )
        genes_reparados.extend(bloque_reparado.reshape(-1).tolist())
        cursor += cantidad_categorias * 4

    return np.asarray(genes_reparados, dtype=float)


# ---------------------------------------------------------------------------
# Codificacion y decodificacion
# ---------------------------------------------------------------------------

def decodificar_cromosoma(cromosoma):
    """Convierte el cromosoma plano en variable -> categoria -> [a, b, c, d]."""
    membresias = OrderedDict()
    cursor = 0
    for variable, especificacion in ESPECIFICACIONES_VARIABLES.items():
        membresias[variable] = OrderedDict()
        for categoria in especificacion["categorias"].keys():
            membresias[variable][categoria] = np.asarray(cromosoma[cursor : cursor + 4], dtype=float)
            cursor += 4
    return membresias


def aplanar_membresias(membresias):
    """Convierte las membresias en arreglos utiles para el optimizador.

    Devuelve cuatro arreglos:
    genes, minimos, maximos y rangos.
    """
    genes, minimos, maximos, rangos = [], [], [], []

    for variable, especificacion in ESPECIFICACIONES_VARIABLES.items():
        limite_inferior, limite_superior = especificacion["limites"]
        rango = limite_superior - limite_inferior
        for categoria in especificacion["categorias"].keys():
            puntos = np.asarray(membresias[variable][categoria], dtype=float)
            genes.extend(puntos.tolist())
            minimos.extend([limite_inferior] * 4)
            maximos.extend([limite_superior] * 4)
            rangos.extend([rango] * 4)

    return (
        np.asarray(genes, dtype=float),
        np.asarray(minimos, dtype=float),
        np.asarray(maximos, dtype=float),
        np.asarray(rangos, dtype=float),
    )

# Es el punto de partida
def crear_membresias_base():
    """Crea las membresias iniciales usando ESPECIFICACIONES_VARIABLES."""
    membresias = OrderedDict()
    for variable, especificacion in ESPECIFICACIONES_VARIABLES.items():
        membresias[variable] = OrderedDict()
        for categoria, puntos in especificacion["categorias"].items():
            membresias[variable][categoria] = np.asarray(puntos, dtype=float)
    return membresias

# ---------------------------------------------------------------------------
# Cromosoma base: punto de partida del optimizador
# ---------------------------------------------------------------------------

MEMBRESIAS_BASE = crear_membresias_base()
CROMOSOMA_BASE_CRUDO, LIMITES_INFERIORES, LIMITES_SUPERIORES, RANGOS_GENES = aplanar_membresias(MEMBRESIAS_BASE)

# Solo es por seguridad, aunqeu al ser base ya deberian ser validso en primer momento.
CROMOSOMA_BASE = reparar_cromosoma(CROMOSOMA_BASE_CRUDO)
