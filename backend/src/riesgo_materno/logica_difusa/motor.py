import numpy as np
import skfuzzy as fuzz

from .variables import (
    ESPECIFICACIONES_VARIABLES,
    PUNTOS_GRAFICA,
    PUNTOS_SALIDA,
    SALIDA_DIFUSA,
    VARIABLES_ENTRADA,
)
from .reglas import REGLAS


class SistemaDifusoMamdani:
    """Motor de inferencia Mamdani para clasificacion de riesgo materno.

    Flujo: fusificacion -> evaluacion de reglas -> desfusificacion (centroide).
    """

    def __init__(self, membresias_entrada):
        """Inicializa el motor con las membresias dadas, construye universos, curvas trapezoidales y compila reglas."""
        self.membresias_entrada = membresias_entrada
        self.puntaje_neutro = 50.0
        self.universos_entrada = self._crear_universos_entrada()
        self.universo_salida = self._crear_universo_salida()
        self.curvas_entrada = self._crear_curvas_entrada()
        self.curvas_salida = self._crear_curvas_salida()
        self._reglas_compiladas = self._compilar_reglas()

    # -- Inferencia (vectorizada para velocidad) --
    def inferir_lote(self, entradas):
        """Clasifica un lote de casos de forma vectorizada."""
        n = len(next(iter(entradas.values())))

        # Paso 1: fusificar — calcular grado de pertenencia de cada valor en cada categoria
        pertenencias = {}  # pertenencias[var][cat] = array de n floats
        for variable in VARIABLES_ENTRADA:
            pertenencias[variable] = {}
            universo = self.universos_entrada[variable]
            valores = np.asarray(entradas[variable], dtype=float)
            for categoria, curva in self.curvas_entrada[variable].items():
                pertenencias[variable][categoria] = np.array(
                    [fuzz.interp_membership(universo, curva, v) for v in valores],
                    dtype=float,
                )
        
        # Paso 2: evaluar reglas — AND=minimo acumulativo, OR=maximo entre reglas
        act_bajo = np.zeros(n, dtype=float)
        act_medio = np.zeros(n, dtype=float)
        act_alto = np.zeros(n, dtype=float)
        mapa_act = {"bajo": act_bajo, "medio": act_medio, "alto": act_alto}

        for antecedentes, consecuente in self._reglas_compiladas:
            # AND de antecedentes: minimo a lo largo de antecedentes
            fuerza = np.ones(n, dtype=float)
            for termino, categoria in antecedentes:
                fuerza = np.minimum(fuerza, pertenencias[termino][categoria])
            # OR entre reglas: maximo
            np.maximum(mapa_act[consecuente], fuerza, out=mapa_act[consecuente])

        # Paso 3: desfusificar cada caso
        puntajes = np.empty(n, dtype=float)
        sin_activacion = np.zeros(n, dtype=bool)
        for i in range(n):
            activaciones_i = {
                "bajo": float(act_bajo[i]),
                "medio": float(act_medio[i]),
                "alto": float(act_alto[i]),
            }
            # Si ninguna regla activo nada, todas las activaciones son 0
            sin_activacion[i] = (act_bajo[i] == 0.0 and act_medio[i] == 0.0 and act_alto[i] == 0.0)
            puntajes[i] = self._desfusificar(activaciones_i)

        riesgos = np.array([puntaje_a_riesgo(p) for p in puntajes], dtype=object)

        activaciones = [
            {"bajo": float(act_bajo[i]), "medio": float(act_medio[i]), "alto": float(act_alto[i])}
            for i in range(n)
        ]

        return {
            "puntajes": puntajes,
            "riesgos": riesgos,
            "activaciones": activaciones,
            "sin_activacion": sin_activacion,
        }

    def inferir_con_explicacion(self, entradas):
        """Infiere el riesgo de un paciente exponiendo cada paso del proceso difuso.

        entradas: dict {variable: float}

        Retorna:
          pertenencias:      grado de membresia de cada valor en cada categoria.
          reglas_activadas:  reglas con fuerza > 0, con el detalle del AND de sus antecedentes.
          activaciones:      fuerza final de cada nivel de riesgo tras aplicar OR entre reglas.
          puntaje y riesgo:  resultado de la desfusificacion.
        """
        # Paso 1: fusificacion — cuanto pertenece cada valor ingresado a cada categoria
        pertenencias = {}
        for variable in VARIABLES_ENTRADA:
            universo = self.universos_entrada[variable]
            valor = float(entradas[variable])
            pertenencias[variable] = {
                categoria: float(fuzz.interp_membership(universo, curva, valor))
                for categoria, curva in self.curvas_entrada[variable].items()
            }

        # Paso 2: evaluacion de reglas
        activaciones = {"bajo": 0.0, "medio": 0.0, "alto": 0.0}
        reglas_activadas = []

        for regla in REGLAS:

            antecedentes_con_valor = [
                {
                    "variable":    termino,
                    "categoria":   categoria,
                    "pertenencia": pertenencias[termino][categoria],
                }
                for termino, categoria in regla["antecedentes"]
            ]

            # AND: la fuerza de la regla es el minimo de las pertenencias de sus antecedentes
            fuerza = min(a["pertenencia"] for a in antecedentes_con_valor)

            # OR: si otra regla ya activo este consecuente con mas fuerza, se conserva el maximo
            activaciones[regla["consecuente"]] = max(activaciones[regla["consecuente"]], fuerza)

            if fuerza > 0.0:
                reglas_activadas.append({
                    "numero":       regla["numero"],
                    "antecedentes": antecedentes_con_valor,
                    "fuerza":       fuerza,
                    "consecuente":  regla["consecuente"],
                })

        # Paso 3: desfusificacion — centroide del area resultante
        puntaje = self._desfusificar(activaciones)

        # Ninguna regla se activo si todas las activaciones son 0 — puntaje=50 es neutro, no clinico
        sin_activacion = all(v == 0.0 for v in activaciones.values())

        return {
            "pertenencias":     pertenencias,
            "reglas_activadas": reglas_activadas,
            "activaciones":     activaciones,
            "puntaje":          float(puntaje),
            "riesgo":           puntaje_a_riesgo(puntaje),
            "sin_activacion":   sin_activacion,
        }

    # -- Desfusificacion --
    def _desfusificar(self, activaciones):
        """Agrega las salidas recortadas y calcula el centroide."""
        salida_agregada = np.zeros_like(self.universo_salida, dtype=float)

        for categoria, curva_salida in self.curvas_salida.items():
            salida_recortada = np.fmin(activaciones[categoria], curva_salida)
            salida_agregada = np.fmax(salida_agregada, salida_recortada)

        
        if float(np.max(salida_agregada)) == 0.0:
            return self.puntaje_neutro

        return float(fuzz.defuzz(self.universo_salida, salida_agregada, "centroid"))

    # -- Compilacion de reglas --
    def _compilar_reglas(self):
        """Convierte las reglas a tuplas para evaluacion rapida."""
        compiladas = []
        for regla in REGLAS:
            var_cats = [(v, c) for v, c in regla["antecedentes"]]
            compiladas.append((var_cats, regla["consecuente"]))
        return compiladas

    # -- Construccion de universos y curvas --
    def _crear_universos_entrada(self):
        """Genera un array lineal de PUNTOS_GRAFICA valores entre los limites de cada variable de entrada."""
        universos = {}
        for variable, espec in ESPECIFICACIONES_VARIABLES.items():
            minimo, maximo = espec["limites"]
            universos[variable] = np.linspace(minimo, maximo, PUNTOS_GRAFICA)
        return universos

    def _crear_universo_salida(self):
        """Genera el universo de la salida con PUNTOS_SALIDA valores — mas puntos para mejor precision en centroide."""
        minimo, maximo = SALIDA_DIFUSA["universo"]
        return np.linspace(minimo, maximo, PUNTOS_SALIDA)

    def _crear_curvas_entrada(self):
        """Aplica trapmf a cada categoria de cada variable usando las membresias actuales para obtener curvas Y."""
        curvas = {}
        for variable, categorias in self.membresias_entrada.items():
            universo = self.universos_entrada[variable]
            curvas[variable] = {}
            for categoria, puntos in categorias.items():
                curvas[variable][categoria] = fuzz.trapmf(universo, puntos)
        return curvas

    def _crear_curvas_salida(self):
        """Genera las curvas trapezoidales de la salida — definidas manualmente, el AG no las optimiza."""
        curvas = {}
        for categoria, puntos in SALIDA_DIFUSA["categorias"].items():
            curvas[categoria] = fuzz.trapmf(self.universo_salida, puntos)
        return curvas


def puntaje_a_riesgo(puntaje):
    """Convierte un puntaje numerico a una etiqueta de riesgo."""
    if np.isnan(puntaje):
        return None
    if puntaje < 40.0:
        return "low risk"
    if puntaje < 65.0:
        return "mid risk"
    return "high risk"
