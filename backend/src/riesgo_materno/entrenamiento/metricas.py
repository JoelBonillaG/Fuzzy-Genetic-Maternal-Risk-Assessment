from sklearn.metrics import f1_score, recall_score

from ..logica_difusa.variables import ETIQUETAS_RIESGO

def calcular_recall_de_riesgo_alto(riesgos_reales, riesgos_predichos):
    """Fraccion de casos reales de high risk que el sistema detecto correctamente."""
    return float(
        recall_score(
            riesgos_reales,
            riesgos_predichos,
            labels=ETIQUETAS_RIESGO,
            average=None,
            zero_division=0,
            # es mas grave no detectar un riesgo alto,
            # que detectar un riesgo alto cuando no lo es, por eso se le da mas peso al recall de riesgo alto.
        )[ETIQUETAS_RIESGO.index("high risk")]
    )


def calcular_macro_f1(riesgos_reales, riesgos_predichos):
    """Promedio simple del F1 de las 3 clases — trata bajo/medio/alto con igual peso."""
    return float(
        f1_score(
            riesgos_reales,
            riesgos_predichos,
            labels=ETIQUETAS_RIESGO,
            average="macro",
            zero_division=0,
        )
    )
