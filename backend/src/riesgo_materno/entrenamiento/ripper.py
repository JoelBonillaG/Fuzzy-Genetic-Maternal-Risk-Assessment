"""
RIPPER — Inducción de reglas usando la librería wittgenstein.

RIPPER construye reglas IF-THEN por clase y luego poda condiciones que no
aportan. Resultado: reglas más cortas y generales que PRISM.

RIPPER es binario (una clase vs el resto), así que se entrena uno por clase:
high risk → mid risk → low risk.

Requiere: pip install wittgenstein
"""

import pandas as pd
from wittgenstein import RIPPER

from ..logica_difusa.variables import ESPECIFICACIONES_VARIABLES

ORDEN_CLASES = ["high risk", "mid risk", "low risk"]

MAPA_CONSECUENTE = {
    "high risk": "alto",
    "mid risk":  "medio",
    "low risk":  "bajo",
}
CONSECUENTE_A_CLASE = {v: k for k, v in MAPA_CONSECUENTE.items()}


# ── Discretización ────────────────────────────────────────────────────────────

def _grado_trapecio(x, puntos):
    """Membresía de x en un trapecio [a, b, c, d].
    Usa < y > en los extremos para incluir los valores límite del dataset.
    """
    a, b, c, d = puntos
    if x < a or x > d:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    if x <= c:
        return 1.0
    return (d - x) / (d - c) if d != c else 0.0


def _categoria(valor, categorias):
    """Devuelve la categoría con mayor grado de membresía para un valor."""
    return max(categorias, key=lambda cat: _grado_trapecio(valor, categorias[cat]))


def _discretizar(tabla):
    """Convierte el DataFrame numérico en lista de dicts con categorías + clase."""
    ejemplos = []
    for _, fila in tabla.iterrows():
        ejemplo = {
            var: _categoria(fila[var], spec["categorias"])
            for var, spec in ESPECIFICACIONES_VARIABLES.items()
        }
        ejemplo["clase"] = fila["riesgo"]
        ejemplos.append(ejemplo)
    return ejemplos


# ── RIPPER ────────────────────────────────────────────────────────────────────

def aprender_reglas_ripper(tabla, orden_clases=None, parametros=None):
    """
    Aprende reglas IF-THEN desde el dataset usando RIPPER.

    Parámetros
    ----------
    tabla : pd.DataFrame
        Resultado de cargar_dataset() — columnas numéricas + columna "riesgo".

    Retorna
    -------
    list[dict]
        Reglas en el formato de reglas.py, listas para usar en el motor difuso.
    """
    ejemplos = _discretizar(tabla)
    df = pd.DataFrame(ejemplos).rename(columns={"clase": "riesgo"})

    reglas = []
    numero = 1

    orden = orden_clases or ORDEN_CLASES
    parametros = parametros or {}

    for clase in orden:
        # RIPPER es binario: convertir a "clase vs resto"
        df_binario = df.copy()
        df_binario["riesgo"] = df["riesgo"].apply(lambda x: clase if x == clase else "otro")

        clf = RIPPER(**parametros)
        clf.fit(df_binario, class_feat="riesgo", pos_class=clase)

        for rule in clf.ruleset_.rules:
            condiciones = [(cond.feature, cond.val) for cond in rule.conds]

            if not condiciones:
                continue

            reglas.append({
                "numero":       numero,
                "antecedentes": condiciones,
                "consecuente":  MAPA_CONSECUENTE[clase],
            })
            numero += 1

    return reglas

# Después de aprender_reglas_ripper, evalúa así:
def evaluar_reglas_duras(reglas, tabla):
    ejemplos = _discretizar(tabla)
    aciertos = 0
    for ej in ejemplos:
        clase_real = ej["clase"]
        # Aplica reglas en orden, primera que matche gana
        prediccion = None
        for r in reglas:
            if all(ej[var] == cat for var, cat in r["antecedentes"]):
                prediccion = {"alto": "high risk", "medio": "mid risk", 
                              "bajo": "low risk"}[r["consecuente"]]
                break
        if prediccion == clase_real:
            aciertos += 1
    return aciertos / len(ejemplos)

# ── Ejecución directa: genera reglas y las guarda en JSON ────────────────────

if __name__ == "__main__":
    import json
    from .datos import cargar_dataset, dividir_entrenamiento_prueba
    from .modelo import RUTA_CSV, RUTA_REGLAS_APRENDIDAS

    print("Cargando dataset...")
    datos = cargar_dataset(RUTA_CSV)

    print("Dividiendo 70/30 estratificado...")
    splits = dividir_entrenamiento_prueba(datos, semilla=42)
    entrenamiento = splits["entrenamiento"]
    prueba = splits["prueba"]
    print(f"  Entrenamiento: {len(entrenamiento)} instancias")
    print(f"  Prueba:        {len(prueba)} instancias")

    print("Ejecutando RIPPER sobre el conjunto de entrenamiento...")
    reglas = aprender_reglas_ripper(entrenamiento)

    print("Evaluando reglas duras sobre entrenamiento...")
    precision_train = evaluar_reglas_duras(reglas, entrenamiento)
    print(f"  Precision entrenamiento: {precision_train:.2%}")

    print("Evaluando reglas duras sobre prueba...")
    precision_test = evaluar_reglas_duras(reglas, prueba)
    print(f"  Precision prueba:        {precision_test:.2%}")

    # Guardar en JSON — antecedentes como listas (JSON no tiene tuplas)
    contenido = [
        {
            "numero":       r["numero"],
            "antecedentes": [list(ant) for ant in r["antecedentes"]],
            "consecuente":  r["consecuente"],
        }
        for r in reglas
    ]
    RUTA_REGLAS_APRENDIDAS.write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n{len(reglas)} reglas guardadas en: {RUTA_REGLAS_APRENDIDAS}\n")
    for r in reglas:
        ants = " AND ".join(f"{v}={c}" for v, c in r["antecedentes"])
        print(f"  Regla {r['numero']:>2}: SI {ants} → {r['consecuente']}")

    
