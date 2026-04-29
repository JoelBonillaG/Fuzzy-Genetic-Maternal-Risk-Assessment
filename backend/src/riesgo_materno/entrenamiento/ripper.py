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

def aprender_reglas_ripper(tabla):
    """
    Aprende reglas IF-THEN desde el dataset usando RIPPER.

    Parámetros
    ----------
    tabla : pd.DataFrame
        Resultado de cargar_datos() — columnas numéricas + columna "riesgo".

    Retorna
    -------
    list[dict]
        Reglas en el formato de reglas.py, listas para usar en el motor difuso.
    """
    ejemplos = _discretizar(tabla)
    df = pd.DataFrame(ejemplos).rename(columns={"clase": "riesgo"})

    reglas = []
    numero = 1

    for clase in ORDEN_CLASES:
        # RIPPER es binario: convertir a "clase vs resto"
        df_binario = df.copy()
        df_binario["riesgo"] = df["riesgo"].apply(lambda x: clase if x == clase else "otro")

        clf = RIPPER()
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


# ── Ejecución directa: genera reglas y las guarda en JSON ────────────────────

if __name__ == "__main__":
    import json
    from .datos import cargar_datos
    from .modelo import RUTA_CSV, RUTA_REGLAS_APRENDIDAS

    print("Cargando dataset...")
    datos = cargar_datos(RUTA_CSV)

    print("Ejecutando RIPPER...")
    reglas = aprender_reglas_ripper(datos)

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
