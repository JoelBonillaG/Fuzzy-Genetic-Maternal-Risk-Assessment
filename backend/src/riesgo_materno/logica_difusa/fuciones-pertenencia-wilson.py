from math import sqrt

# ============================================
# DATASET
# ============================================

low = 404
mid = 336
high = 272

n = low + mid + high

# ============================================
# WILSON SCORE INTERVAL
# (MISMA ESTRUCTURA DEL PAPER)
# ============================================

def wilson_paper_formula(X, n, kappa=1.96):
    """
    Implementa exactamente la estructura:

    CIw =
    (X + kappa^2 / 2) / (n + kappa^2)

    ±

    (kappa * sqrt(n)) / (n + kappa^2)
    *
    sqrt(
        p_hat * q_hat
        +
        kappa^2 / (4n)
    )
    """

    # p-hat
    p_hat = X / n

    # q-hat
    q_hat = 1 - p_hat

    # ============================================
    # A = centro
    # ============================================

    A = (
        X + (kappa**2 / 2)
    ) / (
        n + kappa**2
    )

    # ============================================
    # B = margen
    # ============================================

    B = (
        (kappa * sqrt(n))
        /
        (n + kappa**2)
    ) * sqrt(
        (p_hat * q_hat)
        +
        (kappa**2 / (4 * n))
    )

    # ============================================
    # Límites
    # ============================================

    L = A - B
    U = A + B

    return {
        "X": X,
        "n": n,
        "p_hat": p_hat,
        "q_hat": q_hat,
        "A": A,
        "B": B,
        "L": L,
        "U": U,
        "L_100": L * 100,
        "U_100": U * 100
    }

# ============================================
# PROPORCIONES INDIVIDUALES
# ============================================

print("=" * 60)
print("PROPORCIONES INDIVIDUALES")
print("=" * 60)

niveles = {
    "low risk": low,
    "mid risk": mid,
    "high risk": high
}

for nombre, X in niveles.items():

    r = wilson_paper_formula(X, n)

    print("\n" + "=" * 50)
    print(f"Nivel: {nombre}")

    print(f"X = {r['X']}")
    print(f"n = {r['n']}")

    print(f"p_hat = {r['p_hat']:.6f}")
    print(f"q_hat = {r['q_hat']:.6f}")

    print(f"A = {r['A']:.6f}")
    print(f"B = {r['B']:.6f}")

    print(f"L = {r['L']:.6f}  -> {r['L_100']:.2f}%")
    print(f"U = {r['U']:.6f}  -> {r['U_100']:.2f}%")

# ============================================
# CORTES ACUMULADOS PARA MAMDANI
# ============================================

print("\n")
print("=" * 60)
print("CORTES ACUMULADOS PARA MAMDANI")
print("=" * 60)

# --------------------------------------------
# Corte bajo / medio
# --------------------------------------------

corte_1 = wilson_paper_formula(low, n)

print("\nCorte bajo / medio")
print(f"X = low = {low}")

print(f"L = {corte_1['L_100']:.2f}%")
print(f"U = {corte_1['U_100']:.2f}%")

# --------------------------------------------
# Corte medio / alto
# --------------------------------------------

corte_2 = wilson_paper_formula(low + mid, n)

print("\nCorte medio / alto")
print(f"X = low + mid = {low + mid}")

print(f"L = {corte_2['L_100']:.2f}%")
print(f"U = {corte_2['U_100']:.2f}%")

# ============================================
# SALIDA DIFUSA FINAL
# ============================================

SALIDA_DIFUSA = {
    "nombre": "puntaje_riesgo",
    "universo": (0.0, 100.0),
    "categorias": {
        "bajo": [
            0.0,
            0.0,
            round(corte_1["L_100"], 2),
            round(corte_1["U_100"], 2)
        ],

        "medio": [
            round(corte_1["L_100"], 2),
            round(corte_1["U_100"], 2),
            round(corte_2["L_100"], 2),
            round(corte_2["U_100"], 2)
        ],

        "alto": [
            round(corte_2["L_100"], 2),
            round(corte_2["U_100"], 2),
            100.0,
            100.0
        ]
    }
}

print("\n")
print("=" * 60)
print("SALIDA DIFUSA")
print("=" * 60)

print(SALIDA_DIFUSA)