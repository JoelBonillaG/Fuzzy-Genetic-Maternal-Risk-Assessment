"""Busca bases de reglas RIPPER prometedoras antes de correr el AG.

Este script NO ejecuta el algoritmo genetico. Su objetivo es evitar probar a
mano conjuntos de reglas pobres: genera muchas bases RIPPER con semilla fija 42,
las evalua con metricas duras y difusas en validacion interna, y rankea las mas
utiles para que luego el AG tenga una base candidata razonable.

Para evitar fuga de informacion, el ranking NO usa el 30% de prueba del split
principal. Ese test se calcula solo como auditoria informativa.

Uso:
    python -m riesgo_materno.herramientas.ag.buscar_reglas_ripper
    python -m riesgo_materno.herramientas.ag.buscar_reglas_ripper --iteraciones 120 --top 10
    python -m riesgo_materno.herramientas.ag.buscar_reglas_ripper --guardar-mejor
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from wittgenstein import RIPPER

from ...entrenamiento.datos import (
    cargar_dataset,
    convertir_split_a_diccionario,
    dividir_entrenamiento_prueba,
)
from ...entrenamiento.modelo import RUTA_CSV, RUTA_REGLAS_APRENDIDAS
from ...entrenamiento.ripper import MAPA_CONSECUENTE, _discretizar
from ...logica_difusa.motor import SistemaDifusoMamdani
from ...logica_difusa.variables import ETIQUETAS_RIESGO
from ...entrenamiento.entrenador import construir_membresias_base


SEMILLA = 42
ORDEN_BASE = ("high risk", "mid risk", "low risk")
MAPA_INVERSO = {"alto": "high risk", "medio": "mid risk", "bajo": "low risk"}
RUTA_SALIDA = Path(__file__).resolve().parents[2] / "modelos" / "busqueda_ripper"


@dataclass
class MetricasReglas:
    candidato: int
    origen: str
    orden_clases: list[str]
    cantidad_reglas: int
    antecedentes_promedio: float
    antecedentes_max: int
    cobertura_train: float
    cobertura_val: float
    cobertura_test: float
    hard_ba_train: float
    hard_ba_val: float
    hard_ba_test: float
    hard_f1_macro_val: float
    hard_f1_macro_test: float
    hard_acc_val: float
    hard_acc_test: float
    fuzzy_ba_train: float
    fuzzy_ba_val: float
    fuzzy_ba_test: float
    fuzzy_f1_macro_val: float
    fuzzy_f1_macro_test: float
    fuzzy_acc_val: float
    fuzzy_acc_test: float
    gap_fuzzy_ba: float
    reglas_por_clase: dict[str, int]
    puntaje_pre_ag: float


def principal():
    args = parsear_argumentos()
    fijar_semillas(args.semilla)

    tabla = cargar_dataset(RUTA_CSV)
    splits = dividir_entrenamiento_prueba(tabla, semilla=args.semilla)
    entrenamiento_total = splits["entrenamiento"]
    prueba = splits["prueba"]
    entrenamiento_ripper, validacion = dividir_entrenamiento_validacion(
        entrenamiento_total,
        fraccion_validacion=args.fraccion_validacion,
        semilla=args.semilla + 1,
    )

    candidatos = generar_candidatos(
        entrenamiento=entrenamiento_ripper,
        iteraciones=args.iteraciones,
        semilla=args.semilla,
        fraccion_bootstrap=args.fraccion_bootstrap,
    )

    evaluados = []
    for indice, candidato in enumerate(candidatos, start=1):
        reglas = candidato["reglas"]
        if not reglas:
            continue
        metricas = evaluar_candidato(
            indice=indice,
            origen=candidato["origen"],
            orden_clases=candidato["orden_clases"],
            reglas=reglas,
            entrenamiento=entrenamiento_ripper,
            validacion=validacion,
            prueba=prueba,
        )
        evaluados.append((metricas, reglas))

    evaluados.sort(key=lambda item: item[0].puntaje_pre_ag, reverse=True)
    mejores = evaluados[: args.top]

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    guardar_reporte(mejores, evaluados, args)

    imprimir_resumen(mejores, total=len(evaluados), args=args)

    if args.guardar_mejor and mejores:
        reglas = serializar_reglas(mejores[0][1])
        RUTA_REGLAS_APRENDIDAS.write_text(
            json.dumps(reglas, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()
        print(f"Mejor base guardada en: {RUTA_REGLAS_APRENDIDAS}")


def parsear_argumentos():
    parser = argparse.ArgumentParser(
        description="Busca reglas RIPPER adecuadas antes de optimizarlas con AG."
    )
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    parser.add_argument("--iteraciones", type=int, default=80)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--fraccion-bootstrap", type=float, default=0.90)
    parser.add_argument(
        "--fraccion-validacion",
        type=float,
        default=0.25,
        help="Fraccion del 70%% de entrenamiento usada solo para rankear candidatos.",
    )
    parser.add_argument(
        "--guardar-mejor",
        action="store_true",
        help="Sobrescribe modelos/reglas_aprendidas.json con la mejor base encontrada.",
    )
    return parser.parse_args()


def fijar_semillas(semilla):
    random.seed(semilla)
    np.random.seed(semilla)


def dividir_entrenamiento_validacion(entrenamiento_total, fraccion_validacion, semilla):
    """Divide el 70% original en train RIPPER y validacion para model selection."""
    train, validacion = train_test_split(
        entrenamiento_total,
        test_size=fraccion_validacion,
        stratify=entrenamiento_total["riesgo"],
        shuffle=True,
        random_state=semilla,
    )
    return train.reset_index(drop=True), validacion.reset_index(drop=True)


def generar_candidatos(entrenamiento, iteraciones, semilla, fraccion_bootstrap):
    """Genera bases con diferentes ordenes de clase y muestras bootstrap.

    RIPPER aprende reglas binarias clase-vs-resto. Cambiar el orden de clases y
    el subconjunto de entrenamiento cambia la base final sin cambiar el split
    70/30 semilla 42 usado para medir.
    """
    candidatos = []
    ordenes = list(itertools.permutations(ORDEN_BASE))

    for orden in ordenes:
        reglas = aprender_reglas_ripper_configurable(entrenamiento, orden)
        candidatos.append({
            "origen": "entrenamiento_completo",
            "orden_clases": list(orden),
            "reglas": reglas,
        })

    rng = np.random.default_rng(semilla)
    for i in range(iteraciones):
        orden = ordenes[i % len(ordenes)]
        muestra = bootstrap_estratificado(entrenamiento, fraccion_bootstrap, rng)
        reglas = aprender_reglas_ripper_configurable(muestra, orden)
        candidatos.append({
            "origen": f"bootstrap_{i + 1:03d}",
            "orden_clases": list(orden),
            "reglas": reglas,
        })

    return candidatos


def bootstrap_estratificado(tabla, fraccion, rng):
    partes = []
    for _, grupo in tabla.groupby("riesgo", sort=False):
        n = max(1, int(round(len(grupo) * fraccion)))
        indices = rng.choice(grupo.index.to_numpy(), size=n, replace=True)
        partes.append(tabla.loc[indices])
    muestra = pd.concat(partes, ignore_index=True)
    return muestra.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))


def aprender_reglas_ripper_configurable(tabla, orden_clases):
    ejemplos = _discretizar(tabla)
    df = pd.DataFrame(ejemplos).rename(columns={"clase": "riesgo"})

    reglas = []
    numero = 1

    for clase in orden_clases:
        df_binario = df.copy()
        df_binario["riesgo"] = df["riesgo"].apply(lambda x: clase if x == clase else "otro")

        clf = construir_ripper()
        clf.fit(df_binario, class_feat="riesgo", pos_class=clase)

        for rule in clf.ruleset_.rules:
            condiciones = [(cond.feature, cond.val) for cond in rule.conds]
            if not condiciones:
                continue
            reglas.append({
                "numero": numero,
                "antecedentes": condiciones,
                "consecuente": MAPA_CONSECUENTE[clase],
            })
            numero += 1

    return quitar_reglas_duplicadas(reglas)


def construir_ripper():
    """Crea RIPPER intentando fijar random_state si la version instalada lo soporta."""
    try:
        return RIPPER(random_state=SEMILLA)
    except TypeError:
        return RIPPER()


def quitar_reglas_duplicadas(reglas):
    vistas = set()
    unicas = []
    for regla in reglas:
        clave = (
            tuple(sorted(tuple(ant) for ant in regla["antecedentes"])),
            regla["consecuente"],
        )
        if clave in vistas:
            continue
        vistas.add(clave)
        regla = {
            "numero": len(unicas) + 1,
            "antecedentes": regla["antecedentes"],
            "consecuente": regla["consecuente"],
        }
        unicas.append(regla)
    return unicas


def evaluar_candidato(indice, origen, orden_clases, reglas, entrenamiento, validacion, prueba):
    hard_train = evaluar_reglas_duras_detallado(reglas, entrenamiento)
    hard_val = evaluar_reglas_duras_detallado(reglas, validacion)
    hard_test = evaluar_reglas_duras_detallado(reglas, prueba)
    fuzzy_train = evaluar_reglas_difusas(reglas, entrenamiento)
    fuzzy_val = evaluar_reglas_difusas(reglas, validacion)
    fuzzy_test = evaluar_reglas_difusas(reglas, prueba)

    cantidades_antecedentes = [len(r["antecedentes"]) for r in reglas]
    reglas_por_clase = {clase: 0 for clase in ("bajo", "medio", "alto")}
    for regla in reglas:
        reglas_por_clase[regla["consecuente"]] += 1

    gap = max(0.0, fuzzy_train["balanced_accuracy"] - fuzzy_val["balanced_accuracy"])
    compacidad_relativa = min(1.0, len(reglas) / 40.0)
    balance_clases = clases_con_reglas(reglas_por_clase) / 3.0

    puntaje_pre_ag = (
        0.50 * fuzzy_val["balanced_accuracy"]
        + 0.20 * hard_val["balanced_accuracy"]
        + 0.10 * hard_val["cobertura"]
        + 0.10 * fuzzy_val["f1_macro"]
        + 0.05 * balance_clases
        - 0.05 * gap
        - 0.03 * compacidad_relativa
    )

    return MetricasReglas(
        candidato=indice,
        origen=origen,
        orden_clases=list(orden_clases),
        cantidad_reglas=len(reglas),
        antecedentes_promedio=float(np.mean(cantidades_antecedentes)),
        antecedentes_max=int(np.max(cantidades_antecedentes)),
        cobertura_train=hard_train["cobertura"],
        cobertura_val=hard_val["cobertura"],
        cobertura_test=hard_test["cobertura"],
        hard_ba_train=hard_train["balanced_accuracy"],
        hard_ba_val=hard_val["balanced_accuracy"],
        hard_ba_test=hard_test["balanced_accuracy"],
        hard_f1_macro_val=hard_val["f1_macro"],
        hard_f1_macro_test=hard_test["f1_macro"],
        hard_acc_val=hard_val["accuracy"],
        hard_acc_test=hard_test["accuracy"],
        fuzzy_ba_train=fuzzy_train["balanced_accuracy"],
        fuzzy_ba_val=fuzzy_val["balanced_accuracy"],
        fuzzy_ba_test=fuzzy_test["balanced_accuracy"],
        fuzzy_f1_macro_val=fuzzy_val["f1_macro"],
        fuzzy_f1_macro_test=fuzzy_test["f1_macro"],
        fuzzy_acc_val=fuzzy_val["accuracy"],
        fuzzy_acc_test=fuzzy_test["accuracy"],
        gap_fuzzy_ba=gap,
        reglas_por_clase=reglas_por_clase,
        puntaje_pre_ag=float(puntaje_pre_ag),
    )


def evaluar_reglas_duras_detallado(reglas, tabla):
    ejemplos = _discretizar(tabla)
    reales = []
    predichos = []
    cubiertos = 0

    for ejemplo in ejemplos:
        prediccion = None
        for regla in reglas:
            if all(ejemplo[var] == cat for var, cat in regla["antecedentes"]):
                prediccion = MAPA_INVERSO[regla["consecuente"]]
                break
        if prediccion is not None:
            cubiertos += 1
        reales.append(ejemplo["clase"])
        predichos.append(prediccion or clase_mayoritaria(tabla))

    return calcular_metricas(reales, predichos) | {"cobertura": cubiertos / len(ejemplos)}


def evaluar_reglas_difusas(reglas, tabla):
    membresias = construir_membresias_base()
    datos = convertir_split_a_diccionario(tabla)
    sistema = SistemaDifusoMamdani(membresias, reglas=reglas)
    inferencia = sistema.inferir_lote(datos["entradas"])
    predichos = predicciones_con_sin_activacion(
        inferencia["riesgos"],
        inferencia["sin_activacion"],
    )
    return calcular_metricas(datos["riesgos"], predichos)


def calcular_metricas(reales, predichos):
    return {
        "accuracy": float(accuracy_score(reales, predichos)),
        "balanced_accuracy": float(
            balanced_accuracy_score(reales, predichos)
        ),
        "f1_macro": float(
            f1_score(reales, predichos, labels=ETIQUETAS_RIESGO, average="macro", zero_division=0)
        ),
    }


def predicciones_con_sin_activacion(predichos, sin_activacion):
    predichos = np.asarray(predichos, dtype=object).copy()
    predichos[np.asarray(sin_activacion, dtype=bool)] = "__sin_activacion__"
    return predichos


def clase_mayoritaria(tabla):
    return str(tabla["riesgo"].mode().iloc[0])


def clases_con_reglas(reglas_por_clase):
    return sum(1 for cantidad in reglas_por_clase.values() if cantidad > 0)


def guardar_reporte(mejores, evaluados, args):
    resumen = {
        "semilla": args.semilla,
        "iteraciones_bootstrap": args.iteraciones,
        "fraccion_bootstrap": args.fraccion_bootstrap,
        "fraccion_validacion_sobre_entrenamiento": args.fraccion_validacion,
        "total_candidatos_validos": len(evaluados),
        "criterio_ranking": (
            "0.50*fuzzy_ba_val + 0.20*hard_ba_val + 0.10*cobertura_val + "
            "0.10*fuzzy_f1_macro_val + 0.05*balance_clases - 0.05*gap - 0.03*compacidad"
        ),
        "nota": "El ranking usa validacion interna, no el 30% de prueba. Las metricas *_test son auditoria.",
        "mejores": [
            {
                "metricas": asdict(metricas),
                "reglas": serializar_reglas(reglas),
            }
            for metricas, reglas in mejores
        ],
    }
    ruta_json = RUTA_SALIDA / "mejores_reglas_ripper.json"
    ruta_json.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    filas = [asdict(metricas) for metricas, _ in evaluados]
    pd.DataFrame(filas).to_csv(RUTA_SALIDA / "ranking_reglas_ripper.csv", index=False)


def serializar_reglas(reglas):
    return [
        {
            "numero": int(i),
            "antecedentes": [list(ant) for ant in regla["antecedentes"]],
            "consecuente": regla["consecuente"],
        }
        for i, regla in enumerate(reglas, start=1)
    ]


def imprimir_resumen(mejores, total, args):
    print("=" * 88)
    print("Busqueda previa de reglas RIPPER")
    print("=" * 88)
    print(f"Semilla: {args.semilla}")
    print(f"Validacion interna: {args.fraccion_validacion:.0%} del 70% de entrenamiento")
    print(f"Candidatos validos evaluados: {total}")
    print(f"Reporte JSON: {RUTA_SALIDA / 'mejores_reglas_ripper.json'}")
    print(f"Ranking CSV:  {RUTA_SALIDA / 'ranking_reglas_ripper.csv'}")
    print()

    if not mejores:
        print("No se genero ningun conjunto de reglas valido.")
        return

    columnas = [
        "rank",
        "cand",
        "reglas",
        "fuzzy_ba_val",
        "hard_ba_val",
        "cob_val",
        "f1_val",
        "fuzzy_ba_test",
        "gap",
        "score",
        "clases",
    ]
    print(" | ".join(f"{c:>12}" for c in columnas))
    print("-" * 133)
    for rank, (metricas, _) in enumerate(mejores, start=1):
        clases = (
            f"B{metricas.reglas_por_clase['bajo']}/"
            f"M{metricas.reglas_por_clase['medio']}/"
            f"A{metricas.reglas_por_clase['alto']}"
        )
        fila = [
            rank,
            metricas.candidato,
            metricas.cantidad_reglas,
            f"{metricas.fuzzy_ba_val:.4f}",
            f"{metricas.hard_ba_val:.4f}",
            f"{metricas.cobertura_val:.4f}",
            f"{metricas.fuzzy_f1_macro_val:.4f}",
            f"{metricas.fuzzy_ba_test:.4f}",
            f"{metricas.gap_fuzzy_ba:.4f}",
            f"{metricas.puntaje_pre_ag:.4f}",
            clases,
        ]
        print(" | ".join(f"{str(v):>12}" for v in fila))

    mejor = mejores[0][0]
    print()
    print("Lectura rapida:")
    print(f"- Mejor candidato: {mejor.candidato} ({mejor.origen})")
    print(f"- Reglas por clase: {mejor.reglas_por_clase}")
    print(f"- El ranking usa validacion interna; el 30% de prueba queda solo como auditoria.")
    print(f"- Si fuzzy_ba_val esta claramente bajo 0.70, el AG parte con mala materia prima.")
    print(f"- Si cobertura_val es baja, faltan reglas que activen casos fuera del bootstrap.")


if __name__ == "__main__":
    principal()
