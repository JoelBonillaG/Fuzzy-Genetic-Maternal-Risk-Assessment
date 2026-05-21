"""Postprocesado supervisado de la frontera bajo/medio.

Desde backend:
    python -m src.riesgo_materno.postprocesado.postprocesado --reglas RUTA_JSON

Los resultados se guardan automaticamente en:
    src/riesgo_materno/postprocemiento/resultados/<nombre_del_archivo_de_reglas>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..entrenamiento.datos import cargar_dataset, convertir_split_a_diccionario
from ..entrenamiento.modelo import RUTA_CSV, RUTA_PAQUETE
from ..logica_difusa.motor import SistemaDifusoMamdani, puntaje_a_riesgo
from ..logica_difusa.variables import ESPECIFICACIONES_VARIABLES
from ..reportes.exportadores import guardar_csv, guardar_json, limpiar_pngs_raiz
from ..reportes.graficos import guardar_histograma_puntajes, guardar_matriz_confusion
from ..reportes.metricas import calcular_metricas


CLASES = ["low risk", "mid risk", "high risk"]
SIN_ACTIVACION = "__sin_activacion__"
CLASE_A_CONSECUENTE = {"low risk": "bajo", "mid risk": "medio", "high risk": "alto"}
DIRECCION_SUBE = "real_mid_pred_low"
DIRECCION_BAJA = "real_low_pred_mid"
DESCRIPCIONES_DIRECCION = {
    DIRECCION_SUBE: "Riesgo real: medio | Riesgo predicho: bajo",
    DIRECCION_BAJA: "Riesgo real: bajo | Riesgo predicho: medio",
}
GRUPOS_HISTOGRAMA = [
    {
        "campo": "direccion",
        "valor": DIRECCION_SUBE,
        "color": "#2CA02C",
        "etiqueta": "Real: Riesgo medio | Predicho: Riesgo bajo",
    },
    {
        "campo": "direccion",
        "valor": DIRECCION_BAJA,
        "color": "#F2C94C",
        "etiqueta": "Real: Riesgo bajo | Predicho: Riesgo medio",
    },
]
HISTOGRAMAS_DIRECCION = [
    (DIRECCION_SUBE, "puntaje_antes", "antes.png", "Antes del postprocesado"),
    (DIRECCION_SUBE, "puntaje_despues", "despues.png", "Despues del postprocesado"),
    (DIRECCION_BAJA, "puntaje_antes", "antes.png", "Antes del postprocesado"),
    (DIRECCION_BAJA, "puntaje_despues", "despues.png", "Despues del postprocesado"),
]
VARIABLES = [
    "edad",
    "presion_sistolica",
    "presion_diastolica",
    "azucar_sangre",
    "temperatura_corporal",
    "frecuencia_cardiaca",
]


@dataclass(frozen=True)
class Configuracion:
    reglas: Path
    salida: Path
    limite_inferior: float = 36.95
    limite_superior: float = 42.97
    corte: float = 39.92

    @property
    def delta_subir(self) -> float:
        return self.limite_superior - self.corte

    @property
    def delta_bajar(self) -> float:
        return self.corte - self.limite_inferior


class ExperimentoPostprocesado:
    def __init__(self, config: Configuracion):
        self.config = config
        self.tabla = cargar_dataset(RUTA_CSV)
        self.reglas = cargar_reglas(config.reglas)

    def ejecutar(self) -> dict:
        inferencia = self._inferir()
        ajuste = self._postprocesar(inferencia)
        resultado = {
            "reglas": len(self.reglas),
            "frontera": {
                "limite_inferior": self.config.limite_inferior,
                "limite_superior": self.config.limite_superior,
                "corte": self.config.corte,
                "delta_subir": self.config.delta_subir,
                "delta_bajar": self.config.delta_bajar,
            },
            "antes": metricas_clasificacion(
                inferencia["reales"],
                inferencia["predichos"],
                inferencia["sin_activacion"],
            ),
            "despues": metricas_clasificacion(
                inferencia["reales"],
                ajuste["predichos"],
                inferencia["sin_activacion"],
            ),
            "casos": ajuste["casos"],
        }
        resultado["delta"] = {
            "accuracy": resultado["despues"]["accuracy"] - resultado["antes"]["accuracy"],
            "balanced_accuracy": (
                resultado["despues"]["balanced_accuracy"] - resultado["antes"]["balanced_accuracy"]
            ),
            "aciertos": resultado["despues"]["aciertos"] - resultado["antes"]["aciertos"],
        }
        return resultado

    def _inferir(self) -> dict:
        datos = convertir_split_a_diccionario(self.tabla)
        sistema = SistemaDifusoMamdani(membresias_base(), reglas=self.reglas)
        salida = sistema.inferir_lote(datos["entradas"])
        return {
            "reales": np.asarray(datos["riesgos"], dtype=object),
            "predichos": np.asarray(salida["riesgos"], dtype=object),
            "puntajes": np.asarray(salida["puntajes"], dtype=float),
            "sin_activacion": np.asarray(salida["sin_activacion"], dtype=bool),
        }

    def _postprocesar(self, inferencia: dict) -> dict:
        reales = inferencia["reales"]
        predichos = inferencia["predichos"].copy()
        puntajes = inferencia["puntajes"]
        nuevos_puntajes = puntajes.copy()

        en_frontera = (
            (puntajes >= self.config.limite_inferior)
            & (puntajes <= self.config.limite_superior)
            & ~inferencia["sin_activacion"]
        )
        
        subir = (reales == "mid risk") & (predichos == "low risk") & en_frontera
        bajar = (reales == "low risk") & (predichos == "mid risk") & en_frontera

        nuevos_puntajes[subir] += self.config.delta_subir
        nuevos_puntajes[bajar] -= self.config.delta_bajar
        nuevos_puntajes = np.clip(nuevos_puntajes, 0.0, 100.0)

        for indice in np.where(subir | bajar)[0]:
            predichos[indice] = puntaje_a_riesgo(nuevos_puntajes[indice])

        return {
            "predichos": predichos,
            "puntajes": nuevos_puntajes,
            "casos": self._casos(inferencia, nuevos_puntajes, predichos, subir, bajar),
        }

    def _casos(self, inferencia, nuevos_puntajes, nuevos_predichos, subir, bajar) -> list[dict]:
        casos = []
        for indice in np.where(subir | bajar)[0]:
            fila = self.tabla.iloc[int(indice)]
            direccion = DIRECCION_SUBE if subir[indice] else DIRECCION_BAJA
            caso = {
                "indice": int(indice),
                "direccion": direccion,
                "y_true": str(inferencia["reales"][indice]),
                "y_pred_antes": str(inferencia["predichos"][indice]),
                "y_pred_despues": str(nuevos_predichos[indice]),
                "puntaje_antes": round(float(inferencia["puntajes"][indice]), 2),
                "puntaje_despues": round(float(nuevos_puntajes[indice]), 2),
                "delta": round(float(nuevos_puntajes[indice] - inferencia["puntajes"][indice]), 2),
            }
            for variable in VARIABLES:
                caso[variable] = float(fila[variable])
            casos.append(caso)
        return casos


class Reporte:
    def __init__(self, carpeta: Path):
        self.carpeta = carpeta

    def guardar(self, resultado: dict):
        self.carpeta.mkdir(parents=True, exist_ok=True)
        limpiar_pngs_raiz(self.carpeta)
        matrices = self.carpeta / "imagenes" / "matrices"
        histogramas = self.carpeta / "imagenes" / "histogramas"
        limite_y = limite_y_histogramas(resultado["casos"])

        guardar_json(self.carpeta / "resultado.json", sin_casos(resultado))
        guardar_csv(self.carpeta / "casos_postprocesados.csv", resultado["casos"])
        guardar_matriz_confusion(matrices / "antes.png", resultado["antes"], "Antes")
        guardar_matriz_confusion(matrices / "despues.png", resultado["despues"], "Despues")
        guardar_histograma_postprocesado(
            histogramas / "general" / "antes.png",
            resultado["casos"],
            "puntaje_antes",
            "Antes",
            limite_y,
        )
        guardar_histograma_postprocesado(
            histogramas / "general" / "despues.png",
            resultado["casos"],
            "puntaje_despues",
            "Despues",
            limite_y,
        )
        guardar_histogramas_por_direccion(histogramas, resultado["casos"], limite_y)


def cargar_reglas(ruta: Path) -> list[dict]:
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    reglas_crudas = contenido.get("reglas_finales", contenido)
    reglas = []
    for numero, regla in enumerate(reglas_crudas, start=1):
        reglas.append({
            "numero": numero,
            "antecedentes": [
                (a["variable"], a.get("etiqueta_linguistica", a.get("categoria")))
                for a in regla["antecedentes"]
            ],
            "consecuente": CLASE_A_CONSECUENTE.get(regla["consecuente"], regla["consecuente"]),
        })
    return reglas


def membresias_base() -> dict:
    return {
        variable: {
            categoria: np.asarray(puntos, dtype=float)
            for categoria, puntos in especificacion["categorias"].items()
        }
        for variable, especificacion in ESPECIFICACIONES_VARIABLES.items()
    }


def metricas_clasificacion(reales, predichos, sin_activacion) -> dict:
    return calcular_metricas(
        reales,
        predichos,
        clases=CLASES,
        sin_activacion=sin_activacion,
        etiqueta_sin=SIN_ACTIVACION,
    )


def guardar_histogramas_por_direccion(carpeta: Path, casos: list[dict], limite_y: int):
    for direccion, campo, archivo, titulo in HISTOGRAMAS_DIRECCION:
        casos_direccion = [caso for caso in casos if caso["direccion"] == direccion]
        guardar_histograma_postprocesado(carpeta / direccion / archivo, casos_direccion, campo, titulo, limite_y)


def guardar_histograma_postprocesado(ruta: Path, casos: list[dict], campo: str, titulo: str, limite_y: int):
    guardar_histograma_puntajes(
        ruta=ruta,
        casos=casos,
        campo=campo,
        titulo=titulo,
        descripcion=descripcion_casos(casos),
        grupos=GRUPOS_HISTOGRAMA,
        intervalo_confianza=(36.95, 42.97),
        corte=39.92,
        limite_y=limite_y,
    )


def limite_y_histogramas(casos: list[dict]) -> int:
    if not casos:
        return 1
    bins = np.linspace(36.95, 42.97, 17)
    maximo = 0
    for campo in ("puntaje_antes", "puntaje_despues"):
        for grupo in GRUPOS_HISTOGRAMA:
            valores = [caso[campo] for caso in casos if caso["direccion"] == grupo["valor"]]
            conteos, _ = np.histogram(valores, bins=bins)
            maximo = max(maximo, int(conteos.max()) if len(conteos) else 0)
    return maximo + 1


def descripcion_casos(casos: list[dict]) -> str:
    direcciones = {caso["direccion"] for caso in casos}
    if len(direcciones) == 1:
        return DESCRIPCIONES_DIRECCION[next(iter(direcciones))]
    return "Casos de frontera bajo-medio"


def sin_casos(resultado: dict) -> dict:
    copia = dict(resultado)
    copia["casos_postprocesados"] = len(resultado["casos"])
    del copia["casos"]
    return copia


def resolver_ruta(texto: str) -> Path:
    ruta = Path(texto)
    if ruta.exists():
        return ruta
    ruta_desde_backend = RUTA_PAQUETE.parents[1] / texto
    if ruta_desde_backend.exists():
        return ruta_desde_backend
    raise FileNotFoundError(f"No existe el archivo de reglas: {texto}")


def crear_configuracion() -> Configuracion:
    parser = argparse.ArgumentParser(description="Postprocesado supervisado bajo/medio.")
    parser.add_argument("--reglas", required=True)
    parser.add_argument("--minimo", type=float, default=36.95)
    parser.add_argument("--maximo", type=float, default=42.97)
    parser.add_argument("--corte", type=float, default=39.92)
    args = parser.parse_args()

    reglas = resolver_ruta(args.reglas)
    salida = Path(__file__).resolve().parent / "resultados" / reglas.stem
    return Configuracion(
        reglas=reglas,
        salida=salida,
        limite_inferior=args.minimo,
        limite_superior=args.maximo,
        corte=args.corte,
    )


def main():
    config = crear_configuracion()
    resultado = ExperimentoPostprocesado(config).ejecutar()
    Reporte(config.salida).guardar(resultado)
    print(f"Reglas: {resultado['reglas']}")
    print(f"Casos postprocesados: {len(resultado['casos'])}")
    print(f"Accuracy: {resultado['antes']['accuracy']:.4f} -> {resultado['despues']['accuracy']:.4f}")
    print(f"BA: {resultado['antes']['balanced_accuracy']:.4f} -> {resultado['despues']['balanced_accuracy']:.4f}")
    print(f"Resultados: {config.salida.resolve()}")


if __name__ == "__main__":
    main()
