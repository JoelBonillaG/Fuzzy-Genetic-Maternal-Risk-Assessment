# Backend de `riesgo_materno`

Este backend implementa un sistema difuso tipo Mamdani para estimar riesgo materno a partir de variables clinicas. El proyecto tambien incorpora un algoritmo genetico que ajusta las funciones de pertenencia de entrada y persiste el mejor cromosoma encontrado en disco.

## Objetivo

El sistema trabaja con estas entradas:

- `edad`
- `presion_sistolica`
- `presion_diastolica`
- `azucar_sangre`
- `temperatura_corporal`
- `frecuencia_cardiaca`

La salida final es una etiqueta de riesgo:

- `low risk`
- `mid risk`
- `high risk`

El flujo general es el siguiente:

1. El dataset CSV se carga desde `src/riesgo_materno/datos/`.
2. La capa de entrenamiento limpia, valida y divide los datos.
3. El sistema difuso base se evalua sobre los splits definidos.
4. El algoritmo genetico optimiza las membresias de entrada.
5. El mejor cromosoma se guarda en `src/riesgo_materno/modelos/modelo_optimizado.json`.
6. La capa de prediccion reutiliza ese modelo persistido; si no existe, lo entrena de nuevo desde el CSV.

## Responsabilidad por modulo

### `src/app/`

Contiene la capa HTTP del proyecto.

- `main.py`: crea la aplicacion FastAPI.
- `run.py`: levanta `uvicorn` usando la configuracion central.
- `core/config.py`: define nombre, puerto, host y otras variables de entorno.

### `src/riesgo_materno/logica_difusa/`

Contiene la definicion del sistema Mamdani.

- `variables.py`: universos, categorias difusas, limites y salida difusa.
- `reglas.py`: conocimiento experto expresado como reglas.
- `motor.py`: inferencia, agregacion y desfusificacion.

### `src/riesgo_materno/optimizacion/`

Contiene la optimizacion de las membresias.

- `algoritmo_genetico.py`: ejecucion del AG.
- `cromosoma.py`: codificacion, decodificacion y reparacion del cromosoma.
- `penalizaciones.py`: restricciones y penalizaciones del fitness.

### `src/riesgo_materno/entrenamiento/`

Contiene la orquestacion de datos, evaluacion y persistencia.

- `datos.py`: carga, limpieza y split estratificado del dataset.
- `evaluacion.py`: evaluacion del sistema sobre entrenamiento, validacion y prueba.
- `metricas.py`: metricas de clasificacion.
- `modelo.py`: rutas, parametros del AG y proporciones de split.
- `entrenador.py`: entrena, compara y guarda el mejor modelo.

### `src/riesgo_materno/prediccion/`

Contiene el caso de uso de prediccion.

- `predictor.py`: arma la prediccion completa a partir de una entrada nueva.
- `validacion_entrada.py`: valida valores y satura pequenas desviaciones cercanas al rango permitido.

### `src/riesgo_materno/herramientas/`

Agrupa los scripts operativos del proyecto.

- `predecir_cli.py`: prediccion individual por linea de comandos.
- `optimizar_mamdani_ag.py`: reentrenamiento forzado y reporte de comparacion.
- `pruebas_algoritmos.py`: ejecucion masiva desde CSV y exportacion a texto.

## Rutas importantes

- Dataset: `src/riesgo_materno/datos/Maternal Health Risk Data Set.csv`
- Modelo persistido: `src/riesgo_materno/modelos/modelo_optimizado.json`
- Documento tecnico complementario: `docs/roadmap_codigo_riesgo_materno.md`

## Requisitos

Las dependencias declaradas actualmente son:

- `numpy`
- `pandas`
- `scikit-fuzzy`
- `pygad`
- `scikit-learn`
- `fastapi[standard]`

## Instalacion

La ejecucion esperada parte desde la carpeta `backend`.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecucion

### Levantar la API

```bash
python -m src.app.run
```

Endpoint disponible:

```text
GET http://127.0.0.1:8000/health
```

### Ejecutar una prediccion individual por CLI

```bash
python -m src.riesgo_materno.herramientas.predecir_cli ^
  --edad 28 ^
  --presion-sistolica 120 ^
  --presion-diastolica 80 ^
  --azucar-sangre 7.5 ^
  --temperatura-corporal 98.6 ^
  --frecuencia-cardiaca 72
```

### Reentrenar y guardar el modelo optimizado

```bash
python -m src.riesgo_materno.herramientas.optimizar_mamdani_ag
```

## Flujo de prediccion actual

1. `predecir_caso(...)` solicita las membresias optimizadas.
2. Si `modelo_optimizado.json` existe y es valido, el sistema lo reutiliza.
3. Si el archivo no existe o es invalido, `entrenador.py` vuelve a entrenar desde el CSV.
4. `validacion_entrada.py` valida cada valor y ajusta pequenas desviaciones cercanas al limite permitido.
5. `SistemaDifusoMamdani` calcula el puntaje difuso y lo convierte a una clase de riesgo.

## Estado actual de la arquitectura

La arquitectura vigente ya separa responsabilidades de forma clara:

- `src/app` administra la configuracion y la exposicion HTTP.
- `src/riesgo_materno` concentra el dominio del problema.
- `docs/` conserva documentacion tecnica adicional.

El README anterior describia una organizacion mas antigua. Esta version documenta la estructura real presente hoy en el repositorio.
