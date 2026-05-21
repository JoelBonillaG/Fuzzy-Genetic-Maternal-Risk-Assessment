# Backend de riesgo materno difuso

Este backend implementa un sistema de inferencia difusa Mamdani para clasificar riesgo materno usando reglas IF-THEN. El proyecto tambien incluye un pipeline experimental para comparar y procesar reglas generadas por RIPPER, PRISM y un algoritmo genetico tipo Michigan binario.

La salida del sistema es una de estas clases:

- `low risk`
- `mid risk`
- `high risk`

Si ninguna regla se activa, el sistema no inventa una clase ni usa un valor neutro. En ese caso devuelve:

- puntaje: `NaN` internamente, `null` en JSON
- riesgo: `None` internamente, `null` en JSON
- `sin_activacion: true`

Esto evita reportar falsamente `50 -> mid risk` cuando no existe evidencia difusa.

## Variables usadas

El dataset se carga desde:

```text
src/riesgo_materno/datos/Maternal Health Risk Data Set.csv
```

Variables de entrada:

- `edad`
- `presion_sistolica`
- `presion_diastolica`
- `azucar_sangre`
- `temperatura_corporal`
- `frecuencia_cardiaca`

## Estructura importante

```text
src/app/
```

Capa HTTP/FastAPI del proyecto.

```text
src/riesgo_materno/logica_difusa/
```

Motor Mamdani, variables difusas, salida difusa y carga de reglas activas.

Archivos clave:

- `variables.py`: funciones de pertenencia de entrada y salida.
- `motor.py`: fuzzificacion, evaluacion de reglas, agregacion y desfusificacion.
- `reglas.py`: carga las reglas que consume el motor web.

```text
src/riesgo_materno/optimizacion/
```

Algoritmos geneticos y variantes experimentales.

Subcarpetas importantes:

- `michigan_binario/`: AG actual para evolucionar reglas binarias.
- `pittsburgh/`: selector de subconjuntos de reglas.
- `pittsburgh_michigan/`: variante anterior/experimental.

```text
src/riesgo_materno/herramientas/pipeline_reglas/
```

Pipeline principal para experimentos de reglas.

Archivos clave:

- `experimento_reglas.py`: ejecuta RIPPER, PRISM y AG por iteraciones.
- `limpiar_reglas_iteraciones.py`: elimina reglas duplicadas y recalcula metricas.
- `preparar_reglas_web.py`: publica la mejor base limpia para que la use el sistema web.

```text
src/riesgo_materno/reglas/
```

Reglas limpias y reglas publicadas para produccion.

Archivos/carpetas clave:

- `limpias/vN/`: resultados limpios versionados.
- `reglas_sistema_difuso.json`: reglas activas que consume la web.
- `metadata_reglas_sistema_difuso.json`: metadata de la base publicada.

## Instalacion

Desde la carpeta `backend`:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si el entorno ya existe, solo activa:

```powershell
.\env\Scripts\Activate.ps1
```

## Levantar la API

Desde `backend`:

```powershell
python -m src.app.run
```

Health check:

```text
GET http://127.0.0.1:8000/health
```

## Prediccion individual por CLI

Desde `backend`:

```powershell
python -m src.riesgo_materno.herramientas.predecir_cli `
  --edad 28 `
  --presion-sistolica 120 `
  --presion-diastolica 80 `
  --azucar-sangre 7.5 `
  --temperatura-corporal 98.6 `
  --frecuencia-cardiaca 72
```

Si no se activa ninguna regla, la salida esperada es similar a:

```text
Puntaje de riesgo: NaN
Riesgo: sin clasificacion
Motivo: ninguna regla se activo para este perfil.
```

## Pipeline principal de reglas

Este es el flujo recomendado para generar, limpiar y publicar reglas.

### 1. Ejecutar experimento completo

Corre RIPPER, PRISM y AG Michigan binario usando el dataset completo, sin split entrenamiento/prueba.

```powershell
python -m src.riesgo_materno.herramientas.pipeline_reglas.experimento_reglas
```

Guarda resultados crudos en una version nueva:

```text
src/riesgo_materno/herramientas/pipeline_reglas/resultados/vN/
```

Al terminar, tambien ejecuta la limpieza automaticamente y guarda reglas limpias en:

```text
src/riesgo_materno/reglas/limpias/vN/
```

### 2. Limpiar reglas duplicadas manualmente

Usa esto si ya tienes resultados crudos y quieres recalcular metricas despues de limpiar duplicadas.

```powershell
python -m src.riesgo_materno.herramientas.pipeline_reglas.limpiar_reglas_iteraciones
```

Por defecto toma la ultima version disponible en:

```text
src/riesgo_materno/herramientas/pipeline_reglas/resultados/
```

Tambien puedes indicar una entrada concreta:

```powershell
python -m src.riesgo_materno.herramientas.pipeline_reglas.limpiar_reglas_iteraciones `
  --entrada src\riesgo_materno\herramientas\pipeline_reglas\resultados\v1
```

O limpiar solo un algoritmo:

```powershell
python -m src.riesgo_materno.herramientas.pipeline_reglas.limpiar_reglas_iteraciones `
  --algoritmo AG_MICHIGAN_BINARIO
```

Salidas generadas:

```text
src/riesgo_materno/reglas/limpias/vN/resumen_iteraciones.csv
src/riesgo_materno/reglas/limpias/vN/resumen_estadistico.csv
src/riesgo_materno/reglas/limpias/vN/resumen_metricas.json
src/riesgo_materno/reglas/limpias/vN/mejor_ag_limpio.json
src/riesgo_materno/reglas/limpias/vN/mejor_global_limpio.json
```

### 3. Publicar reglas para la web

Este comando toma la base limpia elegida y la copia al archivo que consume el motor difuso conectado al frontend.

```powershell
python -m src.riesgo_materno.herramientas.pipeline_reglas.preparar_reglas_web
```

Por defecto publica la iteracion 12 del AG si existe en la ultima version limpia.

Salidas:

```text
src/riesgo_materno/reglas/reglas_sistema_difuso.json
src/riesgo_materno/reglas/metadata_reglas_sistema_difuso.json
```

Para publicar la mejor iteracion entre todas:

```powershell
python -m src.riesgo_materno.herramientas.pipeline_reglas.preparar_reglas_web `
  --todas-iteraciones
```

Para publicar una iteracion especifica:

```powershell
python -m src.riesgo_materno.herramientas.pipeline_reglas.preparar_reglas_web `
  --iteracion 12
```

Para publicar desde una carpeta limpia especifica:

```powershell
python -m src.riesgo_materno.herramientas.pipeline_reglas.preparar_reglas_web `
  --entrada src\riesgo_materno\reglas\limpias\v1 `
  --iteracion 12
```

## Comandos rapidos

Activar entorno:

```powershell
.\env\Scripts\Activate.ps1
```

API:

```powershell
python -m src.app.run
```

Experimento completo:

```powershell
python -m src.riesgo_materno.herramientas.pipeline_reglas.experimento_reglas
```

Limpiar reglas:

```powershell
python -m src.riesgo_materno.herramientas.pipeline_reglas.limpiar_reglas_iteraciones
```

Publicar reglas para la web:

```powershell
python -m src.riesgo_materno.herramientas.pipeline_reglas.preparar_reglas_web
```

Prediccion CLI:

```powershell
python -m src.riesgo_materno.herramientas.predecir_cli `
  --edad 28 `
  --presion-sistolica 120 `
  --presion-diastolica 80 `
  --azucar-sangre 7.5 `
  --temperatura-corporal 98.6 `
  --frecuencia-cardiaca 72
```

## Frontend

Desde la carpeta `frontend`:

```powershell
npm install
npm run dev
```

Build:

```powershell
npm run build
```

Nota: si el build falla en `OptimizationSection.tsx`, esos errores pertenecen a tipos ya existentes en esa seccion y no al cambio del motor difuso.

## Notas metodologicas

- El motor ya no tiene modo neutro.
- Un caso sin activacion no se clasifica.
- En evaluacion experimental, `sin_activacion` se cuenta como error mediante la etiqueta interna `__sin_activacion__`.
- Las rutas guardadas en JSON del pipeline deben ser relativas desde `backend`, no rutas absolutas de la PC.
- No se debe volver a guardar resultados nuevos en `backend/limpieza`; esa carpeta era de pruebas antiguas.

## Orden recomendado para un experimento nuevo

1. Activar entorno.
2. Ejecutar `experimento_reglas`.
3. Revisar `src/riesgo_materno/reglas/limpias/vN/resumen_estadistico.csv`.
4. Revisar `mejor_ag_limpio.json` y `mejor_global_limpio.json`.
5. Publicar reglas con `preparar_reglas_web`.
6. Levantar API y probar desde el frontend.
