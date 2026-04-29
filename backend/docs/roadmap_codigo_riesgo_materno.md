# Roadmap Del Codigo De `riesgo_materno`

## 1. Objetivo de este documento

Este documento existe para responder una pregunta concreta:

> "Si alguien me entrega este proyecto y yo no quiero asumir nada, en que orden debo leerlo para entenderlo y poder explicarlo?"

La idea no es solo listar archivos. La idea es entender:

1. cual es el punto de entrada;
2. cual es la clase o funcion que hace el trabajo real;
3. por que cada archivo esta en la carpeta donde esta;
4. como viaja un caso desde los datos de entrada hasta el resultado final;
5. como se entrena el modelo optimizado que luego consume la prediccion.

Este documento parte desde cero. No asume que ya sabes que es cada carpeta.

---

## 2. Vista general del paquete

El codigo central esta en:

```text
src/
└─ riesgo_materno/
   ├─ datos/
   ├─ modelos/
   ├─ herramientas/
   ├─ logica_difusa/
   ├─ optimizacion/
   ├─ entrenamiento/
   └─ prediccion/
```

Cada carpeta responde una pregunta distinta.

### `datos/`

Pregunta que responde:

> "De donde sale el dataset original?"

Contiene el archivo CSV crudo del problema.

Aqui esta:

- `src/riesgo_materno/datos/Maternal Health Risk Data Set.csv`

Esta carpeta existe porque el dataset no es codigo. Es un insumo del sistema.

### `modelos/`

Pregunta que responde:

> "Donde se guarda el resultado persistido del entrenamiento?"

Contiene el archivo JSON del mejor modelo optimizado.

Aqui esta:

- `src/riesgo_materno/modelos/modelo_optimizado.json`

Esta carpeta existe porque el modelo persistido tampoco es codigo fuente. Es un artefacto generado.

### `herramientas/`

Pregunta que responde:

> "Con que scripts ejecuto el sistema manualmente?"

Contiene scripts de uso operativo:

- `predecir_cli.py`
- `optimizar_mamdani_ag.py`
- `pruebas_algoritmos.py`

Esta carpeta existe porque estos archivos no son el nucleo del negocio. Son formas de invocar el nucleo.

### `logica_difusa/`

Pregunta que responde:

> "Cual es la definicion del sistema difuso?"

Aqui viven:

- las variables linguisticas;
- las funciones de pertenencia;
- las reglas;
- el motor Mamdani.

Esta carpeta existe porque esa es la parte teorica y operativa del sistema difuso en si.

### `optimizacion/`

Pregunta que responde:

> "Como se mejoran las funciones de pertenencia?"

Aqui viven:

- el algoritmo genetico;
- la representacion del cromosoma;
- las penalizaciones;
- la logica para reparar soluciones.

Esta carpeta existe porque optimizar no es lo mismo que predecir. Son dos responsabilidades distintas.

### `entrenamiento/`

Pregunta que responde:

> "Como se pasa del CSV a un modelo optimizado reutilizable?"

Aqui viven:

- la carga del dataset;
- el split entrenamiento-validacion-prueba;
- la evaluacion;
- las metricas;
- la orquestacion del entrenamiento;
- las rutas del CSV y del modelo persistido.

Esta carpeta existe porque entrenar y evaluar el sistema es otra responsabilidad distinta del motor difuso.

### `prediccion/`

Pregunta que responde:

> "Como se toma un caso nuevo y se devuelve un riesgo?"

Aqui viven:

- la validacion de entradas;
- la funcion que arma la llamada al motor y devuelve el resultado final.

Esta carpeta existe porque predecir es un caso de uso distinto al entrenamiento.

---

## 3. Flujo completo del sistema, de punta a punta

Si quieres explicar el sistema como historia, la historia correcta es esta:

1. un usuario entrega valores clinicos;
2. una herramienta de entrada recibe esos valores;
3. la capa de prediccion valida y ajusta la entrada;
4. la capa de entrenamiento carga el modelo optimizado persistido;
5. si el modelo no existe, se entrena desde el CSV;
6. se construye el motor difuso con las membresias optimizadas;
7. el motor aplica fusificacion, reglas y desfusificacion;
8. se obtiene un puntaje;
9. ese puntaje se convierte en `low risk`, `mid risk` o `high risk`;
10. la herramienta muestra o devuelve el resultado.

Eso es el flujo de prediccion.

El flujo de entrenamiento es otro:

1. se lee el CSV;
2. se validan columnas y etiquetas;
3. se limpian registros anormales;
4. se divide el dataset;
5. se evalua el sistema base;
6. se ejecuta el algoritmo genetico;
7. se obtiene el mejor cromosoma;
8. ese cromosoma se decodifica a nuevas membresias;
9. se reevalua el sistema optimizado;
10. se guarda el mejor cromosoma en JSON.

---

## 4. Punto de entrada para entender la prediccion

Si alguien te pregunta:

> "Desde donde empieza una prediccion individual?"

Empieza aqui:

- `src/riesgo_materno/herramientas/predecir_cli.py`

### Que hace este archivo

Este script:

1. recibe argumentos por linea de comandos;
2. si faltan valores, los pide al usuario;
3. llama a `predecir_caso(...)`;
4. imprime el resultado final.

### Por que esta en `herramientas/`

Porque no contiene la logica del negocio. Solo la usa.

Si mañana eliminas la CLI y la reemplazas por FastAPI, el negocio no deberia romperse. Por eso este archivo no debe vivir en `prediccion/` ni en `logica_difusa/`.

### Que funcion mirar primero

Empieza por:

- `principal()`

Luego sigue con:

- `recoger_valores(...)`
- `crear_parser()`

La funcion importante no esta aqui. Aqui solo se prepara la llamada.

---

## 5. Caso de uso principal de prediccion

Despues de `predecir_cli.py`, el siguiente archivo es:

- `src/riesgo_materno/prediccion/predictor.py`

### Que hace este archivo

Este archivo responde la pregunta:

> "Dado un caso nuevo, como produzco la prediccion?"

La funcion central es:

- `predecir_caso(valores_entrada)`

### Que hace `predecir_caso`

En orden:

1. pide las membresias optimizadas con `obtener_membresias_optimizadas()`;
2. construye un `SistemaDifusoMamdani`;
3. valida y prepara la entrada con `construir_entrada_lote(...)`;
4. llama `inferir_lote(...)` del motor;
5. devuelve un diccionario con puntaje, riesgo, origen del modelo y ajustes realizados.

### Por que esta en `prediccion/`

Porque este archivo no define el sistema difuso ni lo entrena. Solo ejecuta el caso de uso "predecir un caso".

Eso lo convierte en una capa de orquestacion ligera entre:

- la entrada del usuario;
- el modelo optimizado;
- el motor difuso.

---

## 6. Validacion de entrada antes de inferir

El siguiente archivo es:

- `src/riesgo_materno/prediccion/validacion_entrada.py`

### Que hace este archivo

Responde la pregunta:

> "Antes de llamar al motor, como sabemos que los valores son aceptables?"

Funciones principales:

- `construir_entrada_lote(...)`
- `validar_valores_entrada(...)`
- `saturar_si_esta_cerca_del_limite(...)`

### Que significa "saturar"

No se asume que cualquier valor fuera de rango deba rechazarse de inmediato.

El codigo hace una decision de negocio concreta:

- si el valor esta ligeramente fuera del rango permitido, se ajusta al limite;
- si esta demasiado fuera, se rechaza.

Ejemplo conceptual:

- si una variable admite hasta `100` y llega `102`, puede ajustarse a `100`;
- si llega `140`, se rechaza.

### Por que esta en `prediccion/`

Porque esta validacion aplica al caso de uso de inferencia con entradas reales. No pertenece al algoritmo genetico ni al motor Mamdani puro.

---

## 7. La clase central del sistema

Si alguien pregunta:

> "Cual es la clase mas importante del proyecto?"

La respuesta mas correcta es:

- `SistemaDifusoMamdani`

Y esta aqui:

- `src/riesgo_materno/logica_difusa/motor.py`

### Por que esta clase es central

Porque es la clase que realmente transforma entradas numericas en un puntaje de riesgo.

No imprime.
No lee CSV.
No hace parsing de argumentos.
No entrena.

Solo hace inferencia difusa.

### Que debes leer primero dentro de `motor.py`

En este orden:

1. `__init__(...)`
2. `inferir_lote(...)`
3. `_desfusificar(...)`
4. `_compilar_reglas(...)`
5. `_crear_universos_entrada(...)`
6. `_crear_curvas_entrada(...)`
7. `_crear_curvas_salida(...)`
8. `puntaje_a_riesgo(...)`

### Que hace `inferir_lote(...)`

Esta funcion implementa el flujo del motor:

1. fusifica todas las entradas;
2. evalua reglas;
3. agrega activaciones de salida;
4. desfusifica cada caso;
5. convierte el puntaje a una etiqueta.

### Por que esta en `logica_difusa/`

Porque esto ya no es "uso del sistema". Esto es "definicion y ejecucion del sistema difuso".

---

## 8. Donde se define el mundo del sistema difuso

Antes de entender bien el motor, debes leer:

- `src/riesgo_materno/logica_difusa/variables.py`

### Que hace este archivo

Define:

- `VARIABLES_ENTRADA`
- `ETIQUETAS_RIESGO`
- `SALIDA_DIFUSA`
- `ESPECIFICACIONES_VARIABLES`
- `PUNTOS_SALIDA`
- `PUNTOS_GRAFICA`

### Que significa eso

Este archivo responde:

> "Con que variables trabaja el sistema y como se representa cada una?"

Aqui se define:

- el rango de cada variable;
- las categorias linguisticas de cada variable;
- los trapecios iniciales de membresia;
- la forma de la salida difusa.

### Por que esta en `logica_difusa/`

Porque esto es parte de la definicion del modelo difuso.

No es configuracion tecnica del proyecto como puertos, logs o entorno. Es configuracion del dominio difuso.

---

## 9. Donde se define el conocimiento experto

El otro archivo clave del motor es:

- `src/riesgo_materno/logica_difusa/reglas.py`

### Que hace este archivo

Contiene la lista `REGLAS`.

Cada regla tiene:

- antecedentes;
- consecuente.

Ejemplo conceptual:

> "Si azucar esta alta y presion esta elevada, entonces el riesgo es alto."

### Por que esta en `logica_difusa/`

Porque las reglas forman parte del conocimiento del sistema difuso.

Si las reglas cambiaran, cambiaria el comportamiento del sistema, aunque no cambies ni la CLI ni el entrenamiento.

---

## 10. Como se consigue el modelo optimizado

Hasta ahora vimos como se usa el modelo.

Ahora toca responder:

> "De donde salen las membresias optimizadas que usa la prediccion?"

El archivo central es:

- `src/riesgo_materno/entrenamiento/entrenador.py`

### Funcion principal

- `obtener_membresias_optimizadas()`

### Que hace `obtener_membresias_optimizadas()`

1. intenta leer `modelo_optimizado.json`;
2. si existe y es valido, lo usa;
3. si no existe, entrena desde el CSV;
4. devuelve las membresias listas para usar.

### Que hace `entrenar_y_guardar()`

1. carga el dataset;
2. divide en splits;
3. evalua membresias base;
4. corre el algoritmo genetico;
5. decodifica el mejor cromosoma;
6. compara base vs optimizado;
7. guarda el modelo persistido.

### Por que esta en `entrenamiento/`

Porque aqui no se esta ejecutando el motor sobre un caso nuevo, sino construyendo el modelo optimizado que luego otros consumen.

---

## 11. Carga y preparacion de datos

El siguiente archivo es:

- `src/riesgo_materno/entrenamiento/datos.py`

### Que hace este archivo

Responde:

> "Como se lee el CSV y se transforma a un formato util para el sistema?"

Funciones importantes:

- `cargar_datos(...)`
- `dividir_datos_estratificados(...)`
- `convertir_split_a_diccionario(...)`
- `quitar_registros_con_frecuencia_cardiaca_erronea(...)`

### Que debes poder explicar aqui

1. el CSV se valida antes de usarlo;
2. se renombran las columnas al esquema interno;
3. se valida que las etiquetas de riesgo sean conocidas;
4. se eliminan registros anomalos concretos;
5. se divide el dataset preservando la distribucion por clase.

### Por que esta en `entrenamiento/`

Porque esta preparacion de datos se usa para entrenar y evaluar.

No forma parte de la inferencia de un caso individual.

---

## 12. Evaluacion del sistema

Luego debes leer:

- `src/riesgo_materno/entrenamiento/evaluacion.py`

### Que hace este archivo

Responde:

> "Una vez que tengo un sistema difuso, como lo pruebo sobre entrenamiento, validacion o prueba?"

Funciones importantes:

- `evaluar_cromosoma_en_splits(...)`
- `evaluar_membresias_en_splits(...)`
- `evaluar_sistema_en_split(...)`
- `crear_tabla_comparativa(...)`

### Por que esta en `entrenamiento/`

Porque evaluar un modelo no es lo mismo que usarlo en produccion.

Este codigo existe para medir calidad, no para servir una prediccion a un usuario final.

---

## 13. Metricas del problema

El archivo complementario es:

- `src/riesgo_materno/entrenamiento/metricas.py`

### Que hace este archivo

Responde:

> "Con que numeros digo si el sistema funciona bien o mal?"

Aqui estan funciones como:

- `calcular_macro_f1(...)`
- `calcular_recall_de_riesgo_alto(...)`
- `calcular_exactitud_balanceada(...)`
- `crear_matriz_confusion(...)`
- `crear_reporte_clasificacion_tabla(...)`

### Por que esta en `entrenamiento/`

Porque estas metricas se usan para evaluar durante desarrollo y entrenamiento.

No son parte del motor difuso puro.

---

## 14. Configuracion de entrenamiento y persistencia

Debes leer tambien:

- `src/riesgo_materno/entrenamiento/modelo.py`

### Que hace este archivo

Guarda:

- la ruta del CSV;
- la ruta del modelo optimizado;
- el mapa de columnas del CSV;
- los parametros del algoritmo genetico;
- las proporciones del split.

### Por que esta en `entrenamiento/`

Porque estas configuraciones no describen el sistema difuso en abstracto. Describen como se entrena, como se leen los datos y donde se persiste el resultado.

---

## 15. Donde entra el algoritmo genetico

Una vez que entiendes entrenamiento, pasas a:

- `src/riesgo_materno/optimizacion/algoritmo_genetico.py`

### Que hace este archivo

Responde:

> "Como se buscan mejores funciones de pertenencia?"

La pieza estructural mas importante es:

- `Individuo`

Y la funcion principal es:

- `ejecutar_algoritmo_genetico(datos_validacion)`

### Que debes poder decir de este modulo

1. un individuo representa un cromosoma candidato;
2. cada cromosoma codifica funciones de pertenencia;
3. cada individuo se evalua creando un `SistemaDifusoMamdani`;
4. el fitness mezcla calidad predictiva y penalizaciones;
5. el algoritmo genetico itera hasta mejorar o agotar paciencia.

### Por que esta en `optimizacion/`

Porque este codigo no define el sistema difuso ni sirve predicciones al usuario. Su trabajo es ajustar parametros del sistema.

---

## 16. Representacion del cromosoma

El archivo siguiente es:

- `src/riesgo_materno/optimizacion/cromosoma.py`

### Que hace este archivo

Responde:

> "Como se convierte entre membresias difusas y un vector numerico optimizable?"

Funciones importantes:

- `decodificar_cromosoma(...)`
- `aplanar_membresias(...)`
- `reparar_cromosoma(...)`
- `crear_membresias_base()`

Tambien define:

- `MEMBRESIAS_BASE`
- `CROMOSOMA_BASE`
- limites y rangos de genes

### Por que esta en `optimizacion/`

Porque esta es la traduccion entre el lenguaje del algoritmo genetico y el lenguaje del sistema difuso.

---

## 17. Penalizaciones del fitness

El archivo complementario es:

- `src/riesgo_materno/optimizacion/penalizaciones.py`

### Que hace este archivo

Responde:

> "Como evitamos que el algoritmo encuentre soluciones numericamente buenas pero semanticamente malas?"

Aqui se penaliza:

- soporte demasiado pequeno;
- huecos;
- solapamientos excesivos;
- desorden entre categorias;
- desviacion excesiva respecto a la base.

### Por que esta en `optimizacion/`

Porque estas reglas no son inferencia ni carga de datos. Son criterios para guiar la busqueda del AG.

---

## 18. Herramientas operativas y por que estan separadas

### `herramientas/predecir_cli.py`

Sirve para prediccion manual.

No esta en `prediccion/` porque solo consume la capa de prediccion.

### `herramientas/optimizar_mamdani_ag.py`

Sirve para forzar reentrenamiento y mostrar resultados.

No esta en `entrenamiento/` porque no es la libreria de entrenamiento: es un script para ejecutar esa libreria.

### `herramientas/pruebas_algoritmos.py`

Sirve para experimentos o corridas de comprobacion desde CSV.

No esta en `tests/` porque no es un test automatizado de framework. Es una herramienta operativa/manual.

---

## 19. Cual es el orden correcto para estudiar el proyecto

Si quieres aprenderlo sin perderte, lee en este orden:

1. `src/riesgo_materno/herramientas/predecir_cli.py`
2. `src/riesgo_materno/prediccion/predictor.py`
3. `src/riesgo_materno/prediccion/validacion_entrada.py`
4. `src/riesgo_materno/logica_difusa/motor.py`
5. `src/riesgo_materno/logica_difusa/variables.py`
6. `src/riesgo_materno/logica_difusa/reglas.py`
7. `src/riesgo_materno/entrenamiento/entrenador.py`
8. `src/riesgo_materno/entrenamiento/datos.py`
9. `src/riesgo_materno/entrenamiento/evaluacion.py`
10. `src/riesgo_materno/entrenamiento/metricas.py`
11. `src/riesgo_materno/entrenamiento/modelo.py`
12. `src/riesgo_materno/optimizacion/algoritmo_genetico.py`
13. `src/riesgo_materno/optimizacion/cromosoma.py`
14. `src/riesgo_materno/optimizacion/penalizaciones.py`
15. `src/riesgo_materno/herramientas/optimizar_mamdani_ag.py`
16. `src/riesgo_materno/herramientas/pruebas_algoritmos.py`

Ese orden funciona porque sigue el camino natural de una pregunta:

1. como se usa;
2. como se predice;
3. como funciona el motor;
4. como se entrena;
5. como se optimiza;
6. con que herramientas se ejecuta.

---

## 20. Cual es la explicacion corta que deberias poder dar

Si tuvieras que explicarlo en poco tiempo, esta seria una buena version:

> El paquete `riesgo_materno` separa responsabilidades. `prediccion` contiene el caso de uso de inferencia. `logica_difusa` contiene la definicion del sistema Mamdani: variables, reglas y motor. `entrenamiento` se encarga de cargar datos, evaluar, entrenar y persistir el mejor modelo. `optimizacion` implementa el algoritmo genetico que ajusta las funciones de pertenencia. `herramientas` contiene scripts de ejecucion manual como CLI y reentrenamiento. La prediccion final toma entradas clinicas, las valida, carga las membresias optimizadas, construye el motor difuso, calcula un puntaje y lo transforma en una clase de riesgo.

---

## 21. Que deberia consumir FastAPI despues

Cuando construyas la API, el endpoint no deberia hablar directo con el CSV ni con el algoritmo genetico.

Lo correcto es que la API consuma:

- `src/riesgo_materno/prediccion/predictor.py`

Mas especificamente:

- `predecir_caso(...)`

Eso mantiene una separacion sana:

- FastAPI maneja HTTP;
- `prediccion` maneja el caso de uso;
- `logica_difusa` maneja la inferencia;
- `entrenamiento` maneja el modelo optimizado.

Esa separacion es importante porque evita que el endpoint quede acoplado a demasiadas capas a la vez.
