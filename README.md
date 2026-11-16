# Trafico-CDMX

Predicción de congestión vial en avenidas cercanas a Ciudad Universitaria con PySpark ML.

Proyecto final de Inteligencia Artificial (406), Grupo 07, Facultad de Ingeniería, UNAM, semestre 2027-1.

- Hernández Fierros Sonia
- Romero Molina David

## La pregunta

Con lo que sabemos ahora de un punto (la hora, el día y su velocidad en los últimos 20 minutos), ¿podemos saber si va a estar congestionado dentro de 10 minutos? Y si se puede, ¿lo predice mejor un perceptrón o una cadena de Markov?

## Los datos

Armamos nuestro propio dataset con la API Traffic Flow Segment Data de TomTom. Un workflow de GitHub Actions consultó 16 puntos cada 10 minutos (8 avenidas alrededor de CU, en ambos sentidos) y guardó cada lectura en `data/lecturas/`.

La recolección corrió del 5 de noviembre a las 14:58 al 12 de noviembre a las 19:10. Se detuvo ahí porque se agotó la cuota gratuita de TomTom (20,000 consultas al mes). En total son unas 16,300 lecturas.

Para cada lectura calculamos r = velocidad actual / velocidad sin tráfico. Si r es menor que 0.5, el punto cuenta como congestionado.

## Qué hay en el repo

| Archivo o carpeta | Para qué sirve |
|---|---|
| `data/lecturas/` | Las lecturas crudas, un CSV por día. No se editan a mano. |
| `puntos.csv` | Los 16 puntos: avenida, tramo, sentido y coordenadas. |
| `recolectar.py`, `.github/workflows/` | El recolector. Ya está desactivado; se queda como referencia. |
| `validar_puntos.py` | Revisa que cada coordenada caiga en la avenida y el sentido correctos. |
| `trafico/` | El análisis completo en PySpark (limpieza, variables, modelos y experimentos). |
| `tests/` | Pruebas pequeñas con ejemplos calculados a mano. |
| `salidas/` | Lo que genera el código. No se sube a git. |

## Cómo correrlo

Necesitas Java 25 (25.0.3 o posterior), python 3.15 y [uv](https://docs.astral.sh/uv/). En Windows conviene usar WSL con Ubuntu; Spark en Windows directo da muchos problemas.

```bash
git clone git@github.com:AydroPunk/Trafico-CDMX.git
cd Trafico-CDMX
uv sync      # instala Python 3.15, PySpark 4.3 y lo demás, con las versiones exactas de uv.lock
make test    # 5 pruebas, unos 20 segundos
make run     # todo el proyecto, unos 12 minutos
```

Para correr un solo paso: `make run PASO=exp2`. Los pasos son `silver`, `gold`, `exp1`, `exp2`, `exp3`, `figuras` y `manifiesto`, en ese orden.

Si no tienes `make`, puedes usar `uv run pytest` y `uv run python -m trafico`.

## Qué hace el código

1. **silver:** lee los CSV, quita lecturas con error, con cierre vial o repetidas, y calcula r y el estado (fluido, lento o congestionado).
2. **gold:** agrega a cada lectura sus dos lecturas anteriores y la siguiente, y la asigna a una partición: entrenamiento (5 al 9 de noviembre), validación (10) o prueba (11 y 12).
3. **exp1:** con solo la hora, el tipo de día y el punto, compara un perceptrón sin capa oculta contra uno con capa oculta.
4. **exp2:** compara el perceptrón contra dos cadenas de Markov y dos referencias sencillas: "todo sigue igual" y el promedio histórico de esa hora.
5. **exp3:** predice con Markov a 10, 20 y 30 minutos.
6. **figuras** y **manifiesto:** gráficas en PDF y un registro de qué datos y versiones produjeron los resultados.

El umbral de decisión y el tamaño de la capa oculta se eligen con validación. Prueba se usa una sola vez, al final.

## Qué encontramos

- Con solo la hora, la capa oculta mejora mucho al perceptrón: una frontera lineal no puede seguir los picos de la mañana y de la tarde.
- A 10 minutos, ni el perceptrón ni Markov le ganan claramente a suponer que todo sigue igual. El tráfico cambia poco en tan poco tiempo. El perceptrón sí da las probabilidades más confiables.
- Entre más lejos se predice, más rápido empeora suponer que todo sigue igual. A 30 minutos Markov ya queda arriba, aunque por poco margen.

Los números completos están en `salidas/tablas/` y en el reporte.

## Reproducibilidad

- `uv.lock` fija la versión exacta de cada paquete de Python.
- Las semillas, los núcleos de Spark y las particiones están fijos en `trafico/config.py`.
- `salidas/manifiesto.json` guarda el commit, el sha256 de cada CSV usado y el de cada tabla de resultados.
- Los resultados del reporte salen del tag `corte-final`.

En la misma computadora los resultados salen idénticos. En otra, el perceptrón puede cambiar en el tercer decimal por diferencias de redondeo en las bibliotecas numéricas; las conclusiones no cambian.

## Licencia

GPL-3.0. Ver `LICENSE`.