"""Parámetros del proyecto. Todo lo que se puede ajustar está aquí."""

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Rutas del proyecto
RAIZ = Path(__file__).resolve().parent.parent
BRONZE = RAIZ / "data" / "lecturas"  # CSV que escribe recolectar.py
PUNTOS = RAIZ / "puntos.csv"
SALIDAS = RAIZ / "salidas"
SILVER = SALIDAS / "silver"
GOLD = SALIDAS / "gold"
PREDICCIONES = SALIDAS / "predicciones"
TABLAS = SALIDAS / "tablas"
FIGURAS = SALIDAS / "figuras"
MANIFIESTO = SALIDAS / "manifiesto.json"

# Tiempo
ZONA = "America/Mexico_City"
TZ = ZoneInfo(ZONA)
PASO_MIN = 10  # una lectura por punto cada 10 minutos
HUECO_MAX_MIN = 15  # dos lecturas seguidas forman par solo si están a 15 min o menos

# La recolección terminó al agotarse la cuota mensual gratuita de TomTom
# (20,000 consultas). La última corrida completa fue el 12 nov a las 19:10.
CORTE_FINAL = datetime(2026, 11, 12, 19, 20, tzinfo=TZ)
CORTE = min(CORTE_FINAL, datetime.now(TZ))

# Partición cronológica (fechas locales)
INICIO_VALIDACION = date(2026, 11, 10)  # entrenamiento: 5 al 9 (incluye sábado y domingo)
INICIO_PRUEBA = date(2026, 11, 11)  # validación: 10; prueba: 11 y 12
FESTIVOS = {date(2026, 11, 16)}  # queda fuera del corte; solo se usaría para reportar

# Estados de congestión con r = velocidad actual / velocidad libre
UMBRAL_FLUIDO = 0.75  # F si r >= 0.75
UMBRAL_CONGESTION = 0.5  # C si r < 0.5, L en medio
ESTADOS = ["F", "L", "C"]

# Franjas horarias para la cadena de Markov no homogénea: (desde, hasta, etiqueta)
FRANJAS = [
    (0, 6, "00-06"),
    (6, 12, "06-12"),
    (12, 17, "12-17"),
    (17, 21, "17-21"),
    (21, 24, "21-24"),
]
ALFA = 1  # suavizado de Laplace: (n_ij + 1) / (n_i + 3)
HORIZONTES = [1, 2, 3]  # pasos de 10 min: 10, 20 y 30 min

# Perceptrón multicapa
SEMILLAS = [1, 2, 3, 4, 5]
OCULTAS = [4, 8, 16]  # neuronas de la capa oculta que se comparan
MAX_ITER = 200

# Umbral de decisión τ, se elige en validación maximizando F1
TAUS = [round(0.05 * i, 2) for i in range(1, 20)]

# Bootstrap para los intervalos de confianza
N_BOOTSTRAP = 1000
SEMILLA_BOOTSTRAP = 2026

# Spark: núcleos y particiones fijos para que los resultados se repitan en cualquier equipo (reproducibilidad)
SPARK_MAESTRO = "local[4]"
SPARK_PARTICIONES = 8
SPARK_MEMORIA = "2g"