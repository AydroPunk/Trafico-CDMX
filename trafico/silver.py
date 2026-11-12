"""Capa silver: lee los CSV de bronze, los limpia y los guarda en Parquet."""

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from trafico import config

# Mismas columnas, en el mismo orden, que escribe recolectar.py
ESQUEMA_BRONZE = StructType([
    StructField("corrida_id", LongType()),
    StructField("ts_utc", StringType()),
    StructField("ts_local", StringType()),
    StructField("id", StringType()),
    StructField("avenida", StringType()),
    StructField("tramo", StringType()),
    StructField("sentido", StringType()),
    StructField("current_speed_kmh", DoubleType()),
    StructField("free_flow_speed_kmh", DoubleType()),
    StructField("current_travel_time_s", DoubleType()),
    StructField("free_flow_travel_time_s", DoubleType()),
    StructField("confidence", DoubleType()),
    StructField("road_closure", StringType()),
    StructField("frc", StringType()),
    StructField("n_vertices", IntegerType()),
    StructField("longitud_tramo_m", DoubleType()),
    StructField("tramo_hash", StringType()),
    StructField("intentos", IntegerType()),
    StructField("error", StringType()),
])


def leer_bronze(spark):
    return (
        spark.read.option("header", True)
        .option("recursiveFileLookup", True)
        .option("pathGlobFilter", "lecturas.csv")
        .schema(ESQUEMA_BRONZE)
        .csv(str(config.BRONZE))
    )


def columna_estado(r):
    """F (fluido), L (lento) o C (congestionado) según la razón de velocidades."""
    return (
        F.when(r >= config.UMBRAL_FLUIDO, "F")
        .when(r >= config.UMBRAL_CONGESTION, "L")
        .otherwise("C")
    )


def columna_tipo_dia(fecha):
    festivos = [F.lit(d) for d in config.FESTIVOS]
    return (
        F.when(fecha.isin(festivos), "festivo")
        .when(F.dayofweek(fecha).isin(1, 7), "fin de semana")  # 1 = domingo, 7 = sábado
        .otherwise("entre semana")
    )


def limpiar(bronze):
    """Convierte las lecturas crudas en lecturas válidas, una por punto cada 10 min."""
    df = (
        bronze
        .withColumn("ts", F.to_timestamp("ts_utc"))
        .filter(F.col("error").isNull())
        .filter(F.col("current_speed_kmh").isNotNull() & (F.col("free_flow_speed_kmh") > 0))
        # Con cierre vial TomTom reporta velocidades históricas, no tráfico real
        .filter(F.lower("road_closure") != "true")
        .filter(F.col("ts") < F.lit(config.CORTE))
    )

    # Bloque de reloj de 10 min. Si por algún motivo hay dos lecturas del mismo
    # punto en el mismo bloque, se queda la primera.
    df = df.withColumn("bloque", (F.unix_timestamp("ts") / (config.PASO_MIN * 60)).cast("long"))
    primera = Window.partitionBy("id", "bloque").orderBy("ts")
    df = df.withColumn("n", F.row_number().over(primera)).filter("n = 1").drop("n")

    r = F.least(F.col("current_speed_kmh") / F.col("free_flow_speed_kmh"), F.lit(1.0))
    fecha = F.to_date("ts")  # usa la zona de la sesión: America/Mexico_City

    return df.select(
        "id",
        "avenida",
        "sentido",
        "ts",
        "bloque",
        fecha.alias("fecha"),
        (F.hour("ts") + F.minute("ts") / 60).alias("hora"),
        columna_tipo_dia(fecha).alias("tipo_dia"),
        F.dayofweek(fecha).isin(1, 7).cast("int").alias("fin_semana"),
        F.col("current_speed_kmh").alias("v_actual"),
        F.col("free_flow_speed_kmh").alias("v_libre"),
        r.alias("r"),
        columna_estado(r).alias("estado"),
        F.col("confidence").alias("confianza"),
        "frc",
    )


def construir_silver(spark):
    print("Silver: leyendo bronze...")
    bronze = leer_bronze(spark).cache()
    silver = limpiar(bronze)
    silver.orderBy("id", "ts").write.mode("overwrite").parquet(str(config.SILVER))

    silver = spark.read.parquet(str(config.SILVER))
    n_bronze, n_silver = bronze.count(), silver.count()
    rango = silver.agg(F.min("ts"), F.max("ts")).first()
    print(f"  lecturas en bronze: {n_bronze:,}")
    print(f"  descartadas (error, cierre, después del corte o repetidas): {n_bronze - n_silver:,}")
    print(f"  lecturas en silver: {n_silver:,}, de {rango[0]} a {rango[1]}")
    bronze.unpersist()
    return silver
