"""Capa gold: variables de cada lectura, objetivo a 10 min y partición."""

from pyspark.sql import Window
from pyspark.sql import functions as F

from trafico import config


def columna_franja(hora):
    franja = None
    for desde, hasta, etiqueta in config.FRANJAS:
        condicion = (hora >= desde) & (hora < hasta)
        franja = F.when(condicion, etiqueta) if franja is None else franja.when(condicion, etiqueta)
    return franja


def agregar_pares(silver):
    """Agrega las lecturas anteriores y siguientes de cada punto.

    Una lectura solo se conecta con la de al lado si están a 15 min o menos.
    Si no, el valor queda nulo y esa fila no se usa.
    """
    w = Window.partitionBy("id").orderBy("ts")
    hueco_max = config.HUECO_MAX_MIN * 60

    # contiguo = 1 si la lectura está a 15 min o menos de la anterior del mismo punto
    df = silver.withColumn(
        "contiguo",
        (F.unix_timestamp("ts") - F.unix_timestamp(F.lag("ts").over(w)) <= hueco_max).cast("int"),
    )

    # Hacia atrás: r_{t-1} pide contigua la lectura t; r_{t-2} pide también la t-1
    atras_1 = F.col("contiguo") == 1
    atras_2 = atras_1 & (F.lag("contiguo", 1).over(w) == 1)
    df = df.withColumn("r_1", F.when(atras_1, F.lag("r", 1).over(w)))
    df = df.withColumn("r_2", F.when(atras_2, F.lag("r", 2).over(w)))

    # Hacia adelante: la lectura t+n sirve si todas las intermedias son contiguas
    adelante = F.lit(True)
    for n in config.HORIZONTES:
        adelante = adelante & (F.lead("contiguo", n).over(w) == 1)
        df = df.withColumn(f"estado_{n}", F.when(adelante, F.lead("estado", n).over(w)))
        if n == 1:
            df = df.withColumn("r_sig", F.when(adelante, F.lead("r", 1).over(w)))
            df = df.withColumn("ts_sig", F.when(adelante, F.lead("ts", 1).over(w)))

    return df.drop("contiguo")


def construir_gold(spark):
    print("Gold: construyendo variables...")
    silver = spark.read.parquet(str(config.SILVER))
    df = agregar_pares(silver)
    n_silver = df.count()

    # Mismas filas para todos los modelos: se necesitan r_{t-1}, r_{t-2} y el objetivo
    df = df.filter(F.col("r_1").isNotNull() & F.col("r_2").isNotNull() & F.col("r_sig").isNotNull())

    # La partición se asigna por la fecha del objetivo, así ninguna fila de
    # entrenamiento usa como respuesta una lectura de validación o de prueba
    fecha_obj = F.to_date("ts_sig")
    particion = (
        F.when(fecha_obj < F.lit(config.INICIO_VALIDACION), "entrenamiento")
        .when(fecha_obj < F.lit(config.INICIO_PRUEBA), "validacion")
        .otherwise("prueba")
    )

    gold = df.select(
        "id",
        "ts",
        "fecha",
        "hora",
        columna_franja(F.col("hora")).alias("franja"),
        "tipo_dia",
        "fin_semana",
        "r",
        "r_1",
        "r_2",
        "estado",
        F.col("estado_1").alias("estado_sig"),
        "estado_2",
        "estado_3",
        "r_sig",
        (F.col("r_sig") < config.UMBRAL_CONGESTION).cast("int").alias("y"),
        particion.alias("particion"),
    )
    gold.orderBy("id", "ts").write.mode("overwrite").parquet(str(config.GOLD))

    gold = spark.read.parquet(str(config.GOLD))
    n_gold = gold.count()
    print(f"  lecturas en silver: {n_silver:,}")
    print(f"  sin lecturas contiguas suficientes (inicio, huecos o final): {n_silver - n_gold:,}")
    print(f"  filas en gold: {n_gold:,}")
    resumen = (
        gold.groupBy("particion")
        .agg(F.count("*").alias("filas"), F.round(F.avg("y") * 100, 1).alias("congestion_%"))
        .orderBy(F.min("ts"))
    )
    for fila in resumen.collect():
        print(f"    {fila.particion:<14} {fila.filas:>7,} filas   {fila['congestion_%']:>5} % congestionadas")
    return gold
