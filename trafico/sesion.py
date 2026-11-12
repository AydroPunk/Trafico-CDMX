"""Crea la sesión de Spark con la configuración del proyecto."""

import warnings

from pyspark.sql import SparkSession

from trafico import config


def crear_sesion():
    # PySpark 4.3 avisa que pandas 3 no está soportado del todo. Solo usamos pandas
    # al final, para tablas y gráficas, y ahí funciona bien.
    warnings.filterwarnings("ignore", message="PySpark does not yet fully support pandas")
    warnings.filterwarnings("ignore", message="The copy keyword is deprecated")

    spark = (
        SparkSession.builder.master(config.SPARK_MAESTRO)
        .appName("trafico-cdmx")
        .config("spark.driver.memory", config.SPARK_MEMORIA)
        .config("spark.sql.session.timeZone", config.ZONA)
        .config("spark.sql.shuffle.partitions", config.SPARK_PARTICIONES)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
