"""Modelos de referencia. Si el perceptrón o Markov no les ganan, no aportan nada."""

from pyspark.sql import functions as F

from trafico import config


class Persistencia:
    """Supone que dentro de 10 min todo sigue igual que ahora."""

    nombre = "Persistencia"

    def ajustar(self, df):
        return self  # no aprende nada

    def predecir(self, df):
        return df.withColumn("p", (F.col("r") < config.UMBRAL_CONGESTION).cast("double"))


class PromedioHistorico:
    """Fracción de veces que hubo congestión en ese punto, a esa hora y en ese tipo de día."""

    nombre = "Promedio histórico"

    def ajustar(self, df):
        df = df.withColumn("hora_entera", F.floor("hora"))
        self.por_hora = df.groupBy("id", "hora_entera", "fin_semana").agg(F.avg("y").alias("p_hora"))
        # Respaldo por si en prueba aparece una combinación que no estuvo en entrenamiento
        self.por_punto = df.groupBy("id", "fin_semana").agg(F.avg("y").alias("p_punto"))
        self.global_ = df.agg(F.avg("y")).first()[0]
        return self

    def predecir(self, df):
        return (
            df.withColumn("hora_entera", F.floor("hora"))
            .join(self.por_hora, ["id", "hora_entera", "fin_semana"], "left")
            .join(self.por_punto, ["id", "fin_semana"], "left")
            .withColumn("p", F.coalesce("p_hora", "p_punto", F.lit(self.global_)))
            .drop("hora_entera", "p_hora", "p_punto")
        )
