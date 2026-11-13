"""Perceptrón multicapa de Spark ML."""

from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from trafico import config

# Variables numéricas de cada experimento. El punto siempre entra en one-hot.
CALENDARIO = ["hora", "fin_semana"]
COMPLETO = ["hora", "fin_semana", "r", "r_1", "r_2"]


class Perceptron:
    """ocultas=0 da capas [n, 2]: una frontera lineal, como el perceptrón simple de clase.
    ocultas=h da capas [n, h, 2] con sigmoide en la capa oculta."""

    def __init__(self, variables, ocultas, semilla):
        self.variables = variables
        self.ocultas = ocultas
        self.semilla = semilla
        tipo = "simple" if ocultas == 0 else f"h={ocultas}"
        self.nombre = f"Perceptrón {tipo}"

    def _ordenar(self, df):
        # Mismo reparto y orden de filas en cualquier equipo, para que L-BFGS dé lo mismo
        return df.repartition(config.SPARK_PARTICIONES, "id").sortWithinPartitions("id", "ts")

    def ajustar(self, df):
        n_puntos = df.select("id").distinct().count()
        n_entradas = len(self.variables) + n_puntos
        capas = [n_entradas, 2] if self.ocultas == 0 else [n_entradas, self.ocultas, 2]

        etapas = [
            StringIndexer(inputCol="id", outputCol="id_idx"),
            OneHotEncoder(inputCol="id_idx", outputCol="id_vec", dropLast=False),
            VectorAssembler(inputCols=self.variables, outputCol="num"),
            # El escalador aprende media y desviación solo con los datos de ajuste
            StandardScaler(inputCol="num", outputCol="num_esc", withMean=True, withStd=True),
            VectorAssembler(inputCols=["num_esc", "id_vec"], outputCol="x"),
            MultilayerPerceptronClassifier(
                featuresCol="x",
                labelCol="y",
                layers=capas,
                maxIter=config.MAX_ITER,
                seed=self.semilla,
            ),
        ]
        datos = self._ordenar(df.withColumn("y", F.col("y").cast("double"))).cache()
        self.modelo = Pipeline(stages=etapas).fit(datos)
        datos.unpersist()
        return self

    def predecir(self, df):
        salida = self.modelo.transform(df)
        columnas = [c for c in df.columns] + [vector_to_array("probability")[1].alias("p")]
        return salida.select(*columnas)
