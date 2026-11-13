"""Cadena de Markov con estados F, L y C, una matriz por punto (o por punto y franja)."""

import numpy as np
from pyspark.sql import functions as F

from trafico import config

C = config.ESTADOS.index("C")


def matriz_suavizada(conteos, alfa=config.ALFA):
    """P_ij = (n_ij + alfa) / (n_i + 3 alfa). Cada fila suma 1 aunque un estado nunca se haya visto."""
    conteos = np.asarray(conteos, dtype=float)
    k = conteos.shape[0]
    return (conteos + alfa) / (conteos.sum(axis=1, keepdims=True) + k * alfa)


def estacionaria(P):
    """Distribución π con π P = π y suma 1."""
    k = P.shape[0]
    A = np.vstack([P.T - np.eye(k), np.ones(k)])
    b = np.append(np.zeros(k), 1.0)
    return np.linalg.lstsq(A, b, rcond=None)[0]


class Markov:
    """Homogénea: una matriz por punto. No homogénea: una por punto y franja horaria."""

    def __init__(self, por_franja=False):
        self.por_franja = por_franja
        self.claves = ["id", "franja"] if por_franja else ["id"]
        self.nombre = "Markov por franja" if por_franja else "Markov"

    def ajustar(self, df):
        filas = df.groupBy(*self.claves, "estado", "estado_sig").count().collect()
        conteos = {}
        for f in filas:
            clave = tuple(f[c] for c in self.claves)
            n = conteos.setdefault(clave, np.zeros((3, 3)))
            n[config.ESTADOS.index(f["estado"]), config.ESTADOS.index(f["estado_sig"])] += f["count"]
        self.conteos = conteos
        self.matrices = {clave: matriz_suavizada(n) for clave, n in conteos.items()}
        self._spark = df.sparkSession
        return self

    def predecir(self, df, pasos=1):
        """p = probabilidad de estar en C dentro de `pasos` lecturas (Chapman–Kolmogorov: P^n)."""
        tabla = []
        for clave, P in self.matrices.items():
            Pn = np.linalg.matrix_power(P, pasos)
            for i, estado in enumerate(config.ESTADOS):
                tabla.append((*clave, estado, float(Pn[i, C])))
        tabla = self._spark.createDataFrame(tabla, [*self.claves, "estado", "p"])
        # Una clave que no estuvo en entrenamiento recibe la probabilidad uniforme 1/3
        return df.join(tabla, [*self.claves, "estado"], "left").fillna({"p": 1 / 3})
