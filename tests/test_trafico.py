"""Pruebas con ejemplos pequeños cuya respuesta se conoce a mano.

Revisan las piezas que, si fallan, no truenan el programa pero dan números equivocados.
"""

from datetime import datetime

import numpy as np
import pytest
from pyspark.sql import functions as F

from trafico import metricas
from trafico.markov import estacionaria, matriz_suavizada
from trafico.sesion import crear_sesion
from trafico.silver import ESQUEMA_BRONZE, columna_estado, limpiar
from trafico.variables import agregar_pares


@pytest.fixture(scope="module")
def spark():
    sesion = crear_sesion()
    yield sesion
    sesion.stop()


def test_estados_respetan_los_umbrales(spark):
    df = spark.createDataFrame([(r,) for r in [1.0, 0.75, 0.7499, 0.5, 0.4999, 0.1]], ["r"])
    estados = [f.e for f in df.select(columna_estado(F.col("r")).alias("e")).collect()]
    assert estados == ["F", "F", "L", "L", "C", "C"]


def test_limpiar_descarta_errores_cierres_y_repetidas(spark):
    def fila(id_, ts, actual=40.0, cierre="False", error=None):
        return (1, ts, ts, id_, "Av", "Tramo", "N", actual, 50.0, 100.0, 80.0, 1.0,
                cierre, "FRC1", 10, 500.0, "abc", 1, error)

    bronze = spark.createDataFrame([
        fila("C01", "2026-11-09T14:00:05+00:00"),  # válida, r = 0.8
        fila("C01", "2026-11-09T14:04:00+00:00"),  # mismo bloque de 10 min: repetida
        fila("C01", "2026-11-09T14:10:05+00:00", error="HTTP 500"),  # error de la API
        fila("C01", "2026-11-09T14:20:05+00:00", cierre="True"),  # cierre vial
        fila("C02", "2026-11-09T14:00:07+00:00", actual=20.0),  # válida, r = 0.4
    ], ESQUEMA_BRONZE)
    silver = limpiar(bronze).orderBy("id").collect()
    assert [(f.id, f.estado) for f in silver] == [("C01", "F"), ("C02", "C")]
    assert silver[0].r == pytest.approx(0.8)
    assert silver[0].hora == pytest.approx(8.0)  # 14:00 UTC son las 8:00 en la CDMX


def test_pares_no_cruzan_puntos_ni_huecos(spark):
    t = lambda h, m: datetime(2026, 11, 9, h, m)
    df = spark.createDataFrame([
        ("A", t(8, 0), 0.9, "F"),
        ("A", t(8, 10), 0.8, "F"),
        ("A", t(8, 20), 0.4, "C"),
        ("A", t(9, 0), 0.3, "C"),  # 40 min después: hueco
        ("B", t(8, 30), 0.6, "L"),  # otro punto: no debe ver las lecturas de A
        ("B", t(8, 40), 0.2, "C"),
        ("B", t(8, 50), 0.7, "L"),
    ], ["id", "ts", "r", "estado"])
    filas = {(f.id, f.ts.hour, f.ts.minute): f for f in agregar_pares(df).collect()}

    a = filas[("A", 8, 20)]
    assert (a.r_1, a.r_2, a.r_sig) == (0.8, 0.9, None)  # la siguiente está tras el hueco
    assert filas[("A", 9, 0)].r_1 is None
    assert filas[("B", 8, 30)].r_1 is None  # no toma la última de A
    b = filas[("B", 8, 40)]
    assert (b.r_1, b.r_2, b.r_sig, b.estado_1) == (0.6, None, 0.7, "L")


def test_matriz_de_markov_suavizada():
    conteos = [[8, 2, 0],
               [0, 0, 0],  # estado nunca visto
               [1, 1, 2]]
    P = matriz_suavizada(conteos)
    assert np.allclose(P.sum(axis=1), 1)
    assert P[0, 0] == pytest.approx(9 / 13)  # (8 + 1) / (10 + 3)
    assert np.allclose(P[1], 1 / 3)
    pi = estacionaria(P)
    assert np.allclose(pi @ P, pi) and pi.sum() == pytest.approx(1)


def test_metricas_contra_un_ejemplo_a_mano():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0])
    p = np.array([.9, .6, .2, .7, .1, .3, .2, .1, .8, .4])
    # Con τ = 0.5: VP = 3, FP = 1, FN = 1
    m = metricas.calcular(y, p, 0.5)
    assert m["f1"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["exactitud"] == pytest.approx(0.8)
    assert m["brier"] == pytest.approx(0.165)
