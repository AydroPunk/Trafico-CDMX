"""Métricas de la clase congestionada, elección de τ e intervalos por bootstrap.

Trabajan con arreglos de numpy: y (0/1) y p (probabilidad de congestión).
Las predicciones de prueba son pocas miles de filas, así que basta con traerlas a pandas.
"""

import numpy as np
import pandas as pd

from trafico import config


def _conteos(y, pred, w=None):
    w = np.ones_like(y, dtype=float) if w is None else w
    vp = np.sum(w * (pred & (y == 1)))
    fp = np.sum(w * (pred & (y == 0)))
    fn = np.sum(w * (~pred & (y == 1)))
    return vp, fp, fn


def f1(y, pred, w=None):
    vp, fp, fn = _conteos(y, pred, w)
    return 2 * vp / (2 * vp + fp + fn) if vp > 0 else 0.0


def calcular(y, p, tau):
    """Todas las métricas de un modelo con su umbral τ."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    pred = p >= tau
    vp, fp, fn = _conteos(y, pred)
    return {
        "f1": f1(y, pred),
        "precision": vp / (vp + fp) if vp + fp > 0 else 0.0,
        "recall": vp / (vp + fn) if vp + fn > 0 else 0.0,
        "exactitud": np.mean(pred == (y == 1)),
        "brier": np.mean((p - y) ** 2),
    }


def elegir_tau(y, P):
    """τ con el mayor F1. Si P trae varias semillas (una por fila), se usa el F1 promedio."""
    y = np.asarray(y)
    P = np.atleast_2d(P)
    promedios = [np.mean([f1(y, p >= tau) for p in P]) for tau in config.TAUS]
    return config.TAUS[int(np.argmax(promedios))]


def bootstrap_f1(bloques, y, modelos):
    """F1 de cada modelo en N remuestreos de la prueba.

    Se remuestrean bloques de una hora (los 16 puntos juntos) y no lecturas sueltas,
    porque lecturas seguidas del mismo punto se parecen mucho y no son independientes.
    Todos los modelos usan los mismos remuestreos, así sus diferencias son comparables.

    modelos: {nombre: (P, tau)} con P de forma (semillas, filas) o (filas,).
    Devuelve un DataFrame con una fila por remuestreo y una columna por modelo.
    """
    y = np.asarray(y)
    codigos, _ = pd.factorize(pd.Series(bloques))
    n_bloques = codigos.max() + 1
    rng = np.random.default_rng(config.SEMILLA_BOOTSTRAP)
    # Cuántas veces sale cada bloque en cada remuestreo; ese número es el peso de sus filas
    veces = rng.multinomial(n_bloques, np.full(n_bloques, 1 / n_bloques), size=config.N_BOOTSTRAP)
    W = veces[:, codigos].astype(float)

    resultado = {}
    for nombre, (P, tau) in modelos.items():
        pred = np.atleast_2d(P) >= tau
        vp = W @ (pred & (y == 1)).T
        fp = W @ (pred & (y == 0)).T
        fn = W @ (~pred & (y == 1)).T
        f1s = np.where(vp > 0, 2 * vp / np.maximum(2 * vp + fp + fn, 1e-12), 0.0)
        resultado[nombre] = f1s.mean(axis=1)  # promedio de las semillas en cada remuestreo
    return pd.DataFrame(resultado)


def intervalo(valores, nivel=0.95):
    a = (1 - nivel) / 2
    return float(np.quantile(valores, a)), float(np.quantile(valores, 1 - a))
