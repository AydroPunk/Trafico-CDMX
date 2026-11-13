"""Los tres experimentos del reporte.

Procedimiento común a todos los modelos:
  1. Ajustar con entrenamiento (5 al 10) y predecir validación (11).
  2. Elegir τ (y h en el perceptrón) con validación.
  3. Reajustar con entrenamiento + validación y predecir prueba, una sola vez.
"""

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from trafico import config, metricas
from trafico.lineas_base import Persistencia, PromedioHistorico
from trafico.markov import Markov, estacionaria
from trafico.perceptron import CALENDARIO, COMPLETO, Perceptron

COLUMNAS = ["id", "ts", "fecha", "hora", "tipo_dia", "estado", "estado_sig",
            "estado_2", "estado_3", "y"]
GRUPOS = ["entre semana", "fin de semana", "festivo", "cambio de estado"]


class Datos:
    """Particiones en Spark para ajustar y en pandas para evaluar, ordenadas por punto y hora.

    filtro limita las filas de validación y prueba (el experimento 3 pide estado a 30 min).
    """

    def __init__(self, gold, filtro=None):
        self.gold = gold
        self.ent = gold.filter("particion = 'entrenamiento'").cache()
        self.ent_val = gold.filter("particion != 'prueba'").cache()
        val = gold.filter("particion = 'validacion'")
        prueba = gold.filter("particion = 'prueba'")
        if filtro is not None:
            val, prueba = val.filter(filtro), prueba.filter(filtro)
        self.val, self.prueba = val.cache(), prueba.cache()
        self.val_pd = a_pandas(self.val.select(*COLUMNAS))
        self.prueba_pd = a_pandas(self.prueba.select(*COLUMNAS))
        # Bloques de una hora (fecha + hora) para el bootstrap
        p = self.prueba_pd
        self.bloques = p["fecha"].astype(str) + "-" + np.floor(p["hora"]).astype(int).astype(str)


def a_pandas(df):
    return df.toPandas().sort_values(["id", "ts"]).reset_index(drop=True)


def probabilidades(modelo, df, **kwargs):
    return a_pandas(modelo.predecir(df, **kwargs).select("id", "ts", "p"))["p"].to_numpy()


def validar(fabrica, semillas, datos, y_val=None, **kwargs):
    """Ajusta con entrenamiento, predice validación y elige τ. fabrica(semilla) crea el modelo."""
    y_val = datos.val_pd["y"].to_numpy() if y_val is None else y_val
    P = np.vstack([probabilidades(fabrica(s).ajustar(datos.ent), datos.val, **kwargs) for s in semillas])
    return P, metricas.elegir_tau(y_val, P)


def probar(fabrica, semillas, datos, **kwargs):
    """Reajusta con entrenamiento + validación y predice prueba."""
    return np.vstack([probabilidades(fabrica(s).ajustar(datos.ent_val), datos.prueba, **kwargs)
                      for s in semillas])


def resumir(nombre, y, P, tau, mascara=None):
    """Media y desviación de las métricas entre semillas (con una semilla la desviación es 0)."""
    if mascara is not None:
        y, P = y[mascara], P[:, mascara]
    filas = pd.DataFrame([metricas.calcular(y, p, tau) for p in P])
    resumen = {"modelo": nombre, "tau": tau, "n": len(y)}
    for m in filas.columns:
        resumen[m] = filas[m].mean()
        resumen[f"{m}_de"] = filas[m].std(ddof=0)
    return resumen


def tabla_modelos(modelos, y, boot):
    tabla = pd.DataFrame([resumir(m, y, P, tau) for m, (P, tau) in modelos.items()])
    intervalos = [metricas.intervalo(boot[m]) for m in tabla["modelo"]]
    tabla["f1_ic_inf"] = [a for a, _ in intervalos]
    tabla["f1_ic_sup"] = [b for _, b in intervalos]
    return tabla


def comparar(boot, referencia, referencia_primero=True):
    """Diferencia de F1 entre la referencia y cada modelo, con su intervalo del 95 %."""
    filas = []
    for m in boot.columns.drop(referencia):
        a, b = (referencia, m) if referencia_primero else (m, referencia)
        dif = boot[a] - boot[b]
        inf, sup = metricas.intervalo(dif)
        filas.append({"comparacion": f"{a} − {b}", "diferencia_f1": dif.mean(),
                      "ic_inf": inf, "ic_sup": sup, "prob_mayor_que_0": (dif > 0).mean()})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- tablas

def _escapar(texto):
    if "$" in texto:  # ya viene escrito en LaTeX
        return texto
    for a, b in [("%", r"\%"), ("&", r"\&"), ("_", r"\_"), ("#", r"\#"),
                 ("±", r"$\pm$"), ("−", r"$-$"), ("τ", r"$\tau$")]:
        texto = texto.replace(a, b)
    return texto


def a_latex(df, titulo, etiqueta):
    """Tabla con booktabs lista para \\input{} en el reporte."""
    lineas = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        rf"\caption{{{titulo}}}", rf"\label{{{etiqueta}}}",
        rf"\begin{{tabular}}{{l{'r' * (len(df.columns) - 1)}}}", r"\toprule",
        " & ".join(_escapar(str(c)) for c in df.columns) + r" \\", r"\midrule",
    ]
    lineas += [" & ".join(_escapar(str(v)) for v in fila) + r" \\" for fila in df.itertuples(index=False)]
    lineas += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lineas) + "\n"


def guardar(tabla, nombre, titulo, vista=None):
    """CSV con todos los números y .tex con la versión formateada para el reporte."""
    config.TABLAS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(config.TABLAS / f"{nombre}.csv", index=False)
    tex = a_latex(tabla if vista is None else vista(tabla), titulo, f"tab:{nombre}")
    (config.TABLAS / f"{nombre}.tex").write_text(tex, encoding="utf-8")
    print(f"  tabla guardada: {nombre}")


def _f3(serie):
    return serie.map("{:.3f}".format)


def _con_signo(x):
    return f"{x:+.3f}".replace("-", "−")  # signo menos tipográfico, no guion


def vista_modelos(t):
    return pd.DataFrame({
        "Modelo": t["modelo"],
        "τ": ["—" if m == "Persistencia" else f"{x:.2f}" for m, x in zip(t["modelo"], t["tau"])],
        "F1": [f"{a:.3f}" if b == 0 else f"{a:.3f} ± {b:.3f}" for a, b in zip(t["f1"], t["f1_de"])],
        "IC 95% F1": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(t["f1_ic_inf"], t["f1_ic_sup"])],
        "Precisión": _f3(t["precision"]),
        "Recall": _f3(t["recall"]),
        "Exactitud": _f3(t["exactitud"]),
        "Brier": _f3(t["brier"]),
    })


def vista_comparaciones(t):
    return pd.DataFrame({
        "Comparación": t["comparacion"],
        "Diferencia F1": t["diferencia_f1"].map(_con_signo),
        "IC 95%": [f"[{_con_signo(a)}, {_con_signo(b)}]" for a, b in zip(t["ic_inf"], t["ic_sup"])],
    })


def vista_desglose(t):
    grupos = [g for g in GRUPOS if g in set(t["grupo"])]
    modelos = list(dict.fromkeys(t["modelo"]))
    tabla = t.pivot(index="modelo", columns="grupo", values="f1").reindex(index=modelos, columns=grupos)
    return tabla.map("{:.3f}".format).reset_index().rename(columns={"modelo": "Modelo"})


def vista_horizontes(t):
    return pd.DataFrame({
        "Horizonte": t["horizonte_min"].map("{} min".format),
        "Modelo": t["modelo"],
        "F1": _f3(t["f1"]),
        "IC 95% F1": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(t["f1_ic_inf"], t["f1_ic_sup"])],
        "Brier": _f3(t["brier"]),
    })


def guardar_predicciones(nombre, datos, modelos, curvas):
    """Probabilidades de prueba y curvas de F1 contra τ en validación, para las figuras."""
    config.PREDICCIONES.mkdir(parents=True, exist_ok=True)
    df = datos.prueba_pd[["id", "ts", "tipo_dia", "estado", "estado_sig", "y"]].copy()
    for m, (P, _) in modelos.items():
        for i, p in enumerate(P, start=1):
            df[f"{m} | s{i}"] = p
    df.to_parquet(config.PREDICCIONES / f"{nombre}.parquet", index=False)

    y_val = datos.val_pd["y"].to_numpy()
    filas = [{"modelo": m, "tau": tau, "f1": np.mean([metricas.f1(y_val, p >= tau) for p in P])}
             for m, P in curvas.items() for tau in config.TAUS]
    pd.DataFrame(filas).to_csv(config.PREDICCIONES / f"curvas_tau_{nombre}.csv", index=False)


# --- experimentos

def experimento_1(datos):
    """Solo calendario: ¿una capa oculta resuelve lo que una frontera lineal no puede?"""
    print("Experimento 1: perceptrón simple contra multicapa (hora, fin de semana y punto)")
    y = datos.prueba_pd["y"].to_numpy()
    modelos, curvas = {}, {}
    for h in [0, *config.OCULTAS]:
        fabrica = lambda s, h=h: Perceptron(CALENDARIO, h, s)
        nombre = fabrica(0).nombre
        print(f"  {nombre}...")
        curvas[nombre], tau = validar(fabrica, config.SEMILLAS, datos)
        modelos[nombre] = (probar(fabrica, config.SEMILLAS, datos), tau)

    boot = metricas.bootstrap_f1(datos.bloques, y, modelos)
    guardar(tabla_modelos(modelos, y, boot), "exp1_calendario",
            "Experimento 1: solo variables de calendario (prueba)", vista_modelos)
    guardar(comparar(boot, "Perceptrón simple", referencia_primero=False), "exp1_diferencias",
            "Experimento 1: diferencia de F1 contra el perceptrón simple", vista_comparaciones)
    guardar_predicciones("exp1", datos, modelos, curvas)


def experimento_2(datos):
    """Modelo completo contra Markov y las líneas base."""
    print("Experimento 2: modelo completo contra Markov y líneas base")
    y = datos.prueba_pd["y"].to_numpy()
    candidatos = {
        "Persistencia": lambda s: Persistencia(),
        "Promedio histórico": lambda s: PromedioHistorico(),
        "Markov": lambda s: Markov(),
        "Markov por franja": lambda s: Markov(por_franja=True),
        "Perceptrón simple": lambda s: Perceptron(COMPLETO, 0, s),
    }
    modelos, curvas = {}, {}
    for nombre, fabrica in candidatos.items():
        print(f"  {nombre}...")
        semillas = config.SEMILLAS if nombre.startswith("Perceptrón") else [0]
        curvas[nombre], tau = validar(fabrica, semillas, datos)
        modelos[nombre] = (probar(fabrica, semillas, datos), tau)

    # h se elige con el F1 promedio de validación; prueba no se toca para esto
    y_val = datos.val_pd["y"].to_numpy()
    mejor = None
    for h in config.OCULTAS:
        P_val, tau = validar(lambda s, h=h: Perceptron(COMPLETO, h, s), config.SEMILLAS, datos)
        f1_val = np.mean([metricas.f1(y_val, p >= tau) for p in P_val])
        print(f"  Perceptrón h={h}: F1 en validación {f1_val:.3f}")
        if mejor is None or f1_val > mejor[0]:
            mejor = (f1_val, h, tau, P_val)
    _, h, tau, P_val = mejor
    nombre = f"Perceptrón h={h}"
    print(f"  {nombre} elegido, reentrenando...")
    curvas[nombre] = P_val
    modelos[nombre] = (probar(lambda s: Perceptron(COMPLETO, h, s), config.SEMILLAS, datos), tau)

    boot = metricas.bootstrap_f1(datos.bloques, y, modelos)
    guardar(tabla_modelos(modelos, y, boot), "exp2_global",
            "Experimento 2: comparación de modelos (prueba)", vista_modelos)
    guardar(comparar(boot, nombre), "exp2_diferencias",
            f"Experimento 2: diferencia de F1 del {nombre} contra los demás modelos", vista_comparaciones)

    # Desglose por tipo de día y en las lecturas donde el estado cambia
    p = datos.prueba_pd
    mascaras = {g: (p["tipo_dia"] == g).to_numpy() for g in GRUPOS[:3]}
    mascaras["cambio de estado"] = (p["estado"] != p["estado_sig"]).to_numpy()
    filas = [{"grupo": g, **resumir(m, y, P, tau, mascara), "congestion": y[mascara].mean()}
             for g, mascara in mascaras.items() if mascara.any()
             for m, (P, tau) in modelos.items()]
    guardar(pd.DataFrame(filas), "exp2_desglose",
            "Experimento 2: F1 por tipo de día y en cambios de estado (prueba)", vista_desglose)
    guardar_predicciones("exp2", datos, modelos, curvas)


def experimento_3(datos):
    """Horizontes de 10, 20 y 30 min con P^n (Chapman–Kolmogorov)."""
    print("Experimento 3: horizontes de 10, 20 y 30 min con Markov")
    # Mismas filas para los tres horizontes: las que tienen estado conocido a 30 min
    d3 = Datos(datos.gold, filtro=F.col("estado_3").isNotNull())
    filas = []
    for n in config.HORIZONTES:
        columna = "estado_sig" if n == 1 else f"estado_{n}"
        y_val = (d3.val_pd[columna] == "C").astype(int).to_numpy()
        y = (d3.prueba_pd[columna] == "C").astype(int).to_numpy()
        modelos = {}
        for nombre, fabrica, kwargs in [
            ("Persistencia", lambda s: Persistencia(), {}),
            ("Markov", lambda s: Markov(), {"pasos": n}),
            ("Markov por franja", lambda s: Markov(por_franja=True), {"pasos": n}),
        ]:
            _, tau = validar(fabrica, [0], d3, y_val=y_val, **kwargs)
            modelos[nombre] = (probar(fabrica, [0], d3, **kwargs), tau)
        boot = metricas.bootstrap_f1(d3.bloques, y, modelos)
        tabla = tabla_modelos(modelos, y, boot)
        filas += tabla.assign(horizonte_min=n * config.PASO_MIN).to_dict("records")
    guardar(pd.DataFrame(filas), "exp3_horizontes",
            "Experimento 3: predicción a 10, 20 y 30 min (prueba)", vista_horizontes)

    # Distribución estacionaria por punto, con todos los datos de ajuste
    markov = Markov().ajustar(datos.ent_val)
    pis = [{"id": punto, **dict(zip(["pi_F", "pi_L", "pi_C"], estacionaria(P)))}
           for (punto,), P in sorted(markov.matrices.items())]
    guardar(pd.DataFrame(pis), "exp3_estacionaria", r"Distribución estacionaria $\pi$ por punto",
            lambda d: pd.DataFrame({"Punto": d["id"], r"$\pi_F$": _f3(d["pi_F"]),
                                    r"$\pi_L$": _f3(d["pi_L"]), r"$\pi_C$": _f3(d["pi_C"])}))
