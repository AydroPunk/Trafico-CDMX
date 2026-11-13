"""Figuras del reporte en PDF. Solo leen lo que ya está en salidas/, no entrenan nada."""

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo se guardan archivos

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trafico import config
from trafico.markov import matriz_suavizada

# Colores categóricos
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NEUTRO = "#52514e"
TINTA = "#0b0b0b"
TENUE = "#898781"
REJILLA = "#e1e0d9"
SECUENCIAL = "Blues"

ANCHO = 6.3  # ancho de texto de una hoja carta con márgenes de ~1.8 cm, en pulgadas

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": NEUTRO,
    "axes.titlesize": 10,
    "axes.titlecolor": TINTA,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": REJILLA,
    "grid.linewidth": 0.6,
    "xtick.color": TENUE,
    "ytick.color": TENUE,
    "xtick.labelcolor": NEUTRO,
    "ytick.labelcolor": NEUTRO,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "pdf.fonttype": 42,  # fuentes incrustadas, el texto se puede seleccionar en el PDF
})


def guardar(fig, nombre):
    config.FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURAS / f"{nombre}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  figura guardada: {nombre}")


def nombre_corto(avenida):
    for quitar in ["Avenida de los ", "Avenida del ", "Avenida ", "Calzada de ", "Anillo "]:
        avenida = avenida.replace(quitar, "")
    return avenida


# ---- datos

def perfil_horario(silver):
    df = silver.assign(hora_entera=np.floor(silver["hora"]).astype(int), c=silver["estado"] == "C")
    fig, ax = plt.subplots(figsize=(ANCHO, 2.6))
    for color, tipo in zip(COLORES, ["entre semana", "fin de semana", "festivo"]):
        parte = df[df["tipo_dia"] == tipo]
        if parte.empty:
            continue
        perfil = parte.groupby("hora_entera")["c"].mean() * 100
        ax.plot(perfil.index, perfil.values, color=color, label=tipo.capitalize())
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Lecturas congestionadas (%)")
    ax.set_title("Congestión por hora del día", loc="left")
    ax.legend(loc="upper left")
    guardar(fig, "perfil_horario")


def congestion_por_punto(silver):
    df = silver.assign(c=silver["estado"] == "C")
    tabla = df.groupby(["id", "avenida", "sentido"])["c"].mean().mul(100).reset_index().sort_values("c")
    etiquetas = [f"{i} {nombre_corto(a)} ({s})" for i, a, s in zip(tabla["id"], tabla["avenida"], tabla["sentido"])]
    fig, ax = plt.subplots(figsize=(ANCHO, 3.6))
    ax.barh(etiquetas, tabla["c"], color=COLORES[0], height=0.7)
    for y, v in enumerate(tabla["c"]):
        ax.text(v + 0.5, y, f"{v:.1f}", va="center", fontsize=7.5, color=NEUTRO)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Lecturas congestionadas (%)")
    ax.set_title("Congestión por punto de medición", loc="left")
    guardar(fig, "congestion_por_punto")


def matrices_markov(gold):
    """P̂ con todos los puntos juntos y con el punto más congestionado, solo con datos de ajuste."""
    ajuste = gold[gold["particion"] != "prueba"]

    def matriz(df):
        n = pd.crosstab(df["estado"], df["estado_sig"]).reindex(
            index=config.ESTADOS, columns=config.ESTADOS, fill_value=0)
        return matriz_suavizada(n.to_numpy())

    peor = ajuste.groupby("id")["y"].mean().idxmax()
    paneles = [("Todos los puntos", matriz(ajuste)), (f"Punto {peor}", matriz(ajuste[ajuste["id"] == peor]))]

    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 2.7))
    for ax, (titulo, P) in zip(ejes, paneles):
        ax.imshow(P, cmap=SECUENCIAL, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if P[i, j] > 0.6 else TINTA)
        ax.set_xticks(range(3), config.ESTADOS)
        ax.set_yticks(range(3), config.ESTADOS)
        ax.set_xlabel("Estado a los 10 min")
        ax.set_ylabel("Estado actual")
        ax.set_title(titulo, loc="left")
        ax.grid(False)
        ax.spines[:].set_visible(False)
    guardar(fig, "matrices_markov")


# --- resultados

def curvas_tau():
    curvas = pd.read_csv(config.PREDICCIONES / "curvas_tau_exp2.csv")
    curvas = curvas[curvas["modelo"] != "Persistencia"]  # no tiene umbral
    fig, ax = plt.subplots(figsize=(ANCHO, 2.8))
    for color, (modelo, parte) in zip(COLORES, curvas.groupby("modelo", sort=False)):
        ax.plot(parte["tau"], parte["f1"], color=color, label=modelo)
        mejor = parte.loc[parte["f1"].idxmax()]
        ax.plot(mejor["tau"], mejor["f1"], "o", ms=5, color=color)
    ax.set_xlabel("Umbral τ")
    ax.set_ylabel("F1 en validación")
    ax.set_title("Elección del umbral en validación (el punto marca el τ elegido)", loc="left")
    ax.legend(ncols=2, fontsize=8)
    guardar(fig, "curvas_tau")


def comparacion_modelos():
    t = pd.read_csv(config.TABLAS / "exp2_global.csv").sort_values("f1")
    fig, ax = plt.subplots(figsize=(ANCHO, 2.6))
    y = np.arange(len(t))
    ax.hlines(y, t["f1_ic_inf"], t["f1_ic_sup"], color=COLORES[0], linewidth=1.6)
    ax.plot(t["f1"], y, "o", ms=6, color=COLORES[0])
    for yi, f1, sup in zip(y, t["f1"], t["f1_ic_sup"]):
        ax.text(sup + 0.005, yi, f"{f1:.3f}", va="center", fontsize=8, color=NEUTRO)
    ax.set_yticks(y, t["modelo"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("F1 en prueba (punto) e intervalo de confianza del 95 % (línea)")
    ax.set_title("Comparación de modelos a 10 minutos", loc="left")
    guardar(fig, "comparacion_modelos")


def horizontes():
    t = pd.read_csv(config.TABLAS / "exp3_horizontes.csv")
    fig, ax = plt.subplots(figsize=(ANCHO, 2.6))
    for color, (modelo, parte) in zip(COLORES, t.groupby("modelo", sort=False)):
        ax.plot(parte["horizonte_min"], parte["f1"], "-o", ms=5, color=color, label=modelo)
    ax.set_xticks(config.PASO_MIN * np.array(config.HORIZONTES))
    ax.set_xlabel("Horizonte de predicción (min)")
    ax.set_ylabel("F1 en prueba")
    ax.set_title("Predicción a 10, 20 y 30 minutos", loc="left")
    ax.legend()
    guardar(fig, "horizontes")


def perfil_aprendido():
    """Lo que aprende cada perceptrón con solo calendario, contra lo observado (entre semana)."""
    pred = pd.read_parquet(config.PREDICCIONES / "exp1.parquet")
    pred = pred[pred["tipo_dia"] == "entre semana"].copy()
    curvas = pd.read_csv(config.PREDICCIONES / "curvas_tau_exp1.csv")
    multicapa = curvas[curvas["modelo"] != "Perceptrón simple"].groupby("modelo")["f1"].max().idxmax()
    pred["hora"] = pd.to_datetime(pred["ts"]).dt.hour

    fig, ax = plt.subplots(figsize=(ANCHO, 2.6))
    observado = pred.groupby("hora")["y"].mean()
    ax.plot(observado.index, observado.values, color=NEUTRO, linestyle="--", label="Observado")
    for color, modelo in zip(COLORES, ["Perceptrón simple", multicapa]):
        columnas = [c for c in pred.columns if c.startswith(f"{modelo} |")]
        p = pred[columnas].mean(axis=1).groupby(pred["hora"]).mean()
        ax.plot(p.index, p.values, color=color, label=modelo)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Probabilidad de congestión")
    ax.set_title("Perfil aprendido con solo calendario, entre semana (prueba)", loc="left")
    ax.legend()
    guardar(fig, "perfil_aprendido")


def generar_figuras():
    print("Figuras:")
    silver = pd.read_parquet(config.SILVER)
    gold = pd.read_parquet(config.GOLD, columns=["id", "estado", "estado_sig", "y", "particion"])
    perfil_horario(silver)
    congestion_por_punto(silver)
    matrices_markov(gold)
    if (config.TABLAS / "exp1_calendario.csv").exists():
        perfil_aprendido()
    if (config.TABLAS / "exp2_global.csv").exists():
        curvas_tau()
        comparacion_modelos()
    if (config.TABLAS / "exp3_horizontes.csv").exists():
        horizontes()
