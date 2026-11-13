"""Punto de entrada: uv run python -m trafico <paso>

Pasos: silver, gold, exp1, exp2, exp3, figuras, manifiesto o todo (por defecto).
Cada paso lee lo que dejó el anterior en salidas/.
"""

import argparse
import sys
import time

from trafico import config

PASOS = ["silver", "gold", "exp1", "exp2", "exp3", "figuras", "manifiesto"]


def requiere(ruta, paso_previo):
    # Spark deja un archivo _SUCCESS cuando termina de escribir una carpeta Parquet
    if not (ruta / "_SUCCESS").exists():
        sys.exit(f"Falta {ruta.relative_to(config.RAIZ)}. Corre antes: uv run python -m trafico {paso_previo}")


def main():
    parser = argparse.ArgumentParser(prog="python -m trafico", description="Predicción de congestión cerca de CU")
    parser.add_argument("paso", nargs="?", default="todo", choices=[*PASOS, "todo"])
    paso = parser.parse_args().paso
    pasos = PASOS if paso == "todo" else [paso]

    inicio = time.time()
    print(f"Corte: {config.CORTE:%Y-%m-%d %H:%M}" + ("" if config.CORTE == config.CORTE_FINAL else " (provisional)"))

    spark = datos = None
    if any(p != "figuras" for p in pasos):
        from trafico.sesion import crear_sesion  # Spark tarda en arrancar, solo se abre si hace falta
        spark = crear_sesion()

    for p in pasos:
        t = time.time()
        if p == "silver":
            from trafico.silver import construir_silver
            construir_silver(spark)
        elif p == "gold":
            requiere(config.SILVER, "silver")
            from trafico.variables import construir_gold
            construir_gold(spark)
        elif p.startswith("exp"):
            requiere(config.GOLD, "gold")
            from trafico import experimentos
            if datos is None:  # las particiones se cargan una vez y se comparten entre experimentos
                datos = experimentos.Datos(spark.read.parquet(str(config.GOLD)))
            getattr(experimentos, f"experimento_{p[-1]}")(datos)
        elif p == "figuras":
            requiere(config.GOLD, "gold")
            from trafico.figuras import generar_figuras
            generar_figuras()
        elif p == "manifiesto":
            from trafico.manifiesto import escribir_manifiesto
            escribir_manifiesto(spark)
        print(f"  ({time.time() - t:.0f} s)\n")

    if spark is not None:
        spark.stop()
    print(f"Listo en {(time.time() - inicio) / 60:.1f} min. Resultados en {config.SALIDAS.relative_to(config.RAIZ)}/")


if __name__ == "__main__":
    main()
