"""Manifiesto: qué datos, qué código y qué versiones produjeron los resultados.

Con el commit y los sha256 cualquiera puede comprobar que corre lo mismo:
git checkout <commit>, uv sync, make run, y comparar los hashes de las tablas.
"""

import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime
from importlib.metadata import version

from trafico import config


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=config.RAIZ, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def archivos_bronze():
    """Cada lecturas.csv hasta el día del corte, con cuántas de sus filas entran antes del corte."""
    resultado = []
    for ruta in sorted(config.BRONZE.glob("fecha=*/lecturas.csv")):
        fecha = ruta.parent.name.removeprefix("fecha=")
        if fecha > config.CORTE.date().isoformat():
            continue
        with open(ruta, newline="", encoding="utf-8") as f:
            tiempos = [datetime.fromisoformat(fila["ts_utc"]) for fila in csv.DictReader(f)]
        resultado.append({
            "archivo": str(ruta.relative_to(config.RAIZ)),
            "sha256": sha256(ruta),
            "filas": len(tiempos),
            "filas_antes_del_corte": sum(t < config.CORTE for t in tiempos),
        })
    return resultado


def escribir_manifiesto(spark=None):
    bronze = archivos_bronze()
    tablas = sorted(config.TABLAS.glob("*.csv")) if config.TABLAS.exists() else []
    manifiesto = {
        "generado": datetime.now(config.TZ).isoformat(timespec="seconds"),
        "corte": config.CORTE.isoformat(),
        "corte_final": config.CORTE.isoformat() == config.CORTE_FINAL.isoformat(),
        "git": {
            "commit": git("rev-parse", "HEAD"),
            # Si hay cambios sin commit en el código, el commit no basta para repetir el resultado
            "cambios_sin_commit": bool(git("status", "--porcelain", "trafico", "pyproject.toml", "uv.lock")),
        },
        "particion": {
            "entrenamiento": f"antes de {config.INICIO_VALIDACION}",
            "validacion": str(config.INICIO_VALIDACION),
            "prueba": f"desde {config.INICIO_PRUEBA} hasta el corte",
        },
        "parametros": {
            "semillas": config.SEMILLAS,
            "ocultas": config.OCULTAS,
            "max_iter": config.MAX_ITER,
            "taus": config.TAUS,
            "n_bootstrap": config.N_BOOTSTRAP,
            "semilla_bootstrap": config.SEMILLA_BOOTSTRAP,
            "spark_maestro": config.SPARK_MAESTRO,
            "spark_particiones": config.SPARK_PARTICIONES,
        },
        "versiones": {
            "python": platform.python_version(),
            "java": spark.sparkContext._jvm.System.getProperty("java.version") if spark else None,
            **{p: version(p) for p in ["pyspark", "pandas", "numpy", "pyarrow", "matplotlib"]},
        },
        "datos": {
            "archivos": len(bronze),
            "filas_antes_del_corte": sum(a["filas_antes_del_corte"] for a in bronze),
            "bronze": bronze,
        },
        "tablas": {t.name: sha256(t) for t in tablas},
    }
    config.SALIDAS.mkdir(parents=True, exist_ok=True)
    config.MANIFIESTO.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifiesto: {len(bronze)} archivos, {manifiesto['datos']['filas_antes_del_corte']:,} filas, "
          f"commit {str(manifiesto['git']['commit'])[:7]}")
    return manifiesto
