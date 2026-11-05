"""Toma una lectura de tráfico de cada punto de puntos.csv y la agrega al dataset.

Lo corre GitHub Actions cada 10 minutos (.github/workflows/recolectar.yml).

Escribe en data/lecturas/fecha=AAAA-MM-DD/lecturas.csv (fecha en hora de la CDMX).

Reglas del dataset:
  - Una fila por punto por corrida, aunque la consulta falle (con columna error).
    Así los huecos quedan visibles en vez de desaparecer.
  - corrida_id agrupa las 16 lecturas de una misma ejecución.
Solo usa la biblioteca estándar de Python.
"""

import csv
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo

    TZ_CDMX = ZoneInfo("America/Mexico_City")
except Exception:  # sin base de zonas horarias: la CDMX es UTC-6 todo el año desde 2022
    TZ_CDMX = timezone(timedelta(hours=-6))

ZOOM = 10  # el mismo nivel con el que se validaron los puntos
URL = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{ZOOM}/json"
REINTENTOS = 3
CARPETA = Path("data/lecturas")
VENTANA_MINUTOS = int(os.environ.get("VENTANA_MINUTOS", "10"))

COLUMNAS = [
    "corrida_id", "ts_utc", "ts_local", "id", "avenida", "tramo", "sentido",
    "current_speed_kmh", "free_flow_speed_kmh", "current_travel_time_s",
    "free_flow_travel_time_s", "confidence", "road_closure", "frc",
    "n_vertices", "longitud_tramo_m", "tramo_hash", "intentos", "error",
]


def consultar(lat, lon, key):
    """Regresa (datos, intentos, error). Reintenta en errores de red y 429/5xx."""
    params = urllib.parse.urlencode({"point": f"{lat},{lon}", "unit": "KMPH", "key": key})
    error = ""
    for intento in range(1, REINTENTOS + 1):
        try:
            with urllib.request.urlopen(f"{URL}?{params}", timeout=20) as r:
                return json.load(r)["flowSegmentData"], intento, ""
        except urllib.error.HTTPError as e:
            error = f"HTTP {e.code}"
            if e.code not in (429, 500, 502, 503, 504):
                return None, intento, error  # 400/403: reintentar no sirve
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as e:
            error = type(e).__name__
        time.sleep(2 * intento)
    return None, REINTENTOS, error


def longitud_m(coords):
    total = 0.0
    for a, b in zip(coords, coords[1:]):
        lat0 = math.radians((a["latitude"] + b["latitude"]) / 2)
        dx = (b["longitude"] - a["longitude"]) * 111_320 * math.cos(lat0)
        dy = (b["latitude"] - a["latitude"]) * 111_320
        total += math.hypot(dx, dy)
    return round(total, 1)


def huella(coords):
    """Hash corto de la geometría: si TomTom cambia el tramo de un punto, cambia el hash."""
    texto = ";".join(f"{c['latitude']:.5f},{c['longitude']:.5f}" for c in coords)
    return hashlib.sha1(texto.encode()).hexdigest()[:10]


def ultima_lectura_utc():
    """Hora (UTC) de la lectura más reciente en los archivos de hoy y ayer, o None."""
    hoy = datetime.now(timezone.utc).astimezone(TZ_CDMX).date()
    ultima = None
    for fecha in (hoy, hoy - timedelta(days=1)):
        archivo = CARPETA / f"fecha={fecha.isoformat()}" / "lecturas.csv"
        if not archivo.exists():
            continue
        with open(archivo, newline="", encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                try:
                    ts = datetime.fromisoformat(fila["ts_utc"])
                except (KeyError, ValueError, TypeError):
                    continue
                if ultima is None or ts > ultima:
                    ultima = ts
    return ultima


def main():
    key = os.environ.get("TOMTOM_API_KEY", "").strip()
    if not key:
        sys.exit("Falta TOMTOM_API_KEY (en GitHub va como secreto del repositorio).")

    ultima = ultima_lectura_utc()
    if ultima is not None and VENTANA_MINUTOS > 0:
        bloque = VENTANA_MINUTOS * 60
        ahora = datetime.now(timezone.utc)
        if int(ultima.timestamp()) // bloque == int(ahora.timestamp()) // bloque:
            print(f"Ya hay lecturas en este bloque de {VENTANA_MINUTOS} min "
                  f"(última: {ultima.astimezone(TZ_CDMX):%H:%M:%S}); no se consulta.")
            return

    with open("puntos.csv", newline="", encoding="utf-8") as f:
        puntos = [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]

    inicio = datetime.now(timezone.utc)
    corrida_id = os.environ.get("GITHUB_RUN_ID") or inicio.strftime("local-%Y%m%dT%H%M%S")

    filas = []
    for p in puntos:
        d, intentos, error = consultar(p["lat"], p["lon"], key)
        ahora = datetime.now(timezone.utc)
        fila = {
            "corrida_id": corrida_id,
            "ts_utc": ahora.isoformat(timespec="seconds"),
            "ts_local": ahora.astimezone(TZ_CDMX).isoformat(timespec="seconds"),
            "id": p["id"], "avenida": p["avenida"], "tramo": p["tramo"], "sentido": p["sentido"],
            "intentos": intentos, "error": error,
        }
        if d:
            coords = d.get("coordinates", {}).get("coordinate", [])
            fila.update({
                "current_speed_kmh": d.get("currentSpeed"),
                "free_flow_speed_kmh": d.get("freeFlowSpeed"),
                "current_travel_time_s": d.get("currentTravelTime"),
                "free_flow_travel_time_s": d.get("freeFlowTravelTime"),
                "confidence": d.get("confidence"),
                "road_closure": d.get("roadClosure"),
                "frc": d.get("frc"),
                "n_vertices": len(coords),
                "longitud_tramo_m": longitud_m(coords),
                "tramo_hash": huella(coords),
            })
        filas.append(fila)
        time.sleep(0.25)

    # Se guarda en la carpeta del día según la hora de la CDMX al iniciar la corrida
    fecha = inicio.astimezone(TZ_CDMX).date().isoformat()
    destino = CARPETA / f"fecha={fecha}" / "lecturas.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not destino.exists()
    with open(destino, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS, restval="", lineterminator="\n")
        if nuevo:
            w.writeheader()
        w.writerows(filas)

    fallas = [f["id"] for f in filas if f["error"]]
    print(f"{len(filas) - len(fallas)}/{len(filas)} lecturas OK -> {destino}")
    if fallas:
        print("Fallaron:", ", ".join(fallas))
    if len(fallas) == len(filas):
        sys.exit(1)  # todo falló (llave mala, sin red): que el workflow salga en rojo


if __name__ == "__main__":
    main()
