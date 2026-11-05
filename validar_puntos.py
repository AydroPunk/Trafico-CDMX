"""Valida que cada punto de puntos.csv caiga sobre la avenida correcta.

Consulta una vez cada punto en TomTom Traffic Flow y revisa tres cosas:
  1. Tipo de vía (frc). FRC0-FRC2 son vías rápidas y avenidas principales.
  2. Distancia entre el punto pedido y el tramo que TomTom regresó.
  3. Sentido del tramo (TomTom regresa un solo sentido por punto).

Uso (fish):
    set -x TOMTOM_API_KEY "tu_llave"
    python3 validar_puntos.py

Para validar solo algunos puntos (los demás se conservan en el CSV):
    python3 validar_puntos.py P01 P05 P11

Para probar otro nivel de zoom: set -x TOMTOM_ZOOM 12

Gasta una consulta por punto (24 en total). Escribe validacion_puntos.csv.
Solo usa la biblioteca estándar de Python.
"""

import csv
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ZOOM = int(os.environ.get("TOMTOM_ZOOM", "10"))  # nivel de detalle de la red vial
URL = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{ZOOM}/json"
FRC_OK = {"FRC0", "FRC1", "FRC2"}
DIST_MAX_M = 60  # más lejos que esto, probablemente cayó en otra calle


def consultar(lat, lon, key):
    params = urllib.parse.urlencode({"point": f"{lat},{lon}", "unit": "KMPH", "key": key})
    with urllib.request.urlopen(f"{URL}?{params}", timeout=20) as r:
        return json.load(r)["flowSegmentData"]


def a_metros(lat0, lat, lon):
    """Proyección local plana (suficiente para distancias de unos cientos de metros)."""
    k = 111_320.0
    return lon * k * math.cos(math.radians(lat0)), lat * k


def distancia_a_tramo(lat, lon, coords):
    """Distancia mínima en metros del punto a la polilínea del tramo."""
    px, py = a_metros(lat, lat, lon)
    pts = [a_metros(lat, c["latitude"], c["longitude"]) for c in coords]
    if len(pts) == 1:
        return math.dist((px, py), pts[0])
    mejor = float("inf")
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        mejor = min(mejor, math.dist((px, py), (ax + t * dx, ay + t * dy)))
    return mejor


def sentido(coords):
    """Rumbo del primer al último vértice, como punto cardinal."""
    a, b = coords[0], coords[-1]
    lat1, lat2 = math.radians(a["latitude"]), math.radians(b["latitude"])
    dlon = math.radians(b["longitude"] - a["longitude"])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    rumbo = (math.degrees(math.atan2(x, y)) + 360) % 360
    return ["N", "NE", "E", "SE", "S", "SO", "O", "NO"][round(rumbo / 45) % 8]


def main():
    key = os.environ.get("TOMTOM_API_KEY")
    if not key:
        sys.exit("Falta la variable TOMTOM_API_KEY. En fish: set -x TOMTOM_API_KEY \"tu_llave\"")

    with open("puntos.csv", newline="", encoding="utf-8") as f:
        puntos = list(csv.DictReader(f))

    pedidos = {a.upper() for a in sys.argv[1:]}
    if pedidos:
        desconocidos = pedidos - {p["id"] for p in puntos}
        if desconocidos:
            sys.exit(f"Esos ids no están en puntos.csv: {', '.join(sorted(desconocidos))}")
        puntos = [p for p in puntos if p["id"] in pedidos]

    # Resultados previos: al validar solo algunos puntos, los demás se conservan.
    previos = {}
    if pedidos and os.path.exists("validacion_puntos.csv"):
        with open("validacion_puntos.csv", newline="", encoding="utf-8") as f:
            previos = {r["id"]: r for r in csv.DictReader(f)}

    filas = []
    print(f"{'id':<4} {'frc':<5} {'libre':>5} {'actual':>6} {'dist_m':>6} {'sent':<4} estado")
    for p in puntos:
        lat, lon = float(p["lat"]), float(p["lon"])
        try:
            d = consultar(lat, lon, key)
        except urllib.error.HTTPError as e:
            print(f"{p['id']:<4} ERROR HTTP {e.code}: {e.read().decode(errors='replace')[:120]}")
            filas.append({**p, "estado": f"ERROR_{e.code}"})
            continue
        coords = d["coordinates"]["coordinate"]
        dist = distancia_a_tramo(lat, lon, coords)
        problemas = []
        if d["frc"] not in FRC_OK:
            problemas.append("via_secundaria")
        if dist > DIST_MAX_M:
            problemas.append("lejos")
        estado = "OK" if not problemas else "REVISAR:" + "+".join(problemas)
        medio = coords[len(coords) // 2]
        fila = {
            **p,
            "frc": d["frc"],
            "free_flow_kmh": d["freeFlowSpeed"],
            "current_kmh": d["currentSpeed"],
            "confidence": d["confidence"],
            "dist_m": round(dist, 1),
            "sentido": sentido(coords),
            "n_vertices": len(coords),
            "estado": estado,
            "mapa": f"https://www.google.com/maps?q={lat},{lon}",
            "mapa_tramo": f"https://www.google.com/maps?q={medio['latitude']},{medio['longitude']}",
        }
        filas.append(fila)
        print(f"{p['id']:<4} {d['frc']:<5} {d['freeFlowSpeed']:>5} {d['currentSpeed']:>6} "
              f"{dist:>6.0f} {fila['sentido']:<4} {estado}  ({p['avenida']} / {p['tramo']})")
        time.sleep(0.25)  # por debajo del límite de consultas por segundo

    for fila in filas:
        if fila["estado"] != "OK" and "mapa_tramo" in fila:
            print(f"  {fila['id']}: tu punto {fila['mapa']}\n       tramo TomTom {fila['mapa_tramo']}")

    # Mezcla con los resultados previos, en el orden de puntos.csv
    nuevos = {f["id"]: f for f in filas}
    with open("puntos.csv", newline="", encoding="utf-8") as f:
        orden = [r["id"] for r in csv.DictReader(f)]
    todas = [nuevos.get(i) or previos.get(i) for i in orden]
    todas = [t for t in todas if t]

    campos = list(dict.fromkeys(k for fila in todas for k in fila))
    with open("validacion_puntos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, restval="")
        w.writeheader()
        w.writerows(todas)

    malos = [t["id"] for t in todas if t["estado"] != "OK"]
    print(f"\n{len(todas) - len(malos)}/{len(todas)} puntos OK en total. Detalle en validacion_puntos.csv")
    if malos:
        print("Faltan:", ", ".join(malos))


if __name__ == "__main__":
    main()