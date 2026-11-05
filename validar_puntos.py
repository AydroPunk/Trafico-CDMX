"""Valida que cada punto de puntos.csv caiga sobre la avenida y el sentido correctos.

Consulta una vez cada punto en TomTom Traffic Flow y revisa:
  1. Tipo de vía (frc). FRC0-FRC2 son vías rápidas y avenidas principales.
     Si una avenida que sí es la correcta sale con FRC más alto, se acepta
     poniendo frc_max (por ejemplo FRC3) en ese renglón de puntos.csv.
  2. Distancia entre el punto pedido y el tramo que TomTom regresó.
  3. Sentido del tramo contra la columna sentido (N, S, E u O) de puntos.csv.
  4. Que los dos puntos de una misma avenida y tramo midan sentidos opuestos.

El script NO revisa el nombre de la calle

Los renglones sin lat/lon se saltan sin gastar consultas. Escribe validacion_puntos.csv.
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
from collections import defaultdict

ZOOM = int(os.environ.get("TOMTOM_ZOOM", "10"))  # nivel de detalle de la red vial
URL = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{ZOOM}/json"
FRC_MAX_DEFAULT = "FRC2"
DIST_MAX_M = 60  # más lejos que esto, probablemente cayó en otra calle
TOL_SENTIDO = 67.5  # grados de tolerancia contra el sentido esperado
RUMBO_ESPERADO = {"N": 0, "E": 90, "S": 180, "O": 270}
CARDINALES = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]


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


def rumbo(coords):
    """Rumbo en grados (0 = norte, 90 = oriente) del primer al último vértice."""
    a, b = coords[0], coords[-1]
    lat1, lat2 = math.radians(a["latitude"]), math.radians(b["latitude"])
    dlon = math.radians(b["longitude"] - a["longitude"])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def dif_angular(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def main():
    key = os.environ.get("TOMTOM_API_KEY")
    if not key:
        sys.exit("Falta la variable TOMTOM_API_KEY. En fish: set -x TOMTOM_API_KEY \"tu_llave\"")

    with open("puntos.csv", newline="", encoding="utf-8") as f:
        todos = list(csv.DictReader(f))
    orden = [p["id"] for p in todos]

    pedidos = {a.upper() for a in sys.argv[1:]}
    puntos = todos
    if pedidos:
        desconocidos = pedidos - set(orden)
        if desconocidos:
            sys.exit(f"Esos ids no están en puntos.csv: {', '.join(sorted(desconocidos))}")
        puntos = [p for p in todos if p["id"] in pedidos]

    # Resultados previos: al validar solo algunos puntos, los demás se conservan.
    previos = {}
    if pedidos and os.path.exists("validacion_puntos.csv"):
        with open("validacion_puntos.csv", newline="", encoding="utf-8") as f:
            previos = {r["id"]: r for r in csv.DictReader(f)}

    filas = []
    print(f"{'id':<4} {'frc':<5} {'libre':>5} {'actual':>6} {'dist_m':>6} {'sent':<4} estado")
    for p in puntos:
        if not (p.get("lat") or "").strip() or not (p.get("lon") or "").strip():
            print(f"{p['id']:<4} {'':<5} {'':>5} {'':>6} {'':>6} {'':<4} SIN_COORDENADAS"
                  f"  ({p['avenida']} / {p['tramo']} / hacia {p.get('sentido', '?')})")
            filas.append({**p, "estado": "SIN_COORDENADAS"})
            continue
        lat, lon = float(p["lat"]), float(p["lon"])
        try:
            d = consultar(lat, lon, key)
        except urllib.error.HTTPError as e:
            print(f"{p['id']:<4} ERROR HTTP {e.code}: {e.read().decode(errors='replace')[:120]}")
            filas.append({**p, "estado": f"ERROR_{e.code}"})
            continue

        coords = d["coordinates"]["coordinate"]
        dist = distancia_a_tramo(lat, lon, coords)
        grados = rumbo(coords)
        problemas = []
        frc_max = (p.get("frc_max") or FRC_MAX_DEFAULT).strip().upper()
        if int(d["frc"].removeprefix("FRC")) > int(frc_max.removeprefix("FRC")):
            problemas.append("via_secundaria")
        if dist > DIST_MAX_M:
            problemas.append("lejos")
        esperado = (p.get("sentido") or "").strip().upper()
        if esperado in RUMBO_ESPERADO and dif_angular(grados, RUMBO_ESPERADO[esperado]) > TOL_SENTIDO:
            problemas.append("sentido")

        medio = coords[len(coords) // 2]
        fila = {
            **p,
            "frc": d["frc"],
            "free_flow_kmh": d["freeFlowSpeed"],
            "current_kmh": d["currentSpeed"],
            "confidence": d["confidence"],
            "dist_m": round(dist, 1),
            "rumbo_grados": round(grados),
            "sentido_medido": CARDINALES[round(grados / 45) % 8],
            "n_vertices": len(coords),
            "estado": "OK" if not problemas else "REVISAR:" + "+".join(problemas),
            "mapa": f"https://www.google.com/maps?q={lat},{lon}",
            "mapa_tramo": f"https://www.google.com/maps?q={medio['latitude']},{medio['longitude']}",
        }
        filas.append(fila)
        print(f"{p['id']:<4} {d['frc']:<5} {d['freeFlowSpeed']:>5} {d['currentSpeed']:>6} "
              f"{dist:>6.0f} {fila['sentido_medido']:<4} {fila['estado']}"
              f"  ({p['avenida']} / {p['tramo']} / esperado {esperado or '?'})")
        time.sleep(0.25)  # por debajo del límite de consultas por segundo

    for fila in filas:
        if fila["estado"].startswith("REVISAR"):
            print(f"  {fila['id']}: tu punto {fila['mapa']}\n       tramo TomTom {fila['mapa_tramo']}")

    # Mezcla con los resultados previos, en el orden de puntos.csv
    nuevos = {f["id"]: f for f in filas}
    todas = [t for t in (nuevos.get(i) or previos.get(i) for i in orden) if t]

    # Revisión por pares: misma avenida y tramo deben medir sentidos opuestos.
    pares = defaultdict(list)
    for t in todas:
        pares[(t["avenida"], t["tramo"])].append(t)
    avisos_pares = []
    for (avenida, tramo), grupo in pares.items():
        medidos = [g for g in grupo if g.get("rumbo_grados") not in (None, "")]
        if len(medidos) == 2:
            a, b = medidos
            dif = dif_angular(float(a["rumbo_grados"]), float(b["rumbo_grados"]))
            if dif < 120:
                avisos_pares.append(f"{a['id']} y {b['id']} ({avenida}) miden casi el mismo "
                                    f"sentido ({a['sentido_medido']} y {b['sentido_medido']})")

    campos = list(dict.fromkeys(k for fila in todas for k in fila))
    with open("validacion_puntos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, restval="")
        w.writeheader()
        w.writerows(todas)

    malos = [t["id"] for t in todas if t["estado"] != "OK"]
    print(f"\n{len(todas) - len(malos)}/{len(todas)} puntos OK en total. Detalle en validacion_puntos.csv")
    if malos:
        print("Faltan:", ", ".join(malos))
    for aviso in avisos_pares:
        print("Par sospechoso:", aviso)


if __name__ == "__main__":
    main()
