"""Valida el metodo (landmarks + calibracion con marcador de referencia) sobre el
dataset publico de Acme AI / BMGF -- mucho mas grande que el propio (4.728 fotos
reales de campo en Bangladesh, peso de bascula, keypoints anotados por
profesionales, y una calcomania de referencia ya segmentada en las mascaras).

Esto NO es el pipeline de produccion propio (esos son entrenar_pose.py +
medidas_desde_keypoints.py / evaluar_holdout.py, sobre keypoints PREDICHOS por
nuestro modelo). Esto valida que el METODO funciona -- con keypoints de
referencia, no los nuestros -- antes de invertir en recolectar el dataset propio
con el marcador incluido.

Resultado (15/09/2026): sin calibrar, MAPE ~22% (apenas mejor que el baseline
tonto). Calibrando con el marcador, ~14-17% segun banda de peso -- ya cumple el
objetivo del acta (<15%) en la banda de 100-200 kg, que es la mayoria de los datos.
Se investigo si el error restante viene de imprecision de calibracion (tamano
aparente del sticker, o que no quede de frente a camara) y NO -- ninguna de las
dos correlaciona con el error. El techo parece estructural: una sola foto lateral
con un solo marcador no puede recuperar la circunferencia real del pecho (solo un
proxy de profundidad), ni corregir que cada landmark esta a una profundidad
distinta de la camara. Detalle completo en README.md.

Uso:  python src/validar_metodo_acmeai.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# Distintos batches, distinto color de sticker y distinto orden/nombre de keypoints.
BATCHES = [
    dict(
        nombre="b2",
        json_path="Vector/B2/Side/data/Side/COCO_Side.json",
        mask_dir="Pixel/B2/Side/annotations",
        sticker_rgb=(255, 240, 0),
        patron=re.compile(r"^([\d.]+)_s_(\d+)_[\d.]+_([MF])\.jpg$", re.IGNORECASE),
        extraer=lambda g: (g[0], g[1], g[2]),  # id, peso, sexo
        con_altura=False,
    ),
    dict(
        nombre="b3",
        json_path="Vector/B3/Side/data/COCO_Side.json",
        mask_dir="Pixel/B3/annotations",
        sticker_rgb=(0, 117, 255),
        patron=re.compile(r"^(\d+)_s_(\d+)_([MF])\.jpg$", re.IGNORECASE),
        extraer=lambda g: (g[0], g[1], g[2]),  # id, peso, sexo
        con_altura=True,
    ),
    dict(
        nombre="b4",
        json_path="Vector/B4/Side/data/coco_b4_side.json",
        mask_dir="Pixel/B4/Side/annotations",
        sticker_rgb=(0, 117, 255),
        patron=re.compile(r"^(\d+)_(b4-\d+)_s_(\d+)_([MF])\.jpg$", re.IGNORECASE),
        extraer=lambda g: (g[0], g[2], g[3]),  # id, batchid, peso, sexo -> saltea batchid
        con_altura=True,
    ),
]


def normalizar(nombre):
    n = nombre.lower().lstrip("0123456789_")
    return n.replace("-", "_")


def procesar_batch(cfg):
    raiz = config.DATASET_ACMEAI
    with open(raiz / cfg["json_path"], encoding="utf-8") as f:
        d = json.load(f)
    imgs = {im["id"]: im for im in d["images"]}
    nombres_kp = [normalizar(n) for n in d["categories"][0]["keypoints"]]
    idx = {n: i for i, n in enumerate(nombres_kp)}

    necesarios = ["wither", "pinbone", "front_top", "front_bottom", "rear_top", "rear_bottom"]
    # B2 usa "front-top/front-bottom/rear-top/rear-bottom", B3/B4 usan
    # "front_girth_top/bottom" y "rear_girth_top/bottom" -- unificamos alias.
    alias = {
        "front_girth_top": "front_top", "front_girth_bottom": "front_bottom",
        "rear_girth_top": "rear_top", "rear_girth_bottom": "rear_bottom",
    }
    for k, v in list(idx.items()):
        if k in alias:
            idx[alias[k]] = v

    mask_dir = raiz / cfg["mask_dir"]
    filas, sin_mascara, sin_sticker = [], 0, 0
    for ann in d["annotations"]:
        fn = imgs[ann["image_id"]]["file_name"]
        m = cfg["patron"].match(fn)
        if not m or ann["num_keypoints"] < len(nombres_kp):
            continue
        aid, peso, sexo = cfg["extraer"](m.groups())

        mpath = mask_dir / f"{fn}___fuse.png"
        if not mpath.exists():
            sin_mascara += 1
            continue
        mask = np.array(Image.open(mpath).convert("RGB"))
        sticker_px = (mask == cfg["sticker_rgb"]).all(axis=-1)
        if sticker_px.sum() < 20:
            sin_sticker += 1
            continue
        ys, xs = np.where(sticker_px)
        ancho_mask, alto_mask = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        mh, mw = mask.shape[:2]
        im = imgs[ann["image_id"]]
        sx, sy = mw / im["width"], mh / im["height"]
        sticker_size = ((ancho_mask / sx) * (alto_mask / sy)) ** 0.5

        kp = np.array(ann["keypoints"], float).reshape(len(nombres_kp), 3)[:, :2]
        dist = lambda a, b: float(np.linalg.norm(kp[idx[a]] - kp[idx[b]]))
        fila = {
            "batch": cfg["nombre"], "grupo": f"{cfg['nombre']}_{aid}_{peso}_{sexo}",
            "peso_kg": float(peso), "largo": dist("wither", "pinbone"),
            "girth_f": dist("front_top", "front_bottom"),
            "girth_r": dist("rear_top", "rear_bottom"),
            "sticker_px": sticker_size,
        }
        if cfg["con_altura"]:
            fila["altura"] = dist("height_top", "height_bottom")
        filas.append(fila)

    print(f"  {cfg['nombre']}: {len(filas)} utilizables "
          f"(sin mascara {sin_mascara}, sin sticker {sin_sticker})")
    return pd.DataFrame(filas)


def evaluar(df, cols, etiqueta, franjas=None):
    X = df[cols].to_numpy(float)
    y = df["peso_kg"].to_numpy(float)
    g = df["grupo"].to_numpy()
    base = np.mean(np.abs((y - np.median(y)) / y)) * 100
    print(f"\n=== {etiqueta} (n={len(df)}, peso {y.min():.0f}-{y.max():.0f} kg, "
          f"baseline {base:.2f}%) ===")
    for nombre, m in [
        ("Ridge", make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 40)))),
        ("Boosting", HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=400, min_samples_leaf=5,
            l2_regularization=1.0, random_state=0)),
    ]:
        cv = GroupKFold(n_splits=5)
        p = cross_val_predict(m, X, y, cv=cv, groups=g)
        mape = np.mean(np.abs((y - p) / y)) * 100
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  {nombre:9s} MAPE {mape:5.2f}%   R2 {r2:+.3f}")
        if nombre == "Boosting" and franjas:
            err = np.abs((y - p) / y) * 100
            for lo, hi in franjas:
                s = (y >= lo) & (y < hi)
                if s.sum() >= 10:
                    print(f"      {lo:4d}-{hi if hi < 9999 else '+':>4} kg  "
                          f"n={s.sum():4d}  MAPE {err[s].mean():5.2f}%")


def main():
    if not config.DATASET_ACMEAI.exists():
        print(f"No se encontro el dataset en {config.DATASET_ACMEAI}")
        print("Ver CLAUDE.md para donde conseguirlo.")
        return

    print("Procesando batches...")
    dfs = [procesar_batch(cfg) for cfg in BATCHES]
    df = pd.concat(dfs, ignore_index=True)
    config.TABLAS.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TABLAS / "acmeai_b2_b3_b4_side.csv", index=False)

    # las 3 features que tienen TODOS los batches (B2 no tiene 'altura')
    for c in ["largo", "girth_f", "girth_r"]:
        df[f"cal_{c}"] = df[c] / df["sticker_px"]
    cols_comun = [f"cal_{c}" for c in ["largo", "girth_f", "girth_r"]]
    evaluar(df, cols_comun, "B2+B3+B4, 3 features comunes")

    # con altura, solo B3+B4 -- este es el numero que se reporta (14%): un solo
    # modelo entrenado con TODO el rango de peso, y se mira el error por franja
    # (no se entrena un modelo aparte para cada franja, seria optimista)
    df34 = df[df.batch != "b2"].copy()
    df34["cal_altura"] = df34["altura"] / df34["sticker_px"]
    cols_4 = cols_comun + ["cal_altura"]
    franjas = [(0, 100), (100, 200), (200, 300), (300, 10_000)]
    evaluar(df34, cols_4, "B3+B4, con altura (4 features) -- NUMERO A REPORTAR", franjas)


if __name__ == "__main__":
    main()
