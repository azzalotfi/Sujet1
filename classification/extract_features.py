"""
Étape 3a — Extraction de features spectrales par parcelle
À partir des stack.tif Sentinel-2 téléchargés en Étape 2.

Features extraites par parcelle (statistiques sur pixels valides) :
  Bandes brutes  : mean/std de B02, B03, B04, B08, B11
  Indices         : NDVI, NDWI, NDRE (Normalised Difference Red Edge)
  Texture (GLCM) : contraste, homogénéité (calculé sur B04)
  Géométrique     : area_ha
  Label           : 0=extensif, 1=intensif (hyper-intensif=2 si disponible)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from skimage.feature import graycomatrix, graycoprops

warnings.filterwarnings("ignore")

BASE_DIR      = Path(__file__).parent.parent
SENTINEL_DIR  = BASE_DIR / "data_preparation" / "sentinel2"
GEOJSON_PATH  = BASE_DIR / "data_preparation" / "output" / "all_parcelles.geojson"
OUT_CSV       = Path(__file__).parent / "features.csv"

# Ordre des bandes dans stack.tif (défini dans download_sentinel2.py)
BAND_NAMES = ["B02", "B03", "B04", "B08", "B11"]
# Indices : B02=0, B03=1, B04=2, B08=3, B11=4
IDX = {b: i for i, b in enumerate(BAND_NAMES)}

LABEL_MAP = {"extensif": 0, "intensif": 1, "hyper-intensif": 2}


def safe_norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b) avec gestion division par zéro."""
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denom == 0, 0.0, (a - b) / denom)
    return result


def glcm_features(band2d: np.ndarray) -> dict:
    """
    Calcule contraste et homogénéité GLCM sur la bande 2D.
    Quantifie la texture (oliviers ont une texture caractéristique).
    """
    try:
        valid = band2d[np.isfinite(band2d)]
        if len(valid) < 100:
            return {"glcm_contrast": np.nan, "glcm_homogeneity": np.nan}
        # Normaliser sur 256 niveaux
        vmin, vmax = np.nanpercentile(valid, [2, 98])
        if vmax == vmin:
            return {"glcm_contrast": 0.0, "glcm_homogeneity": 1.0}
        img = np.clip((band2d - vmin) / (vmax - vmin) * 255, 0, 255)
        img = np.nan_to_num(img, nan=0).astype(np.uint8)
        # GLCM avec un décalage [1,0] (horizontal)
        g = graycomatrix(img, distances=[1], angles=[0], levels=256,
                         symmetric=True, normed=True)
        return {
            "glcm_contrast":     float(graycoprops(g, "contrast")[0, 0]),
            "glcm_homogeneity":  float(graycoprops(g, "homogeneity")[0, 0]),
        }
    except Exception:
        return {"glcm_contrast": np.nan, "glcm_homogeneity": np.nan}


def extract_parcel_features(stack_path: Path, area_ha: float, systeme: str) -> dict:
    """Lit un stack.tif et retourne un dict de features."""
    with rasterio.open(stack_path) as src:
        data = src.read().astype(np.float32)  # (n_bands, H, W)

    n_bands = data.shape[0]
    if n_bands < 5:
        # Compléter avec NaN si bandes manquantes
        full = np.full((5, data.shape[1], data.shape[2]), np.nan, dtype=np.float32)
        full[:n_bands] = data
        data = full

    # Masque des pixels valides
    valid_mask = np.all(np.isfinite(data), axis=0)

    feats = {"area_ha": area_ha, "systeme": systeme,
             "label": LABEL_MAP.get(systeme, -1)}

    # 1. Statistiques par bande
    for band_name, bi in IDX.items():
        band = data[bi]
        vals = band[valid_mask & np.isfinite(band)]
        if len(vals) == 0:
            feats[f"{band_name}_mean"] = np.nan
            feats[f"{band_name}_std"]  = np.nan
        else:
            feats[f"{band_name}_mean"] = float(np.mean(vals))
            feats[f"{band_name}_std"]  = float(np.std(vals))

    # 2. Indices spectraux
    B02 = data[IDX["B02"]]
    B03 = data[IDX["B03"]]
    B04 = data[IDX["B04"]]
    B08 = data[IDX["B08"]]
    B11 = data[IDX["B11"]]

    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = safe_norm_diff(B08, B04)
    # NDWI = (Green - NIR) / (Green + NIR)
    ndwi = safe_norm_diff(B03, B08)
    # NDRE simulé avec SWIR : (NIR - SWIR) / (NIR + SWIR)
    ndre = safe_norm_diff(B08, B11)

    for name, arr in [("ndvi", ndvi), ("ndwi", ndwi), ("ndre", ndre)]:
        vals = arr[valid_mask & np.isfinite(arr)]
        if len(vals) == 0:
            feats[f"{name}_mean"] = np.nan
            feats[f"{name}_std"]  = np.nan
            feats[f"{name}_p25"]  = np.nan
            feats[f"{name}_p75"]  = np.nan
        else:
            feats[f"{name}_mean"] = float(np.mean(vals))
            feats[f"{name}_std"]  = float(np.std(vals))
            feats[f"{name}_p25"]  = float(np.percentile(vals, 25))
            feats[f"{name}_p75"]  = float(np.percentile(vals, 75))

    # 3. Texture GLCM sur B04 (Red — contraste sol/feuillage)
    glcm = glcm_features(B04)
    feats.update(glcm)

    # 4. Ratio B08/B04 (rapport NIR/Rouge)
    valid_ratio = valid_mask & (B04 > 0)
    ratio = np.where(valid_ratio, B08 / np.where(B04 == 0, 1, B04), np.nan)
    vals_r = ratio[np.isfinite(ratio)]
    feats["nir_red_ratio"] = float(np.mean(vals_r)) if len(vals_r) > 0 else np.nan

    return feats


def run():
    # Charger les métadonnées des parcelles
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        fc = json.load(f)

    rows = []
    missing = []

    for feat in fc["features"]:
        props = feat["properties"]
        pid     = props["id"]
        systeme = props["systeme"]
        area_ha = props.get("area_ha", 0.0)
        split   = props.get("split", "train")

        stack_path = SENTINEL_DIR / pid / "stack.tif"
        if not stack_path.exists():
            missing.append(pid)
            continue

        try:
            feats = extract_parcel_features(stack_path, area_ha, systeme)
            feats["parcel_id"] = pid
            feats["split"]     = split
            rows.append(feats)
            print(f"  ✅ {pid} ({systeme}) — {len(feats)} features")
        except Exception as e:
            print(f"  ❌ {pid}: {e}")
            missing.append(pid)

    df = pd.DataFrame(rows)
    # Mettre parcel_id et split en colonnes de tête
    cols = ["parcel_id", "split", "systeme", "label", "area_ha"]
    rest = [c for c in df.columns if c not in cols]
    df = df[cols + rest]

    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n✅ Features extraites pour {len(rows)} parcelles → {OUT_CSV}")
    if missing:
        print(f"⚠️  {len(missing)} parcelles sans stack.tif : {missing}")
    return df


if __name__ == "__main__":
    df = run()
    print(df.describe())
