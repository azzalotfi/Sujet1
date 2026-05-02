"""
Étape 5 — API REST FastAPI
Endpoint principal : POST /api/cartographier
  - Reçoit un polygone GeoJSON + une date
  - Recherche la meilleure image Sentinel-2 sur la période
  - Calcule les features spectrales
  - Classifie en extensif / intensif / hyper-intensif
  - Retourne un GeoJSON avec les parcelles détectées et classifiées

Lancement : uvicorn api.main:app --reload --port 8000
"""

import json
import sys
import tempfile
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
import joblib

warnings.filterwarnings("ignore")

# Ajouter les modules du projet au path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data_preparation"))
sys.path.insert(0, str(PROJECT_ROOT / "classification"))

from download_sentinel2 import (  # noqa: E402
    BAND_ORDER,
    bbox_from_geom,
    build_cube,
    connect,
)
from extract_features import (  # noqa: E402
    safe_norm_diff,
    glcm_features,
    IDX,
)

# ──────────────────────────────────────────────
# Chargement du modèle
# ──────────────────────────────────────────────
MODEL_DIR = PROJECT_ROOT / "classification" / "model"

def load_models():
    clf     = joblib.load(MODEL_DIR / "classifier.pkl")
    imputer = joblib.load(MODEL_DIR / "imputer.pkl")
    scaler  = joblib.load(MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    return clf, imputer, scaler, feature_names

try:
    clf, imputer, scaler, feature_names = load_models()
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️  Modèle non chargé : {e} — le classifieur ne sera pas disponible")
    MODEL_LOADED = False
    clf = imputer = scaler = feature_names = None

# ──────────────────────────────────────────────
# App FastAPI
# ──────────────────────────────────────────────
app = FastAPI(
    title="Olive Grove Classifier",
    description="Détecte et classe les oliveraies à partir d'une image Sentinel-2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir les fichiers statiques frontend (si présents)
frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ──────────────────────────────────────────────
# Schémas Pydantic
# ──────────────────────────────────────────────
class CartographierRequest(BaseModel):
    polygone_perimetre: Dict[str, Any]
    date: str  # "YYYY-MM-DD"

    @field_validator("polygone_perimetre")
    @classmethod
    def validate_geojson(cls, v):
        if v.get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError("polygone_perimetre doit être un GeoJSON Polygon ou MultiPolygon")
        if "coordinates" not in v:
            raise ValueError("polygone_perimetre manque le champ 'coordinates'")
        return v

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date doit être au format YYYY-MM-DD")
        return v


# ──────────────────────────────────────────────
# Fonctions utilitaires
# ──────────────────────────────────────────────
BAND_NAMES = ["B02", "B03", "B04", "B08", "B11"]

OPENEO_CONN = None


def get_openeo_connection():
    global OPENEO_CONN
    if OPENEO_CONN is None:
        OPENEO_CONN = connect()
    return OPENEO_CONN

CLASS_LABELS = {0: "extensif", 1: "intensif", 2: "hyper-intensif"}
CLASS_COLORS = {
    "extensif":       "#2ecc71",
    "intensif":       "#f39c12",
    "hyper-intensif": "#e74c3c",
    "inconnu":        "#95a5a6",
}


def compute_features_from_stack(bands: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Calcule les features à partir d'un dict {band_name: array2D}."""
    feats = {}
    # Créer le tableau (5, H, W)
    h = w = None
    for arr in bands.values():
        if arr is not None and np.isfinite(arr).any():
            h, w = arr.shape
            break
    if h is None:
        return {}

    data = np.full((5, h, w), np.nan, dtype=np.float32)
    for i, bname in enumerate(BAND_NAMES):
        if bname in bands and bands[bname] is not None:
            data[i] = bands[bname]

    valid_mask = np.all(np.isfinite(data), axis=0)

    for bname, bi in IDX.items():
        arr = data[bi]
        vals = arr[valid_mask & np.isfinite(arr)]
        feats[f"{bname}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
        feats[f"{bname}_std"]  = float(np.std(vals))  if len(vals) else np.nan

    B02, B03, B04, B08, B11 = data[0], data[1], data[2], data[3], data[4]

    ndvi = safe_norm_diff(B08, B04)
    ndwi = safe_norm_diff(B03, B08)
    ndre = safe_norm_diff(B08, B11)

    for name, arr in [("ndvi", ndvi), ("ndwi", ndwi), ("ndre", ndre)]:
        vals = arr[valid_mask & np.isfinite(arr)]
        feats[f"{name}_mean"] = float(np.mean(vals))       if len(vals) else np.nan
        feats[f"{name}_std"]  = float(np.std(vals))        if len(vals) else np.nan
        feats[f"{name}_p25"]  = float(np.percentile(vals, 25)) if len(vals) else np.nan
        feats[f"{name}_p75"]  = float(np.percentile(vals, 75)) if len(vals) else np.nan

    glcm = glcm_features(B04)
    feats.update(glcm)

    valid_ratio = valid_mask & (B04 > 0)
    ratio = np.where(valid_ratio, B08 / np.where(B04 == 0, 1, B04), np.nan)
    vals_r = ratio[np.isfinite(ratio)]
    feats["nir_red_ratio"] = float(np.mean(vals_r)) if len(vals_r) else np.nan

    return feats


def classify_parcel(feats: Dict[str, float], area_ha: float = 0.0) -> Dict[str, Any]:
    """Classifie une parcelle à partir de ses features."""
    if not MODEL_LOADED:
        return {"classe": "inconnu", "probabilites": {}}

    feats["area_ha"] = area_ha
    x = np.array([[feats.get(f, np.nan) for f in feature_names]], dtype=np.float32)
    x = imputer.transform(x)
    x = scaler.transform(x)

    pred = int(clf.predict(x)[0])
    proba = clf.predict_proba(x)[0]
    classe = CLASS_LABELS.get(pred, "inconnu")

    return {
        "classe": classe,
        "probabilites": {CLASS_LABELS[i]: float(p) for i, p in enumerate(proba)},
    }


# ──────────────────────────────────────────────
# Endpoint principal
# ──────────────────────────────────────────────
@app.get("/")
def index():
    return {"message": "Olive Grove Classifier API — voir /docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED}


@app.post("/api/cartographier")
async def cartographier(req: CartographierRequest):
    """
    Détecte et classe les oliveraies dans un périmètre donné.

    Body: { "polygone_perimetre": <GeoJSON Polygon>, "date": "YYYY-MM-DD" }
    Retourne: GeoJSON FeatureCollection avec classe et couleur par zone détectée.
    """
    geom = req.polygone_perimetre
    date_obj = datetime.strptime(req.date, "%Y-%m-%d")

    # Fenêtre de recherche : ±30 jours autour de la date
    date_start = (date_obj - timedelta(days=30)).strftime("%Y-%m-%d")
    date_end   = (date_obj + timedelta(days=30)).strftime("%Y-%m-%d")

    # 1. Télécharger un composite openEO (médiane temporelle, masque SCL côté serveur)
    conn = get_openeo_connection()
    bbox = bbox_from_geom(geom)

    bands = {b: None for b in BAND_ORDER}
    try:
        cube = build_cube(conn, bbox, date_start=date_start, date_end=date_end, max_cloud=30)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            cube.download(str(tmp_path), format="GTiff")
            with rasterio.open(tmp_path) as src:
                data = src.read().astype(np.float32)

            for i, bname in enumerate(BAND_ORDER):
                if i < data.shape[0]:
                    bands[bname] = data[i]
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur openEO pendant le téléchargement Sentinel-2: {e}"
        )

    if not any(v is not None for v in bands.values()):
        raise HTTPException(status_code=500, detail="Impossible de télécharger le composite Sentinel-2")

    # 3. Calculer les features spectrales
    feats = compute_features_from_stack(bands)
    if not feats:
        raise HTTPException(status_code=500, detail="Erreur lors du calcul des features")

    # 4. Calculer l'aire approximative du polygone (en ha)
    from shapely.geometry import shape as shp_shape
    geom_shape = shp_shape(geom)
    # Approximation : 1 deg² ≈ 111km × 111km × cos(lat) à la latitude tunisienne (~35°)
    lat_center = geom_shape.centroid.y
    deg_to_km = 111.0 * np.cos(np.radians(lat_center))
    area_deg2  = geom_shape.area
    area_km2   = area_deg2 * deg_to_km * deg_to_km
    area_ha    = area_km2 * 100.0

    # 5. Classifier
    result = classify_parcel(feats, area_ha)
    classe = result["classe"]
    color  = CLASS_COLORS.get(classe, CLASS_COLORS["inconnu"])

    # Calculer NDVI moyen pour info
    ndvi_mean = feats.get("ndvi_mean", None)

    # 6. Construire la réponse GeoJSON
    feature = {
        "type": "Feature",
        "geometry": geom,
        "properties": {
            "classe":         classe,
            "couleur":        color,
            "probabilites":   result["probabilites"],
            "scene_date":     req.date,
            "cloud_cover_pct": None,
            "area_ha":        round(float(area_ha), 1),
            "ndvi_mean":      round(float(ndvi_mean), 4) if ndvi_mean else None,
            "scene_id":       f"openEO_composite_{date_start}_{date_end}",
        }
    }

    geojson_response = {
        "type": "FeatureCollection",
        "features": [feature],
    }

    return JSONResponse(content=geojson_response)
