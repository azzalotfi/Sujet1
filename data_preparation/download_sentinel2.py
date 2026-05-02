"""
Étape 2 — Téléchargement des bandes Sentinel-2 L2A par parcelle
- Source  : Copernicus Data Space Ecosystem (openEO)
            https://openeo.dataspace.copernicus.eu
- Auth    : OIDC — créer un compte gratuit sur https://dataspace.copernicus.eu
- Bandes  : B02, B03, B04, B08, B11
- Période : mai-juin (contraste olivier/sol maximal)
- Masque  : SCL côté serveur (classes nuages/ombres/eau masquées)
- Réduct. : médiane temporelle sur la période (composite robuste)
- Sortie  : data_preparation/sentinel2/<parcel_id>/stack.tif (5 bandes)
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import numpy as np
import rasterio
from shapely.geometry import shape
import openeo

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
OUT_DIR  = BASE_DIR / "sentinel2"
GEOJSON  = BASE_DIR / "output" / "all_parcelles.geojson"

OUT_DIR.mkdir(exist_ok=True)
STATUS_PATH = OUT_DIR / "_download_status.json"

# ─────────────────────────────────────────────
# Paramètres openEO / Copernicus
# ─────────────────────────────────────────────
OPENEO_URL  = "https://openeo.dataspace.copernicus.eu"
COLLECTION  = "SENTINEL2_L2A"
BAND_ORDER  = ["B02", "B03", "B04", "B08", "B11"]   # ordre dans stack.tif
ALL_BANDS   = BAND_ORDER + ["SCL"]

YEAR       = 2025
DATE_START = f"{YEAR}-05-01"
DATE_END   = f"{YEAR}-06-30"
MAX_CLOUD  = 15   # % couverture nuageuse max
DOWNLOAD_TIMEOUT_SEC = 900  # 5 min max par parcelle pour le download

BUFFER_DEG = 0.001   # buffer autour de la parcelle

# Classes SCL à MASQUER : 0=no data, 1=saturé, 2=dark area, 3=ombre nuage,
# 6=eau, 8=nuage moyen, 9=nuage épais, 10=cirrus, 11=neige
SCL_MASK_CLASSES = [0, 1, 2, 3, 6, 8, 9, 10, 11]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_status() -> dict:
    if not STATUS_PATH.exists():
        return {"updated_at": utc_now(), "completed": [], "failed": {}}
    try:
        with open(STATUS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("updated_at", utc_now())
        data.setdefault("completed", [])
        data.setdefault("failed", {})
        return data
    except Exception:
        return {"updated_at": utc_now(), "completed": [], "failed": {}}


def save_status(status: dict) -> None:
    status["updated_at"] = utc_now()
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


def is_parcel_complete(parcel_dir: Path) -> bool:
    stack_path = parcel_dir / "stack.tif"
    meta_path = parcel_dir / "metadata.json"
    if not stack_path.exists() or not meta_path.exists():
        return False
    try:
        with rasterio.open(stack_path) as src:
            return src.count >= 5 and src.width > 0 and src.height > 0
    except Exception:
        return False


def cleanup_incomplete(parcel_dir: Path) -> None:
    # Supprime les artefacts d'une exécution interrompue pour une reprise propre.
    for p in [
        parcel_dir / "_tmp.tif",
        parcel_dir / "_in_progress.json",
    ]:
        if p.exists():
            p.unlink()


def _download_worker(cube: openeo.DataCube, out_path: Path, q: Queue) -> None:
    try:
        cube.download(str(out_path), format="GTiff")
        q.put((True, None))
    except Exception as e:
        q.put((False, str(e)))


def download_with_timeout(cube: openeo.DataCube, out_path: Path, timeout_sec: int) -> tuple[bool, str | None]:
    """Exécute cube.download avec timeout pour éviter un blocage infini."""
    q: Queue = Queue(maxsize=1)
    t = Thread(target=_download_worker, args=(cube, out_path, q), daemon=True)
    t.start()
    try:
        ok, err = q.get(timeout=timeout_sec)
        return bool(ok), err
    except Empty:
        return False, f"Timeout après {timeout_sec}s"

# ─────────────────────────────────────────────
# Connexion openEO (une seule fois)
# ─────────────────────────────────────────────
def connect() -> openeo.Connection:
    """
    Connexion au backend Copernicus Data Space.
    Premier appel : ouvre un navigateur (ou affiche un code device) pour
    s'authentifier avec votre compte dataspace.copernicus.eu.
    Les tokens sont ensuite sauvegardés localement (~/.config/openeo-python-client/).
    """
    conn = openeo.connect(OPENEO_URL)
    conn.authenticate_oidc()
    info = conn.describe_account()
    print(f"  🔐 Connecté en tant que : {info.get('name', info.get('user_id', '?'))}")
    return conn


# ─────────────────────────────────────────────
# Fonctions utilitaires
# ─────────────────────────────────────────────
def bbox_from_geom(geom_dict: dict) -> dict:
    """Bounding box {west, south, east, north} avec buffer."""
    geom = shape(geom_dict)
    b = geom.bounds
    return {
        "west":  b[0] - BUFFER_DEG,
        "south": b[1] - BUFFER_DEG,
        "east":  b[2] + BUFFER_DEG,
        "north": b[3] + BUFFER_DEG,
    }


def build_cube(
    conn: openeo.Connection,
    bbox: dict,
    date_start: str = DATE_START,
    date_end: str = DATE_END,
    max_cloud: int = MAX_CLOUD,
) -> openeo.DataCube:
    """
    Construit le datacube openEO :
    1. Charge SENTINEL2_L2A sur la bbox et la période
    2. Masque les pixels invalides via SCL (côté serveur)
    3. Réduit la dimension temporelle par médiane
    4. Retourne un cube 5 bandes (B02, B03, B04, B08, B11)
    """
    cube = conn.load_collection(
        COLLECTION,
        spatial_extent=bbox,
        temporal_extent=[date_start, date_end],
        bands=ALL_BANDS,
        max_cloud_cover=max_cloud,
    )

    # Masque SCL : invalide les pixels nuages/ombres/eau/neige
    scl = cube.band("SCL")
    mask = None
    for cls in SCL_MASK_CLASSES:
        m = scl == cls
        mask = m if mask is None else (mask | m)
    cube_masked = cube.mask(mask)

    # Garder uniquement les bandes spectrales
    cube_bands = cube_masked.filter_bands(BAND_ORDER)

    # Réduction temporelle : médiane (composite robuste aux nuages résiduels)
    cube_reduced = cube_bands.reduce_dimension(
        dimension="t",
        reducer="median",
    )
    return cube_reduced


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────
def process_parcel(feature: dict, conn: openeo.Connection) -> bool:
    """
    Traite une parcelle via openEO :
    - Construit le datacube (masque SCL + médiane temporelle)
    - Télécharge le résultat en GeoTIFF multi-bandes (stack.tif)
    Retourne True si succès.
    """
    props     = feature["properties"]
    parcel_id = props["id"]
    systeme   = props["systeme"]
    geom      = feature["geometry"]

    parcel_dir = OUT_DIR / parcel_id
    stack_path = parcel_dir / "stack.tif"
    marker_path = parcel_dir / "_in_progress.json"

    if is_parcel_complete(parcel_dir):
        print(f"  ⏭  {parcel_id} déjà traité, skip.")
        return True

    parcel_dir.mkdir(exist_ok=True)
    cleanup_incomplete(parcel_dir)

    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump({"parcel_id": parcel_id, "started_at": utc_now()}, f, indent=2)

    try:
        print("    ⏳ Préparation du cube openEO...")
        bbox = bbox_from_geom(geom)
        cube = build_cube(conn, bbox)

        tmp_path = parcel_dir / "_tmp.tif"
        print(f"    ⏳ Téléchargement du GeoTIFF depuis openEO (timeout {DOWNLOAD_TIMEOUT_SEC}s)...")
        ok_dl, err_dl = download_with_timeout(cube, tmp_path, DOWNLOAD_TIMEOUT_SEC)
        if not ok_dl:
            raise TimeoutError(err_dl or "Timeout de téléchargement")
        print("    ✅ Téléchargement terminé")

        if not tmp_path.exists():
            raise FileNotFoundError("Le fichier téléchargé est introuvable")

        tmp_path.rename(stack_path)

        # Vérification
        with rasterio.open(stack_path) as src:
            n_bands  = src.count
            shape_rc = (src.height, src.width)

        print(f"    💾 stack.tif sauvegardé ({n_bands} bandes, {shape_rc})")

        # Métadonnées
        meta = {
            "parcel_id":       parcel_id,
            "systeme":         systeme,
            "source":          "openeo-copernicus",
            "temporal_extent": [DATE_START, DATE_END],
            "max_cloud_cover": MAX_CLOUD,
            "bands":           BAND_ORDER,
            "shape":           list(shape_rc),
        }
        with open(parcel_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        if marker_path.exists():
            marker_path.unlink()

        return True

    except Exception as e:
        print(f"  ❌ {parcel_id}: {e}")
        for p in [parcel_dir / "_tmp.tif", stack_path, marker_path]:
            if p.exists():
                try:
                    p.unlink()
                except PermissionError:
                    pass  # fichier encore verrouillé par le thread de téléchargement
        return False

# ─────────────────────────────────────────────
# Entrée principale
# ─────────────────────────────────────────────
if __name__ == "__main__":
    with open(GEOJSON, encoding="utf-8") as f:
        fc = json.load(f)

    features = fc["features"]
    print(f"📡 Traitement de {len(features)} parcelles via openEO Copernicus...")
    print(f"   Période   : {DATE_START} → {DATE_END}")
    print(f"   Max nuages: {MAX_CLOUD}%")
    print(f"   Sortie    : {OUT_DIR}")
    print()

    # Connexion unique partagée pour toutes les parcelles
    conn = connect()
    print()

    status = load_status()
    completed_set = set(status.get("completed", []))

    success, failed = 0, []
    for i, feat in enumerate(features):
        pid = feat["properties"]["id"]
        print(f"[{i+1:02d}/{len(features)}] Parcelle {pid}")

        parcel_dir = OUT_DIR / pid
        if is_parcel_complete(parcel_dir):
            print(f"  ⏭  {pid} déjà complet (reprise), skip.")
            completed_set.add(pid)
            status["completed"] = sorted(completed_set)
            status["failed"].pop(pid, None)
            save_status(status)
            success += 1
            continue

        ok = process_parcel(feat, conn)
        if ok:
            success += 1
            completed_set.add(pid)
            status["completed"] = sorted(completed_set)
            status["failed"].pop(pid, None)
        else:
            failed.append(pid)
            status["failed"][pid] = {"last_error_at": utc_now()}

        save_status(status)

    print(f"\n{'='*50}")
    print(f"✅ Terminé : {success}/{len(features)} parcelles téléchargées")
    if failed:
        print(f"❌ Échecs  : {failed}")
