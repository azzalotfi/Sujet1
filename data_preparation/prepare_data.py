"""
Étape 1 — Préparation des données oliveraies
- Convertit les JSON EZZAYRA en GeoJSON standard
- Infère le gouvernorat par reverse geocoding (Nominatim/OpenStreetMap)
- Effectue un split train/val/test stratifié par gouvernorat (70/15/15)
- Sauvegarde les fichiers résultants dans data_preparation/output/
"""

import json
import time
import random
import math
from pathlib import Path
from collections import defaultdict

import requests

# ─────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "Oliviers"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

INTENSIF_FILE  = DATA_DIR / "parcellesOliviersIntensifs.json"
EXTENSIF_FILE  = DATA_DIR / "parcelles_OlivierExtensif.json"

SEED = 42
random.seed(SEED)

# ─────────────────────────────────────────────
# 1. Chargement + conversion en GeoJSON Feature
# ─────────────────────────────────────────────
def coords_to_geojson_polygon(coordinates: list) -> dict:
    """Convertit une liste [{lat, lng}] en GeoJSON Polygon (ordre lng, lat)."""
    ring = [[c["lng"], c["lat"]] for c in coordinates]
    # Fermer l'anneau si nécessaire
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def load_file(path: Path, systeme: str) -> list:
    """Charge un fichier JSON et retourne une liste de GeoJSON Features."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    features = []
    for parcel in data["parcels"]:
        geom = coords_to_geojson_polygon(parcel["coordinates"])
        feature = {
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "id": parcel["id"],
                "name": parcel["name"],
                "crop": parcel["crop"],
                "systeme": systeme,
                "area_ha": parcel["area_ha"],
                "created_at": parcel["created_at"],
                "gouvernorat": None,   # sera rempli à l'étape 2
                "split": None,         # sera rempli à l'étape 3
            },
        }
        features.append(feature)
    return features


print("📂 Chargement des fichiers source...")
features_intensif = load_file(INTENSIF_FILE, "intensif")
features_extensif = load_file(EXTENSIF_FILE, "extensif")

# Pas de données hyper-intensif dans le pack → on le signale clairement
print(f"  ✅ Intensif   : {len(features_intensif)} parcelles")
print(f"  ✅ Extensif   : {len(features_extensif)} parcelles")
print("  ⚠️  Hyper-intensif : ABSENT du pack — à demander aux organisateurs")

all_features = features_intensif + features_extensif

# ─────────────────────────────────────────────
# 2. Reverse geocoding → gouvernorat
# ─────────────────────────────────────────────
def centroid(geometry: dict) -> tuple:
    """Calcule le centroïde approximatif d'un Polygon GeoJSON."""
    ring = geometry["coordinates"][0]
    lngs = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return sum(lngs) / len(lngs), sum(lats) / len(lats)


def get_gouvernorat(lng: float, lat: float, retries: int = 3) -> str:
    """Appelle Nominatim OSM pour obtenir le gouvernorat (county/state)."""
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lng,
        "format": "json",
        "zoom": 6,          # niveau gouvernorat
        "addressdetails": 1,
    }
    headers = {"User-Agent": "OliveraiesTunisie/1.0"}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            r.raise_for_status()
            addr = r.json().get("address", {})
            # Nominatim renvoie le gouvernorat dans 'state' ou 'county'
            gov = (
                addr.get("state")
                or addr.get("county")
                or addr.get("region")
                or "Inconnu"
            )
            return gov
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"    ⚠️  Reverse geocoding échoué ({lng:.4f},{lat:.4f}): {e}")
                return "Inconnu"
    return "Inconnu"


print("\n🌍 Reverse geocoding des gouvernorats (Nominatim OSM)...")
print("   (1 requête/s pour respecter la politique d'usage Nominatim)")

for i, feat in enumerate(all_features):
    lng, lat = centroid(feat["geometry"])
    gov = get_gouvernorat(lng, lat)
    feat["properties"]["gouvernorat"] = gov
    print(f"  [{i+1:02d}/{len(all_features)}] {feat['properties']['systeme']:10s} | "
          f"{lat:.4f},{lng:.4f} → {gov}")
    time.sleep(1.1)  # respecter 1 req/s Nominatim

# ─────────────────────────────────────────────
# 3. Split train/val/test stratifié par gouvernorat
# ─────────────────────────────────────────────
def spatial_split(features: list, train_ratio=0.70, val_ratio=0.15, seed=42) -> list:
    """
    Split spatial : regroupe par gouvernorat, puis assigne chaque gouvernorat
    entièrement à train, val ou test pour éviter la fuite de données spatiale.
    Ratio cible : 70/15/15.
    Les gouvernorats trop gros (>40% du total) sont divisés en sous-groupes
    pour garantir que val et test ont au moins quelques parcelles.
    """
    random.seed(seed)

    # Grouper les indices par gouvernorat
    gov_to_indices = defaultdict(list)
    for i, feat in enumerate(features):
        gov = feat["properties"]["gouvernorat"]
        gov_to_indices[gov].append(i)

    total = len(features)

    # Éclater les gouvernorats trop grands en sous-groupes de ~10 parcelles max
    # pour éviter qu'un seul gouvernorat monopolise tout le dataset
    expanded_groups = []
    for gov, indices in gov_to_indices.items():
        if len(indices) > max(4, int(total * 0.3)):
            chunk_size = max(3, len(indices) // 3)
            for i in range(0, len(indices), chunk_size):
                expanded_groups.append((f"{gov}_{i}", indices[i:i+chunk_size]))
        else:
            expanded_groups.append((gov, indices))

    random.shuffle(expanded_groups)

    train_target = total * train_ratio
    val_target   = total * val_ratio

    train_groups, val_groups, test_groups = [], [], []
    train_count, val_count = 0, 0

    for name, indices in expanded_groups:
        count = len(indices)
        if train_count < train_target:
            train_groups.append((name, indices))
            train_count += count
        elif val_count < val_target:
            val_groups.append((name, indices))
            val_count += count
        else:
            test_groups.append((name, indices))

    # Construire les sets d'indices
    train_idx = {i for _, idxs in train_groups for i in idxs}
    val_idx   = {i for _, idxs in val_groups   for i in idxs}

    # Assigner le split à chaque feature
    for i, feat in enumerate(features):
        if i in train_idx:
            feat["properties"]["split"] = "train"
        elif i in val_idx:
            feat["properties"]["split"] = "val"
        else:
            feat["properties"]["split"] = "test"

    train_govs = [n for n, _ in train_groups]
    val_govs   = [n for n, _ in val_groups]
    test_govs  = [n for n, _ in test_groups]
    return train_govs, val_govs, test_govs


print("\n✂️  Split train/val/test stratifié par gouvernorat...")
train_govs, val_govs, test_govs = spatial_split(all_features)

# Compter
splits = defaultdict(lambda: defaultdict(int))
for feat in all_features:
    s = feat["properties"]["split"]
    sys = feat["properties"]["systeme"]
    splits[s][sys] += 1
    splits[s]["total"] += 1

print(f"\n  Split résultant :")
for s in ["train", "val", "test"]:
    d = splits[s]
    pct = d["total"] / len(all_features) * 100
    print(f"  {s:5s}: {d['total']:2d} parcelles ({pct:.0f}%) "
          f"— intensif={d.get('intensif',0)}, extensif={d.get('extensif',0)}")

print(f"\n  Gouvernorats train : {train_govs}")
print(f"  Gouvernorats val   : {val_govs}")
print(f"  Gouvernorats test  : {test_govs}")

# ─────────────────────────────────────────────
# 4. Sauvegarde des fichiers GeoJSON
# ─────────────────────────────────────────────
def save_geojson(features: list, path: Path):
    fc = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    print(f"  💾 Sauvegardé : {path.name} ({len(features)} features)")


print("\n💾 Sauvegarde des fichiers GeoJSON...")

# Fichier complet
save_geojson(all_features, OUT_DIR / "all_parcelles.geojson")

# Par split
for split_name in ["train", "val", "test"]:
    subset = [f for f in all_features if f["properties"]["split"] == split_name]
    save_geojson(subset, OUT_DIR / f"{split_name}.geojson")

# Par système
for sys_name in ["intensif", "extensif"]:
    subset = [f for f in all_features if f["properties"]["systeme"] == sys_name]
    save_geojson(subset, OUT_DIR / f"{sys_name}.geojson")

print("\n✅ Étape 1 terminée. Fichiers dans data_preparation/output/")
print("   ⚠️  Rappel : classe hyper-intensif manquante — pipeline limité à 2 classes pour l'instant.")
