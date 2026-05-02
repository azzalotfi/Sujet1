"""
Test rapide du pipeline openEO sur la 1ère parcelle uniquement.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from download_sentinel2 import process_parcel, connect, OUT_DIR

with open(Path(__file__).parent / "output" / "train.geojson", encoding="utf-8") as f:
    fc = json.load(f)

# Prendre la parcelle la plus petite pour un test rapide
feat = min(fc["features"], key=lambda f: f["properties"]["area_ha"])
print(f"Test sur : {feat['properties']['id']} ({feat['properties']['systeme']}, {feat['properties']['area_ha']:.1f} ha)")
print()

conn = connect()
ok = process_parcel(feat, conn)
print("\nRésultat:", "✅ OK" if ok else "❌ ECHEC")
