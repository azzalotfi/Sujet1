"""
Reapplique le split spatial corrigé sur all_parcelles.geojson déjà généré
(évite de relancer le reverse geocoding de 49 requêtes).
"""

import json
import random
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path(__file__).parent / "output"

with open(OUT_DIR / "all_parcelles.geojson", encoding="utf-8") as f:
    fc = json.load(f)

features = fc["features"]
total = len(features)

random.seed(42)

# Grouper par gouvernorat
gov_to_indices = defaultdict(list)
for i, feat in enumerate(features):
    gov = feat["properties"]["gouvernorat"]
    gov_to_indices[gov].append(i)

# Éclater les gouvernorats trop grands (>30% du total) en sous-groupes
train_ratio, val_ratio = 0.70, 0.15
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

train_idx = {i for _, idxs in train_groups for i in idxs}
val_idx   = {i for _, idxs in val_groups   for i in idxs}

for i, feat in enumerate(features):
    if i in train_idx:
        feat["properties"]["split"] = "train"
    elif i in val_idx:
        feat["properties"]["split"] = "val"
    else:
        feat["properties"]["split"] = "test"

# Compter
splits = defaultdict(lambda: defaultdict(int))
for feat in features:
    s = feat["properties"]["split"]
    sys = feat["properties"]["systeme"]
    splits[s][sys] += 1
    splits[s]["total"] += 1

print("Split résultant (corrigé) :")
for s in ["train", "val", "test"]:
    d = splits[s]
    pct = d["total"] / total * 100
    print(f"  {s:5s}: {d['total']:2d} parcelles ({pct:.0f}%) "
          f"— intensif={d.get('intensif',0)}, extensif={d.get('extensif',0)}")

print(f"\nGroupes train : {[n for n,_ in train_groups]}")
print(f"Groupes val   : {[n for n,_ in val_groups]}")
print(f"Groupes test  : {[n for n,_ in test_groups]}")

# Sauvegarder
def save(feats, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f,
                  ensure_ascii=False, indent=2)
    print(f"  💾 {path.name} ({len(feats)} features)")

print("\nSauvegarde...")
save(features, OUT_DIR / "all_parcelles.geojson")
for split_name in ["train", "val", "test"]:
    subset = [f for f in features if f["properties"]["split"] == split_name]
    save(subset, OUT_DIR / f"{split_name}.geojson")

print("\n✅ Split corrigé et sauvegardé.")
