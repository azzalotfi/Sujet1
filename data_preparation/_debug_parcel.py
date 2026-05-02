import json, requests

with open("data_preparation/output/train.geojson", encoding="utf-8") as f:
    fc = json.load(f)

feat = fc["features"][0]
props = feat["properties"]
geom = feat["geometry"]
coords = geom["coordinates"][0]
lngs = [c[0] for c in coords]
lats = [c[1] for c in coords]
bbox = [min(lngs)-0.01, min(lats)-0.01, max(lngs)+0.01, max(lats)+0.01]
cx = (bbox[0]+bbox[2])/2
cy = (bbox[1]+bbox[3])/2
print("Parcelle:", props["id"], props["gouvernorat"])
print("BBox:", bbox)
print("Centroide:", cx, cy)

payload = {
    "collections": ["sentinel-2-l2a"],
    "intersects": {"type": "Point", "coordinates": [cx, cy]},
    "datetime": "2025-05-01T00:00:00Z/2025-06-30T23:59:59Z",
    "limit": 3
}
r = requests.post("https://earth-search.aws.element84.com/v1/search", json=payload, timeout=20)
items = r.json().get("features", [])
print("Status:", r.status_code, "| images:", len(items))
for it in items:
    p = it["properties"]
    iid = it["id"]
    dt = p["datetime"][:10]
    cc = p.get("eo:cloud_cover", "?")
    print(f"  {iid} | {dt} | cc={cc}%")
    print(f"  assets: {list(it.get('assets',{}).keys())[:8]}")
