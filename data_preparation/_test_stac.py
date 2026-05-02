import requests

payload = {
    "collections": ["sentinel-2-l2a"],
    "intersects": {"type": "Point", "coordinates": [10.0, 36.4]},
    "datetime": "2025-05-01T00:00:00Z/2025-06-30T23:59:59Z",
    "limit": 3
}
r = requests.post("https://earth-search.aws.element84.com/v1/search", json=payload, timeout=20)
print("Status:", r.status_code)
items = r.json().get("features", [])
print("Images:", len(items))
for item in items:
    p = item["properties"]
    iid = item["id"]
    dt = p["datetime"][:10]
    cc = p.get("eo:cloud_cover", "?")
    print(f"  {iid} | {dt} | nuages={cc}%")
    assets = item.get("assets", {})
    print(f"  bandes: {list(assets.keys())[:12]}")
