import json
from collections import defaultdict

# Pfad zur JSON-Datei
json_path = "./data/instances_train2022.json"  # ggf. anpassen

# Datei einlesen
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Überblick
print("🔍 Struktur des JSON:")
print("Schlüssel:", list(data.keys()))

# Zeige ein Beispielbild
print("\n📸 Beispiel aus 'images':")
print(json.dumps(data["images"][0], indent=2))

# Zeige ein Beispielannotation
print("\n🔳 Beispiel aus 'annotations':")
print(json.dumps(data["annotations"][0], indent=2))

# Zeige Kategorien
print("\n📋 Kategorien (Tierarten):")
print(data["categories"])

# Anzahl der eindeutigen Einzel-Zuordnungen prüfen
single_id_count = 0
multi_id_count = 0

for ann in data["annotations"]:
    ids = ann.get("individual_ids", [])
    if isinstance(ids, list):
        if len(ids) == 1:
            single_id_count += 1
        elif len(ids) > 1:
            multi_id_count += 1

print(f"\n✅ Annotationen mit GENAU EINER Individual-ID: {single_id_count}")
print(f"⚠️  Annotationen mit MEHREREN IDs: {multi_id_count}")
