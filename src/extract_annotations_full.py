import json
import csv
from pathlib import Path

# Pfade
json_path = Path("data/instances_train2022.json")
output_path = Path("data/annotations_clean.csv")

# JSON laden
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

annotations = data["annotations"]
images = {img["id"]: img["file_name"] for img in data["images"]}

# Ergebnisse sammeln
rows = []
for ann in annotations:
    ids = ann.get("individual_ids", [])
    if ids:  # nur wenn mindestens eine ID da ist
        file_name = images.get(ann["image_id"])
        if file_name:
            rows.append({
                "file_name": file_name,
                "bbox": ann["bbox"],
                "individual_id": ids[0]  # erste ID nehmen
            })

# Speichern
with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file_name", "bbox", "individual_id"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Neue annotations_clean.csv mit {len(rows)} Einträgen gespeichert.")
