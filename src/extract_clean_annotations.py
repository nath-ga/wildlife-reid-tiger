import json
import csv

# Pfad zur COCO-JSON-Datei (ggf. anpassen)
json_path = "./data/instances_train2022.json"
output_csv = "./data/annotations_clean.csv"

# JSON laden
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Bild-Index vorbereiten
image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

# Ergebnisliste
rows = []

for ann in data["annotations"]:
    ids = ann.get("individual_ids", [])
    if isinstance(ids, list) and len(ids) == 1:
        image_id = ann["image_id"]
        file_name = image_id_to_filename.get(image_id)
        bbox = ann["bbox"]
        individual_id = ids[0]
        if file_name:  # Sicherheit: Datei muss existieren
            rows.append([file_name, bbox, individual_id])

# CSV schreiben
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "bbox", "individual_id"])
    writer.writerows(rows)

print(f"✅ Fertig: {len(rows)} Einträge in {output_csv}")
