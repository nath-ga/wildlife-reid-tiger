import os

# Liste der Ordner, die erstellt werden sollen
folders = [
    "data/raw",
    "data/processed",
    "models",
    "outputs/logs",
    "outputs/similarity_examples",
    "src"
]

# Erzeuge jeden Ordner (falls nicht vorhanden) und lege eine .keep-Datei hinein
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    # Leerer Platzhalter für Git – .keep-Dateien bleiben bei leeren Ordnern erhalten
    keep_path = os.path.join(folder, ".keep")
    with open(keep_path, "w") as f:
        f.write("")
        
print("Projektstruktur wurde erfolgreich erstellt.")
