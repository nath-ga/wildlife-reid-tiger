import pandas as pd
import itertools
import random
import csv

# CSV einlesen
df = pd.read_csv("./data/annotations_clean.csv")

# Gruppieren nach individueller ID
groups = df.groupby("individual_id")

positive_pairs = []

# Positive Paare erzeugen
for individual_id, group in groups:
    images = group["file_name"].tolist()
    if len(images) >= 2:
        for pair in itertools.combinations(images, 2):
            positive_pairs.append((pair[0], pair[1], 1))

print(f"✅ Positive Paare: {len(positive_pairs)}")

# Negative Paare erzeugen
all_images = df[["file_name", "individual_id"]].to_dict(orient="records")
negative_pairs = set()

while len(negative_pairs) < len(positive_pairs):
    img1, img2 = random.sample(all_images, 2)
    if img1["individual_id"] != img2["individual_id"]:
        pair = (img1["file_name"], img2["file_name"], 0)
        negative_pairs.add(pair)

print(f"✅ Negative Paare: {len(negative_pairs)}")

# Alles zusammenführen und speichern
all_pairs = positive_pairs + list(negative_pairs)
random.shuffle(all_pairs)

with open("./data/pairs.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_1", "image_2", "label"])
    writer.writerows(all_pairs)

print("📄 Datei gespeichert: data/pairs.csv")
