#from src.pair_dataset import PairDataset
#
#dataset = PairDataset(
#    csv_path="data/pairs.csv",
#    image_dir="data/images/train2022"
#)
#
#print("Anzahl Paare:", len(dataset))
#img1, img2, label = dataset[0]
#print("Bild 1:", img1.shape)
#print("Bild 2:", img2.shape)
#print("Label:", label)

import os
import torch
import torch.nn.functional as F
from src.pair_dataset import PairDataset
from src.siamese_network import SiameseNetwork

# Dataset laden (Pfad ggf. anpassen)
dataset = PairDataset(
    csv_path="data/pairs.csv",
    image_dir="data/images/train2022"
)

# Erstes echtes Paar laden
img1, img2, label = dataset[0]  # label: 1 = gleich, 0 = verschieden

# In Batch-Form bringen (Batchgröße 1)
img1 = img1.unsqueeze(0)
img2 = img2.unsqueeze(0)

# Modell instanziieren
model = SiameseNetwork()

# Keine Gradienten notwendig
model.eval()
with torch.no_grad():
    emb1, emb2 = model(img1, img2)

    # Cosine Similarity
    cosine_sim = F.cosine_similarity(emb1, emb2)
    cosine_dist = 1 - cosine_sim

# Ausgabe
print(f"Label (1=same, 0=different): {label}")
print(f"Cosine Similarity: {cosine_sim.item():.4f}")
print(f"Cosine Distance:   {cosine_dist.item():.4f}")

from PIL import Image
import matplotlib.pyplot as plt

# Pfade aus dem CSV holen
row = dataset.pairs.iloc[0]
img1_relpath = row["image_1"]
img2_relpath = row["image_2"]

# Komplette Pfade
img1_full = os.path.join("data/images/train2022", img1_relpath)
img2_full = os.path.join("data/images/train2022", img2_relpath)

# Bilder laden
img1_pil = Image.open(img1_full)
img2_pil = Image.open(img2_full)

# Anzeige mit Matplotlib
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(img1_pil.convert("L"), cmap="gray")
axs[0].set_title("Bild 1")
axs[0].axis("off")

axs[1].imshow(img2_pil.convert("L"), cmap="gray")
axs[1].set_title("Bild 2")
axs[1].axis("off")

plt.tight_layout()
plt.show()
