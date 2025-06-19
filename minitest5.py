# anzeige bestimmter ergbnispaare

import torch
import torch.nn.functional as F
from src.pair_dataset import PairDataset
from src.siamese_network import SiameseNetwork
from PIL import Image
import matplotlib.pyplot as plt
import os

# Dataset und Modell laden
dataset = PairDataset(
    csv_path="data/pairs.csv",
    image_dir="data/images/train2022"
)
model = SiameseNetwork()
# Modell laden
model_path = "models/siamese_20250617-1348.pt"
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

def zeige_paar_by_index(idx):
    img1, img2, label = dataset[idx]

    img1b = img1.unsqueeze(0)
    img2b = img2.unsqueeze(0)

    with torch.no_grad():
        emb1, emb2 = model(img1b, img2b)
        cos_sim = F.cosine_similarity(emb1, emb2).item()
        cos_dist = 1 - cos_sim

    row = dataset.pairs.iloc[idx]
    img1_path = os.path.join("data/images/train2022", row["image_1"])
    img2_path = os.path.join("data/images/train2022", row["image_2"])
    img1_pil = Image.open(img1_path)
    img2_pil = Image.open(img2_path)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img1_pil.convert("L"), cmap="gray")
    axs[0].set_title("Bild 1")
    axs[0].axis("off")

    axs[1].imshow(img2_pil.convert("L"), cmap="gray")
    axs[1].set_title("Bild 2")
    axs[1].axis("off")

    fig.suptitle(
        f"Index: {idx}   Label: {int(label)}   Cosine Distance: {cos_dist:.4f}",
        fontsize=12
    )
    plt.tight_layout()
    plt.show()

zeige_paar_by_index(7)
zeige_paar_by_index(14)
zeige_paar_by_index(20)
zeige_paar_by_index(100)

