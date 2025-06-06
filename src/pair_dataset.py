from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import torch
import os

class PairDataset(Dataset):
    def __init__(self, csv_path, image_dir, image_size=100):
        self.pairs = pd.read_csv(csv_path)
        self.image_dir = image_dir

        # Transformation: Bild öffnen → Größe anpassen → in Tensor umwandeln
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]

        # Dateinamen aus CSV lesen
        img1_path = os.path.join(self.image_dir, row["image_1"])
        img2_path = os.path.join(self.image_dir, row["image_2"])

        # Bilder laden und transformieren
        img1 = self.transform(Image.open(img1_path).convert("L"))
        img2 = self.transform(Image.open(img2_path).convert("L"))


        label = torch.tensor(row["label"], dtype=torch.float32)

        return img1, img2, label
