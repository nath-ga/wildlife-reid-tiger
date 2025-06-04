import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LeopardDataset(Dataset):
    def __init__(self, csv_path, image_dir, image_size=(100, 100)):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row["file_name"])
        image = Image.open(image_path).convert("RGB")

        # Bounding Box: [x, y, w, h]
        bbox = eval(row["bbox"]) if isinstance(row["bbox"], str) else row["bbox"]
        x, y, w, h = map(int, bbox)
        cropped = image.crop((x, y, x + w, y + h))

        # Transformieren
        img_tensor = self.transform(cropped)
        label = int(row["individual_id"])
        return img_tensor, label
