from src.leopard_dataset import LeopardDataset

dataset = LeopardDataset(
    csv_path="./data/annotations_clean.csv",
    image_dir="./data/images/train2022",  # anpassen!
    image_size=(100, 100)
)

print(f"Datensätze gefunden: {len(dataset)}")
img, label = dataset[0]
print(f"Bildgröße: {img.shape}, Label: {label}")
