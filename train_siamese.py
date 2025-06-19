import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from src.siamese_network import SiameseNetwork
from src.pair_dataset import PairDataset
from src.losses import ContrastiveLoss

# Parameter
batch_size = 8
epochs = 10
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Daten laden
dataset = PairDataset(
    csv_path="data/pairs.csv",
    image_dir="data/images/train2022"
)
from torch.utils.data import Subset
subset = Subset(dataset, list(range(1000)))
loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
#loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Modell & Loss
model = SiameseNetwork().to(device)
loss_fn = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
model.train()
for epoch in range(epochs):
    total_loss = 0.0

    for batch in loader:
        img1, img2, label = batch
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        # Vorwärtslauf
        emb1, emb2 = model(img1, img2)
        loss = loss_fn(emb1, emb2, label)

        # Rückwärtslauf
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs}  Durchschnittlicher Loss: {avg_loss:.4f}")

# Modell speichern
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d-%H%M")
torch.save(model.state_dict(), f"models/siamese_{timestamp}.pt")
print(f"Training abgeschlossen: Modell gespeichert unter models/siamese_{timestamp}.pt")
