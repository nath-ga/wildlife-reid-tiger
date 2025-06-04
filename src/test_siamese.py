import torch
import torch.nn.functional as F
from siamese_network import SiameseNetwork

# Modell instanziieren
model = SiameseNetwork()

# Beispiel: zwei Dummy-Bilder (Batchgröße 1, 1 Kanal, 100x100 Pixel)
img1 = torch.rand(1, 1, 100, 100)
img2 = torch.rand(1, 1, 100, 100)

# Modell anwenden
out1, out2 = model(img1, img2)

# Ausgabeform überprüfen
print("✅ Embedding 1:", out1.shape)
print("✅ Embedding 2:", out2.shape)

# Ähnlichkeitsmessung (kleiner Wert = ähnliche Embeddings)
distance = F.pairwise_distance(out1, out2)
print("📏 Abstand zwischen Embeddings:", distance.item())
