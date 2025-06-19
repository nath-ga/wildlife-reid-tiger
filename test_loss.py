import torch
from src.losses import ContrastiveLoss

loss_fn = ContrastiveLoss(margin=1.0)

# Beispiel 1: gleiches Tier, Embeddings fast identisch
out1 = torch.tensor([[0.1, 0.2, 0.3]])
out2 = torch.tensor([[0.1, 0.2, 0.31]])
label_same = torch.tensor([1.0])

loss_same = loss_fn(out1, out2, label_same)
print(f"✅ Loss bei gleichem Tier (soll klein sein): {loss_same.item():.4f}")

# Beispiel 2: verschiedene Tiere, Embeddings weit auseinander
out3 = torch.tensor([[0.1, 0.2, 0.3]])
out4 = torch.tensor([[0.9, 0.8, 0.7]])
label_diff = torch.tensor([0.0])

loss_diff = loss_fn(out3, out4, label_diff)
print(f"✅ Loss bei verschiedenem Tier (soll klein sein): {loss_diff.item():.4f}")

# Beispiel 3: Fehlerfall – verschieden, aber Embeddings fast gleich
label_mistake = torch.tensor([0.0])
out5 = torch.tensor([[0.1, 0.2, 0.3]])
out6 = torch.tensor([[0.1, 0.2, 0.29]])

loss_mistake = loss_fn(out5, out6, label_mistake)
print(f"⚠️  Loss bei zu ähnlichen Embeddings trotz Label 0: {loss_mistake.item():.4f}")
