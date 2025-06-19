import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Gemeinsames CNN für beide Bilder (geteilte Gewichte)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Voll verbundenes Embedding
        self.fc = nn.Sequential(
            nn.Linear(32 * 22 * 22, 128),
            nn.ReLU(),
            nn.Linear(128, 32)  # Embedding mit 32 Dimensionen
        )

    def forward_once(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)  # Normiert auf Länge 1 (gut für Cosine)

    def forward(self, img1, img2):
        output1 = self.forward_once(img1)
        output2 = self.forward_once(img2)
        return output1, output2
