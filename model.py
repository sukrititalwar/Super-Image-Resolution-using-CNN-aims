import torch
import torch.nn as nn
import torch.nn.functional as F

class DCSCN(nn.Module):
    def __init__(self, channels=3):
        super(DCSCN, self).__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(channels, 96, 3, padding=1), nn.ReLU(),
            nn.Conv2d(96, 76, 3, padding=1), nn.ReLU(),
            nn.Conv2d(76, 65, 3, padding=1), nn.ReLU(),
            nn.Conv2d(65, 55, 3, padding=1), nn.ReLU(),
            nn.Conv2d(55, 47, 3, padding=1), nn.ReLU(),
            nn.Conv2d(47, 39, 3, padding=1), nn.ReLU()
        )

        self.reconstruction = nn.Sequential(
            nn.Conv2d(39, 32, 1), nn.ReLU(),
            nn.Conv2d(32, 3 * 16, 1),
            nn.PixelShuffle(4)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.reconstruction(x)
        return x
