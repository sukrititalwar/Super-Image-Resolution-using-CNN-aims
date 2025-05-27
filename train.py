import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import SRDataset  # <-- Import your dataset class
from tqdm import tqdm

# Simple SRCNN model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.model(x)

# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
num_epochs = 100
learning_rate = 1e-4

# Datasets and loaders
train_dataset = SRDataset("data/train/LR", "data/train/HR", augment=True)
val_dataset = SRDataset("data/val/LR", "data/val/HR", augment=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Model, loss, optimizer
model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for lr_imgs, hr_imgs in pbar:
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

        sr_imgs = model(lr_imgs)
        loss = criterion(sr_imgs, hr_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch}/{num_epochs}] - Avg Train Loss: {avg_train_loss:.4f}")

    # Save model every 10 epochs
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"checkpoints/srcnn_epoch{epoch}.pth")

    # Save validation SR images
    model.eval()
    with torch.no_grad():
        for i, (lr, hr) in enumerate(val_loader):
            lr = lr.to(device)
            sr = model(lr)
            save_image(sr, f"results/epoch{epoch}_img{i}_SR.png")
            save_image(hr, f"results/epoch{epoch}_img{i}_HR.png")
            if i == 2:  # Save only first 3 images
                break
