import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SRDataset
from model import DCSCN  # Replace with EDSR if needed
import os

# Dataset paths (update if needed)
lr_dir = r"/Users/nikhiltalwar/Downloads/dcscn_super_resolution_final/data/train/LR"
hr_dir = r"/Users/nikhiltalwar/Downloads/dcscn_super_resolution_final/data/train/HR"

# Prepare dataset
dataset = SRDataset(lr_dir, hr_dir)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model, loss, optimizer
model = DCSCN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Learning rate scheduler (decays LR every 50 epochs)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Train
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for lr, hr in dataloader:
        sr = model(lr)
        loss = criterion(sr, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()  # Step the learning rate scheduler

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# Save model
os.makedirs("../results", exist_ok=True)
torch.save(model.state_dict(), "../results/dcscn.pth")
