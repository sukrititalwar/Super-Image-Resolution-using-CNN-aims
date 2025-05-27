import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
import numpy as np
from PIL import Image
import os
import csv
from model import DCSCN  # Change to EDSR if replacing model

# Load trained model
model = DCSCN()
model.load_state_dict(torch.load("../results/dcscn.pth"))
model.eval()

# Dataset paths
lr_dir = r"/Users/nikhiltalwar/Downloads/dcscn_super_resolution_final/data/train/LR"
hr_dir = r"/Users/nikhiltalwar/Downloads/dcscn_super_resolution_final/data/train/HR"

# Create output folder
os.makedirs("results/images", exist_ok=True)

psnr_scores, ssim_scores = [], []

# For logging results
results = []

for img_name in os.listdir(lr_dir):
    # Load images
    lr_img = Image.open(os.path.join(lr_dir, img_name)).convert('RGB')
    hr_img = Image.open(os.path.join(hr_dir, img_name)).convert('RGB')

    # Convert to tensor and predict
    lr_tensor = TF.to_tensor(lr_img).unsqueeze(0)
    sr_tensor = model(lr_tensor).squeeze().detach().numpy().transpose(1, 2, 0)

    # Normalize HR image
    hr_np = np.array(hr_img).astype(np.float32) / 255.0

    # Resize SR image to match HR image
    sr_resized = resize(sr_tensor, hr_np.shape, anti_aliasing=True)

    # Compute PSNR and SSIM
    psnr = peak_signal_noise_ratio(hr_np, sr_resized, data_range=1.0)
    ssim = structural_similarity(hr_np, sr_resized, channel_axis=-1, data_range=1.0)

    psnr_scores.append(psnr)
    ssim_scores.append(ssim)
    results.append([img_name, psnr, ssim])

    # Save SR and HR images
    sr_save = torch.tensor(sr_resized.transpose(2, 0, 1))
    hr_save = torch.tensor(hr_np.transpose(2, 0, 1))

    save_image(sr_save, f"results/images/{img_name}_SR.png")
    save_image(hr_save, f"results/images/{img_name}_HR.png")

# Log to CSV
with open("results/evaluation_scores.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "PSNR", "SSIM"])
    writer.writerows(results)

# Print averages
print(f"Average PSNR: {np.mean(psnr_scores):.2f}")
print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
