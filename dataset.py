import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, augment=True):
        self.lr_images = sorted([os.path.join(lr_dir, img) for img in os.listdir(lr_dir)])
        self.hr_images = sorted([os.path.join(hr_dir, img) for img in os.listdir(hr_dir)])
        self.augment = augment

        # Resize transforms
        self.lr_resize = transforms.Resize((80, 80), interpolation=Image.BICUBIC)
        self.hr_resize = transforms.Resize((320, 320), interpolation=Image.BICUBIC)

        # Common to-tensor conversion
        self.to_tensor = transforms.ToTensor()

        # Augmentations applied to both LR and HR images
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),  # rotate within ±15 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr = Image.open(self.lr_images[idx]).convert('RGB')
        hr = Image.open(self.hr_images[idx]).convert('RGB')

        # Resize both images to fixed dimensions
        lr = self.lr_resize(lr)
        hr = self.hr_resize(hr)

        # Apply the same augmentation to both images
        if self.augment:
            seed = random.randint(0, 99999)
            random.seed(seed)
            lr = self.augmentations(lr)
            random.seed(seed)
            hr = self.augmentations(hr)

        return self.to_tensor(lr), self.to_tensor(hr)
