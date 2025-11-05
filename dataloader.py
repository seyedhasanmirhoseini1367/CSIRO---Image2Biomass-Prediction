import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class BiomassDataset(Dataset):
    def __init__(self, base_path, targets, img_size=(256, 256)):  # or (512, 1024) etc.
        super().__init__()
        self.targets = targets
        self.base_path = base_path
        self.img_size = img_size

        # Define transform with resizing
        self.transform = transforms.Compose([
            transforms.Resize(img_size),  # Resize to smaller dimensions
            transforms.ToTensor()  # converts to [0, 1] float32 automatically
        ])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_id = self.targets['image_path'][idx]
        target = self.targets['target'][idx]

        img_path = os.path.join(self.base_path, img_id)
        img = Image.open(img_path)

        # Apply transform (includes resize + ToTensor)
        img_tensor = self.transform(img)
        target = torch.tensor(target, dtype=torch.float32)

        return img_tensor, target

'''
Targets:
Dry_Clover_g
Dry_Dead_g
Dry_Green_g
Dry_Total_g
GDM_g

train_transform = transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),
        # torchvision.transforms.RandomHorizontalFlip(p=0.8),
        # torchvision.transforms.RandomVerticalFlip(p=0.8),       
        # RandomApply([torchvision.transforms.RandomAutocontrast()], p=0.4), 
        # RandomApply([torchvision.transforms.RandomRotation(degrees=15)], p=0.6), 
        # torchvision.transforms.RandomInvert(p=0.1),
        # RandomApply([torchvision.transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.4),
        # Color jitter for brightness, contrast, saturation, and hue
        # RandomApply([torchvision.transforms.ColorJitter(brightness=0, contrast=0.2, saturation=0, hue=0)], p=0.5),    
        # Random perspective transformation
        # torchvision.transforms.RandomPerspective(distortion_scale=0.1, p=0.2, interpolation=3),        
        # Random affine transformation
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # Randomly erase a portion of the image
        # transforms.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        # transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
        
'''

train_path = "D:\datasets\csiro-biomass/train.csv"
test_path = "D:\datasets\csiro-biomass/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

from sklearn.model_selection import train_test_split

# Group by image_path to aggregate multiple targets for the same image
df = train_df.groupby('image_path')['target'].agg(list).reset_index()

# Random split (e.g., 80% train, 20% validation)
train, val = train_test_split(
    df,
    test_size=0.2,
    random_state=42,  # ensures reproducibility
    shuffle=True
)

base_path = "D:\datasets\csiro-biomass"

train_dataset = BiomassDataset(base_path, train.reset_index())
val_dataset = BiomassDataset(base_path, val.reset_index())

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=True)

# for data, targets in train_dataloader:
#     print(data.shape)
#     print(targets)
#     break
