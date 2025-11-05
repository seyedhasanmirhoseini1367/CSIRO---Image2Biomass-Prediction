import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import time

from csiro_biomass.u_net import AttentionUNet
# from csiro_biomass.cnn import CNNModel

import warnings

warnings.filterwarnings('ignore')

# ========================================

import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from PIL import Image

from torchvision import transforms

from torchvision import transforms


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

# ========================================
# HYPERPARAMETERS
# ========================================
learning_rate = 0.0001  # Initial learning rate for Adam optimizer
epochs = 200  # Total number of training epochs
weight_decay = 1e-3  # L2 regularization factor to prevent overfitting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# ========================================
# MODEL INITIALIZATION
# ========================================
# Initialize the Attention U-Net model
# - in_channels=3: RGB images
# - num_outputs=5: Regression with 5 target values
# - base_features=64: Starting number of feature channels
# - bilinear=True: Use bilinear upsampling (memory efficient)
model = AttentionUNet(in_channels=3, num_outputs=5, base_features=64, bilinear=True).to(device)
# model = CNNModel(in_channels=3, out_channels=(16, 32, 64), hidden_fc=128, target_size=5)

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ========================================
# LOSS FUNCTION AND OPTIMIZER
# ========================================
# MSE Loss is standard for regression tasks
# It penalizes predictions proportional to squared error: Loss = (pred - target)^2
criterion = nn.MSELoss()

# Adam optimizer: adaptive learning rate with momentum
# - lr: learning rate controls step size during gradient descent
# - weight_decay: adds L2 penalty (lambda * ||weights||^2) to prevent overfitting
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# ========================================
# LEARNING RATE SCHEDULER
# ========================================
# ReduceLROnPlateau: reduces learning rate when validation loss plateaus
# This helps the model converge better by taking smaller steps when improvement slows
# - mode='min': we want to minimize val_loss
# - factor=0.5: multiply LR by 0.5 when triggered
# - patience=5: wait 5 epochs without improvement before reducing LR
# - verbose=True: print message when LR is reduced
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
)

# ========================================
# EARLY STOPPING SETUP
# ========================================
# Early stopping prevents overfitting by stopping training when validation loss
# stops improving for a specified number of epochs (patience)
early_stop_patience = 15  # Stop if no improvement for 15 epochs
early_stop_counter = 0  # Tracks epochs without improvement
best_val_loss = float("inf")  # Initialize with infinity (any loss will be better)
best_model_state = None  # Will store the best model weights

# ========================================
# TRAINING HISTORY TRACKING
# ========================================
# Store losses for each epoch to analyze training progress and plot curves
history = {
    'train_loss': [],  # Average training loss per epoch
    'val_loss': [],  # Average validation loss per epoch
    'learning_rates': []  # Learning rate at each epoch
}


# ========================================
# TRAINING FUNCTION
# ========================================
def train_one_epoch(model, trainloader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Args:
        model: Neural network model
        trainloader: DataLoader for training data
        optimizer: Optimizer for updating weights
        criterion: Loss function
        device: CPU or CUDA device

    Returns:
        avg_train_loss: Average loss across all training batches
    """
    model.train()  # Set model to training mode (enables dropout, batchnorm updates)
    train_loss_sum = 0.0
    n_train_batches = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # Move data to GPU/CPU
        # dtype=torch.float32 ensures consistent precision
        inputs = inputs.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)

        # Zero gradients from previous iteration
        # PyTorch accumulates gradients by default, so we must clear them
        optimizer.zero_grad()

        # Forward pass: compute predictions
        preds = model(inputs)  # Shape: [batch_size, 5]

        # Compute loss: Mean Squared Error between predictions and targets
        loss = criterion(preds, targets)

        # Backward pass: compute gradients of loss w.r.t. model parameters
        # This uses automatic differentiation (backpropagation)
        loss.backward()

        # Update model parameters using computed gradients
        # Adam adjusts learning rate per parameter based on gradient history
        optimizer.step()

        # Accumulate loss for averaging
        # .item() extracts scalar value from tensor (doesn't keep computation graph)
        train_loss_sum += loss.item()
        n_train_batches += 1

    # Compute average loss across all batches
    avg_train_loss = train_loss_sum / max(1, n_train_batches)
    return avg_train_loss


# ========================================
# VALIDATION FUNCTION
# ========================================
def validate(model, valloader, criterion, device):
    """
    Evaluate the model on validation data.

    Validation helps monitor overfitting: if train loss decreases but val loss
    increases, the model is memorizing training data rather than learning patterns.

    Args:
        model: Neural network model
        valloader: DataLoader for validation data
        criterion: Loss function
        device: CPU or CUDA device

    Returns:
        avg_val_loss: Average loss across all validation batches
    """
    model.eval()  # Set model to evaluation mode (disables dropout, freezes batchnorm)
    val_loss_sum = 0.0
    n_val_batches = 0

    # Disable gradient computation for validation (saves memory and speeds up)
    # We don't need gradients since we're not updating weights
    with torch.no_grad():
        for v_inputs, v_targets in valloader:
            # Move data to device
            v_inputs = v_inputs.to(device, dtype=torch.float32)
            v_targets = v_targets.to(device, dtype=torch.float32)

            # Forward pass only (no backward pass needed)
            v_preds = model(v_inputs)

            # Compute loss
            v_loss = criterion(v_preds, v_targets)

            # Accumulate validation loss
            val_loss_sum += v_loss.item()
            n_val_batches += 1

    # Compute average validation loss
    avg_val_loss = val_loss_sum / max(1, n_val_batches)
    return avg_val_loss


# ========================================
# MAIN TRAINING LOOP
# ========================================
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60 + "\n")

for epoch in range(1, epochs + 1):
    epoch_start = time.time()

    # ----- Training Phase -----
    avg_train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)

    # ----- Validation Phase -----
    avg_val_loss = validate(model, val_dataloader, criterion, device)

    # ----- Learning Rate Scheduling -----
    # Scheduler monitors val_loss and reduces LR if it plateaus
    # This helps the model escape local minima and converge better
    scheduler.step(avg_val_loss)

    # Get current learning rate (can change due to scheduler)
    current_lr = optimizer.param_groups[0]['lr']

    # ----- Record History -----
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['learning_rates'].append(current_lr)

    # ----- Best Model Checkpointing -----
    # Save model weights when validation loss improves
    # This ensures we keep the best-performing model, not the most recently trained one
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Deep copy creates independent copy of model weights
        # Prevents best_model_state from being affected by future training
        best_model_state = copy.deepcopy(model.state_dict())
        early_stop_counter = 0  # Reset counter when improvement occurs
        improvement_flag = " ← NEW BEST!"
    else:
        early_stop_counter += 1  # Increment counter when no improvement
        improvement_flag = ""

    # ----- Epoch Summary -----
    epoch_time = time.time() - epoch_start

    # Print progress every epoch
    print(f"Epoch {epoch:03d}/{epochs} | "
          f"Train Loss: {avg_train_loss:.6f} | "
          f"Val Loss: {avg_val_loss:.6f} | "
          f"LR: {current_lr:.2e} | "
          f"Time: {epoch_time:.1f}s"
          f"{improvement_flag}")

    # ----- Early Stopping Check -----
    # If validation loss hasn't improved for 'patience' epochs, stop training
    # This prevents wasting time on training that's no longer improving
    if early_stop_counter >= early_stop_patience:
        print(f"\n⚠️  Early stopping triggered! No improvement for {early_stop_patience} epochs.")
        print(f"Best validation loss: {best_val_loss:.6f}")
        break

# ========================================
# POST-TRAINING: LOAD BEST MODEL
# ========================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

if best_model_state is not None:
    # Load the best model weights (from epoch with lowest validation loss)
    model.load_state_dict(best_model_state)
    print(f"✓ Loaded best model with validation loss: {best_val_loss:.6f}")
else:
    print("⚠️  No best model saved (this shouldn't happen).")

# ========================================
# SAVE MODEL TO DISK
# ========================================
# Save trained model for future use without retraining
model_save_path = "best_attention_unet_model.pth"
torch.save({
    'epoch': epoch,
    'model_state_dict': best_model_state,
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_loss': best_val_loss,
    'history': history,
}, model_save_path)
print(f"✓ Model saved to: {model_save_path}")

# ========================================
# TRAINING SUMMARY STATISTICS
# ========================================
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"Total epochs trained: {epoch}")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Final training loss: {history['train_loss'][-1]:.6f}")
print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
print(f"Final learning rate: {history['learning_rates'][-1]:.2e}")

# Calculate improvement
if len(history['val_loss']) > 1:
    initial_val_loss = history['val_loss'][0]
    improvement = ((initial_val_loss - best_val_loss) / initial_val_loss) * 100
    print(f"Validation loss improvement: {improvement:.2f}%")

print("=" * 60 + "\n")

# ========================================
# OPTIONAL: PLOT TRAINING CURVES
# ========================================
"""
Uncomment this section to visualize training progress

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

# Plot 1: Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Learning Rate Schedule
plt.subplot(1, 2, 2)
plt.plot(history['learning_rates'], linewidth=2, color='green')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.yscale('log')  # Log scale to see small LR changes
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Training curves saved to: training_history.png")
"""

# ========================================
# HOW TO LOAD THE MODEL LATER
# ========================================
"""
To load the trained model for inference or further training:

checkpoint = torch.load('best_attention_unet_model.pth')
model = AttentionUNet(in_channels=3, num_outputs=5, base_features=64, bilinear=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()  # Set to evaluation mode for inference

# Access training history
history = checkpoint['history']
print(f"Model was trained for {checkpoint['epoch']} epochs")
print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
"""
