import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import copy
import time
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# from csiro_biomass.u_net import AttentionUNet
from csiro_biomass.cnn import CNNModel

import warnings

warnings.filterwarnings('ignore')


# ========================================
# WEIGHTED R² CALCULATION
# ========================================
class WeightedR2Loss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, y_pred, y_true):
        # Move weights to same device as inputs
        self.weights = self.weights.to(y_pred.device)

        # Flatten and repeat weights
        n_samples = y_true.shape[0]
        weights_flat = self.weights.repeat(n_samples, 1).reshape(-1)

        # Flatten predictions and targets
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)

        # Calculate weighted mean
        y_weighted_mean = torch.sum(weights_flat * y_true_flat) / torch.sum(weights_flat)

        # Calculate RSS and TSS
        rss = torch.sum(weights_flat * (y_true_flat - y_pred_flat) ** 2)
        tss = torch.sum(weights_flat * (y_true_flat - y_weighted_mean) ** 2)

        # Avoid division by zero
        if tss == 0:
            return torch.tensor(0.0, device=y_pred.device)

        # Return 1 - R² (so we minimize this loss)
        return 1 - (1 - rss / tss)


def weighted_r2_score(y_true, y_pred, weights):
    """
    Calculate globally weighted R² score

    Args:
        y_true: Ground truth values [n_samples, n_targets]
        y_pred: Predicted values [n_samples, n_targets]
        weights: Per-target weights [n_targets]

    Returns:
        weighted_r2: Globally weighted R² score
    """
    # Flatten all predictions and targets
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    # Create weight array matching the flattened arrays
    n_samples = y_true.shape[0]
    weights_flat = np.repeat(weights, n_samples)

    # Calculate weighted mean of ground truth
    y_weighted_mean = np.average(y_true_flat, weights=weights_flat)

    # Calculate residual sum of squares (RSS)
    rss = np.sum(weights_flat * (y_true_flat - y_pred_flat) ** 2)

    # Calculate total sum of squares (TSS)
    tss = np.sum(weights_flat * (y_true_flat - y_weighted_mean) ** 2)

    # Calculate weighted R²
    weighted_r2 = 1 - (rss / tss) if tss != 0 else 0.0

    return weighted_r2


# ========================================
# TARGET WEIGHTS
# ========================================
target_weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])  # Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g


# ========================================
# DATASET CLASS (Modified with resizing)
# ========================================
class BiomassDataset(Dataset):
    def __init__(self, base_path, targets, img_size=(512, 1024)):
        super().__init__()
        self.targets = targets
        self.base_path = base_path
        self.img_size = img_size

        # Define transform with resizing
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
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


# ========================================
# HYPERPARAMETERS
# ========================================
learning_rate = 0.0001
epochs = 200
weight_decay = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# ========================================
# MODEL INITIALIZATION
# ========================================
# Try a more powerful architecture
import torchvision.models as models


# Option 1: Use pretrained backbone
class BiomassModel(nn.Module):
    def __init__(self, num_targets=5):
        super().__init__()
        self.backbone = models.resnet34(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_targets)
        )

    def forward(self, x):
        return self.backbone(x)


model = BiomassModel(num_targets=5).to(device)
# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ========================================
# LOSS FUNCTION AND OPTIMIZER
# ========================================
criterion = WeightedR2Loss(weights=target_weights)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# ========================================
# LEARNING RATE SCHEDULER
# ========================================
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
)

# ========================================
# EARLY STOPPING SETUP
# ========================================
early_stop_patience = 15
early_stop_counter = 0
best_val_r2 = -float("inf")  # Start with negative infinity since we want to maximize R²
best_model_state = None

# ========================================
# TRAINING HISTORY TRACKING
# ========================================
history = {
    'train_loss': [],
    'val_r2': [],  # Now tracking R² instead of validation loss
    'learning_rates': []
}


# ========================================
# TRAINING FUNCTION
# ========================================
def train_one_epoch(model, trainloader, optimizer, criterion, device):
    """
    Train the model for one epoch using weighted R² loss
    """
    model.train()
    train_loss_sum = 0.0
    n_train_batches = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        preds = model(inputs)
        loss = criterion(preds, targets)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        n_train_batches += 1

    avg_train_loss = train_loss_sum / max(1, n_train_batches)
    return avg_train_loss


# ========================================
# VALIDATION FUNCTION
# ========================================
def validate_with_weighted_r2(model, valloader, device, weights):
    """
    Evaluate the model on validation data using weighted R² metric
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in valloader:
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            preds = model(inputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Calculate weighted R²
    weighted_r2 = weighted_r2_score(all_targets, all_preds, weights)

    return weighted_r2


# ========================================
# PREDICTION AND SUBMISSION FUNCTION
# ========================================
def create_submission(model, test_dataloader, device, image_ids, output_path='submission.csv'):
    """
    Create submission file in the required format
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in test_dataloader:  # Assuming test dataloader returns (inputs, _)
            inputs = inputs.to(device, dtype=torch.float32)
            preds = model(inputs)
            all_preds.append(preds.cpu().numpy())

    # Concatenate all predictions
    all_preds = np.vstack(all_preds)

    # Create submission DataFrame
    submission_data = []
    target_names = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

    for i, image_id in enumerate(image_ids):
        for j, target_name in enumerate(target_names):
            sample_id = f"{image_id}__{target_name}"
            target_value = all_preds[i, j]
            submission_data.append({'sample_id': sample_id, 'target': target_value})

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_path, index=False)
    print(f"✓ Submission file saved to: {output_path}")

    return submission_df


# ========================================
# MAIN TRAINING LOOP
# ========================================
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60 + "\n")

# Assuming you have train_dataloader and val_dataloader defined
# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

for epoch in range(1, epochs + 1):
    epoch_start = time.time()

    # ----- Training Phase -----
    avg_train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)

    # ----- Validation Phase -----
    weighted_r2 = validate_with_weighted_r2(model, val_dataloader, device, target_weights)
    val_loss = 1 - weighted_r2  # Convert R² to loss for scheduler

    # ----- Learning Rate Scheduling -----
    scheduler.step(val_loss)  # Scheduler minimizes (1 - R²)

    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']

    # ----- Record History -----
    history['train_loss'].append(avg_train_loss)
    history['val_r2'].append(weighted_r2)
    history['learning_rates'].append(current_lr)

    # ----- Best Model Checkpointing -----
    if weighted_r2 > best_val_r2:
        best_val_r2 = weighted_r2
        best_model_state = copy.deepcopy(model.state_dict())
        early_stop_counter = 0
        improvement_flag = " ← NEW BEST!"
    else:
        early_stop_counter += 1
        improvement_flag = ""

    # ----- Epoch Summary -----
    epoch_time = time.time() - epoch_start

    print(f"Epoch {epoch:03d}/{epochs} | "
          f"Train Loss: {avg_train_loss:.6f} | "
          f"Val R²: {weighted_r2:.6f} | "
          f"LR: {current_lr:.2e} | "
          f"Time: {epoch_time:.1f}s"
          f"{improvement_flag}")

    # ----- Early Stopping Check -----
    if early_stop_counter >= early_stop_patience:
        print(f"\n⚠️  Early stopping triggered! No improvement for {early_stop_patience} epochs.")
        print(f"Best validation R²: {best_val_r2:.6f}")
        break

# ========================================
# POST-TRAINING: LOAD BEST MODEL
# ========================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"✓ Loaded best model with validation R²: {best_val_r2:.6f}")
else:
    print("⚠️  No best model saved.")

# ========================================
# SAVE MODEL TO DISK
# ========================================
model_save_path = "best_biomass_model.pth"
torch.save({
    'epoch': epoch,
    'model_state_dict': best_model_state,
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_r2': best_val_r2,
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
print(f"Best validation R²: {best_val_r2:.6f}")
print(f"Final training loss: {history['train_loss'][-1]:.6f}")
print(f"Final validation R²: {history['val_r2'][-1]:.6f}")
print(f"Final learning rate: {history['learning_rates'][-1]:.2e}")

# Calculate improvement
if len(history['val_r2']) > 1:
    initial_val_r2 = history['val_r2'][0]
    improvement = ((best_val_r2 - initial_val_r2) / abs(initial_val_r2)) * 100 if initial_val_r2 != 0 else float('inf')
    print(f"Validation R² improvement: {improvement:.2f}%")

print("=" * 60 + "\n")

# ========================================
# CREATE SUBMISSION FILE (when ready)
# ========================================
"""
# When you have test data ready, use this:
# test_image_ids = [...]  # List of test image IDs
# submission = create_submission(model, test_dataloader, device, test_image_ids, 'submission.csv')
"""


# ========================================
# PLOTTING FUNCTION (Optional)
# ========================================
def plot_training_history(history):
    """
    Plot training history
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))

    # Plot 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('1 - Weighted R²')
    plt.title('Training Loss (1 - Weighted R²)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Validation R²
    plt.subplot(1, 2, 2)
    plt.plot(history['val_r2'], label='Val R²', linewidth=2, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted R²')
    plt.title('Validation Weighted R²')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Training curves saved to: training_history.png")

# Uncomment to plot
# plot_training_history(history)
# plot_training_history(history)
