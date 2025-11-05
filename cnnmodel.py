
import torch
import torch.nn as nn
import torch.nn.functional as F



class CNNModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=(32, 64, 128), hidden_fc=512, target_size=5):
        """
        A simple CNN-based feature extractor + MLP head.
        Can be used for regression or classification tasks.

        Args:
            in_channels: number of input channels (e.g., 3 for RGB)
            out_channels: tuple/list defining the number of filters in each conv block
            hidden_fc: size of the hidden fully-connected layer
            target_size: number of output values (e.g., 5 regression targets)
        """
        super().__init__()
        assert len(out_channels) >= 2, "out_channels must have at least two elements"

        # --- Convolutional feature extractor ---
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),

            # Block 2
            nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        # --- Global pooling to make features spatially invariant ---
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # --- Fully connected projection (latent representation) ---
        self.fc = nn.Sequential(
            nn.Flatten(),  # [B, C]
            nn.Linear(out_channels[-1], hidden_fc),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_fc),
            nn.Dropout(0.3),
        )

        # --- Final prediction head ---
        self.head = nn.Sequential(
            nn.Linear(hidden_fc, hidden_fc // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_fc // 2),
            nn.Dropout(0.25),

            nn.Linear(hidden_fc // 2, hidden_fc // 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_fc // 4),
            nn.Dropout(0.25),
            nn.Linear(hidden_fc // 4, target_size)
        )

    def forward(self, x):
        # x shape: [B, C, H, W]
        x = self.features(x)  # convolutional features
        x = self.global_pool(x)  # [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # flatten to [B, C]
        x = self.fc(x)  # latent representation [B, hidden_fc]
        out = self.head(x)  # final prediction [B, target_size]
        return out


