import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution block: (Conv2d -> BatchNorm -> ReLU) x 2
    This is the basic building block used in both encoder and decoder paths.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate as proposed in Attention U-Net (Additive Attention).

    The attention gate learns to suppress irrelevant regions in the skip connection
    while highlighting salient features useful for the task.

    Formula: α = σ(ψ(ReLU(W_g * g + W_x * x + b)))
    Output: x_att = α ⊙ x (element-wise multiplication)

    Args:
        F_g (int): Number of channels in gating signal (from decoder)
        F_l (int): Number of channels in skip connection (from encoder)
        F_int (int): Number of intermediate channels for attention computation
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        # Transform gating signal to intermediate dimension
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Transform skip connection to intermediate dimension
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Output attention coefficients (1 channel per spatial location)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()  # Produces values in [0, 1]
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder path (coarser scale, upsampled)
            x: Skip connection from encoder path (finer scale)

        Returns:
            x_att: Attention-weighted skip connection
            attention_weights: The attention map (useful for visualization)
        """
        # Apply transformations
        g1 = self.W_g(g)  # Transform gating signal
        x1 = self.W_x(x)  # Transform skip connection

        # Additive attention: combine both signals
        psi = self.relu(g1 + x1)  # Element-wise addition + ReLU
        psi = self.psi(psi)  # Generate attention coefficients [0, 1]

        # Apply attention weights to skip connection
        x_att = x * psi  # Element-wise multiplication (broadcasting)

        return x_att, psi


class Down(nn.Module):
    """
    Downsampling block: MaxPool2d -> DoubleConv
    Reduces spatial dimensions by 2x while increasing feature channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block with Attention Gate:
    Upsample -> Conv2d -> Apply Attention to skip connection -> Concatenate -> DoubleConv

    The attention gate is applied BEFORE concatenation to filter the skip connection.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        # Attention gate: F_g = in_channels // 2 (upsampled), F_l = in_channels // 2 (skip)
        self.attention = AttentionGate(
            F_g=in_channels // 2,  # Channels from upsampled path
            F_l=in_channels // 2,  # Channels from skip connection
            F_int=in_channels // 4  # Intermediate channels (typically F_g/2)
        )

    def forward(self, x1, x2):
        """
        Args:
            x1: Input from previous decoder layer (lower resolution, gating signal)
            x2: Skip connection from encoder path (higher resolution)

        Returns:
            output: Concatenated and processed features
            attention_map: Attention weights (for visualization/debugging)
        """
        # Upsample the decoder features
        x1 = self.up(x1)

        # Handle potential size mismatches between x1 and x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Apply attention gate: x1 is gating signal, x2 is skip connection
        x2_att, attention_map = self.attention(g=x1, x=x2)

        # Concatenate attention-weighted skip connection with upsampled features
        x = torch.cat([x2_att, x1], dim=1)

        return self.conv(x), attention_map


class AttentionUNet(nn.Module):
    """
    Attention U-Net architecture for image-to-regression tasks.

    Key improvements over standard U-Net:
    - Attention gates filter skip connections to focus on relevant regions
    - Reduces feature redundancy and improves gradient flow
    - Particularly useful when only certain image regions are relevant for regression

    Architecture:
        - Encoder: 4 downsampling blocks extracting hierarchical features
        - Bottleneck: Deepest layer with maximum feature channels
        - Decoder: 4 upsampling blocks with attention-gated skip connections
        - Output: Global Average Pooling -> FC layer -> 5 regression values

    Args:
        in_channels (int): Number of input image channels (3 for RGB)
        num_outputs (int): Number of regression outputs (5 in your case)
        base_features (int): Number of features in first layer
        bilinear (bool): Use bilinear upsampling (True) or transposed conv (False)
    """

    def __init__(self, in_channels=3, num_outputs=5, base_features=64, bilinear=True):
        super(AttentionUNet, self).__init__()
        self.in_channels = in_channels
        self.num_outputs = num_outputs
        self.bilinear = bilinear

        # Initial convolution (no downsampling)
        self.inc = DoubleConv(in_channels, base_features)

        # Encoder path (downsampling)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)

        # Adjust factor for bilinear upsampling
        factor = 2 if bilinear else 1
        self.down4 = Down(base_features * 8, base_features * 16 // factor)

        # Decoder path (upsampling with attention-gated skip connections)
        self.up1 = Up(base_features * 16, base_features * 8 // factor, bilinear)
        self.up2 = Up(base_features * 8, base_features * 4 // factor, bilinear)
        self.up3 = Up(base_features * 4, base_features * 2 // factor, bilinear)
        self.up4 = Up(base_features * 2, base_features, bilinear)

        # Global pooling and regression head
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Reduces to 1x1 spatial dims
        self.fc = nn.Linear(base_features, num_outputs)  # Maps to regression outputs

        # Optional: store attention maps for visualization
        self.attention_maps = []

    def forward(self, x):
        """
        Forward pass through Attention U-Net.

        Args:
            x: Input tensor of shape (batch_size, 3, 1000, 2000)

        Returns:
            output: Regression values of shape (batch_size, 5)
        """
        # Clear previous attention maps
        self.attention_maps = []

        # Encoder path - save skip connections
        x1 = self.inc(x)  # (B, 64, H, W)
        x2 = self.down1(x1)  # (B, 128, H/2, W/2)
        x3 = self.down2(x2)  # (B, 256, H/4, W/4)
        x4 = self.down3(x3)  # (B, 512, H/8, W/8)
        x5 = self.down4(x4)  # (B, 512, H/16, W/16) - bottleneck

        # Decoder path with attention gates on skip connections
        x, att1 = self.up1(x5, x4)  # (B, 256, H/8, W/8)
        x, att2 = self.up2(x, x3)  # (B, 128, H/4, W/4)
        x, att3 = self.up3(x, x2)  # (B, 64, H/2, W/2)
        x, att4 = self.up4(x, x1)  # (B, 64, H, W)

        # Store attention maps for potential visualization
        self.attention_maps = [att1, att2, att3, att4]

        # Global pooling and regression
        x = self.global_pool(x)  # (B, 64, 1, 1)
        x = torch.flatten(x, 1)  # (B, 64)
        x = self.fc(x)  # (B, 5)

        return x

    def get_attention_maps(self):
        """
        Retrieve attention maps from the last forward pass.
        Useful for visualizing which regions the model focuses on.

        Returns:
            List of attention maps from each decoder level [deepest -> shallowest]
        """
        return self.attention_maps

'''
# Example usage and comparison
# Initialize Attention U-Net
model = AttentionUNet(in_channels=3, num_outputs=5, base_features=64, bilinear=True)

for data, targets in dataloader:
    print('data.shape:', data.shape)
    y = model(data)
    print(y)
    print(targets)

    # Get attention maps
    attention_maps = model.get_attention_maps()

    print(f"Model initialized successfully")
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nAttention maps at each decoder level:")
    for i, att_map in enumerate(attention_maps):
        print(f"  Level {i + 1}: {att_map.shape}")

    # Optional: Visualize attention (requires matplotlib)
    """
    import matplotlib.pyplot as plt
    
    # Get attention map from finest scale (last decoder level)
    att = attention_maps[-1][0, 0].detach().cpu().numpy()  # First sample, single channel
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(data[0].permute(1, 2, 0).cpu().numpy())
    plt.title('Input Image')
    plt.subplot(122)
    plt.imshow(att, cmap='hot')
    plt.colorbar()
    plt.title('Attention Map (Brightest = Most Important)')
    plt.show()
    """
    break

'''
