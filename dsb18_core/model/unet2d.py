"""U-Net 2D segmentation model."""
import torch.nn as nn
from .util.modules import DoubleConv, Down2D, Up2D, Out2D


class UNet(nn.Module):
    """U-Net for 2D image segmentation.
    
    Architecture:
        - Encoder: 5 downsampling blocks
        - Decoder: 5 upsampling blocks with skip connections
    """
    
    def __init__(self, in_channels, n_classes, n_channels, isDeeply=False):
        """Initialize U-Net.
        
        Args:
            in_channels: Number of input channels
            n_classes: Number of output classes
            n_channels: Number of base channels
            isDeeply: Whether to use deep supervision
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.isDeeply = isDeeply
        
        # Encoder
        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down2D(n_channels, 2 * n_channels)
        self.enc2 = Down2D(2 * n_channels, 4 * n_channels)
        self.enc3 = Down2D(4 * n_channels, 8 * n_channels)
        self.enc4 = Down2D(8 * n_channels, 16 * n_channels)
        self.enc5 = Down2D(16 * n_channels, 16 * n_channels)
        
        # Decoder
        self.dec1 = Up2D(32 * n_channels, 8 * n_channels)
        self.dec2 = Up2D(16 * n_channels, 4 * n_channels)
        self.dec3 = Up2D(8 * n_channels, 2 * n_channels)
        self.dec4 = Up2D(4 * n_channels, n_channels)
        self.dec5 = Up2D(2 * n_channels, n_channels)
        
        # Output
        if self.isDeeply:
            self.out1 = Out2D(8 * n_channels, n_classes)
            self.out2 = Out2D(4 * n_channels, n_classes)
            self.out3 = Out2D(2 * n_channels, n_classes)
            self.out4 = Out2D(n_channels, n_classes)
        self.out = Out2D(n_channels, n_classes)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input image (B, C, H, W)
            
        Returns:
            Segmentation mask (B, num_classes, H, W)
        """
        # Encoder: store features for skip connections
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        x6 = self.enc5(x5)
        
        # Decoder: upsample and combine with encoder features
        mask1 = self.dec1(x6, x5)
        mask2 = self.dec2(mask1, x4)
        mask3 = self.dec3(mask2, x3)
        mask4 = self.dec4(mask3, x2)
        mask5 = self.dec5(mask4, x1)

        mask = self.out(mask5)
        if self.isDeeply:
            mask4Out = self.out4(mask4)
            mask3Out = self.out3(mask3)
            mask2Out = self.out2(mask2)
            mask1Out = self.out1(mask1)
            out = [mask, mask4Out, mask3Out, mask2Out, mask1Out]
            return out
        return mask
