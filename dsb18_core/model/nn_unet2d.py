import torch.nn as nn
from .util.module_nn import DoubleConv
from .util.module_nn import Down2D as Down
from .util.module_nn import Up2D as Up
from .util.module_nn import Out2D as Out
import torch.nn.functional as F

class nnUNet2D(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels, isDeeply=False):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.isDeeply = isDeeply

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 16 * n_channels)
        self.enc5 = Down(16 * n_channels, 16 * n_channels)
        
        self.dec1 = Up(32 * n_channels, 8 * n_channels)
        self.dec2 = Up(16 * n_channels, 4 * n_channels)
        self.dec3 = Up(8 * n_channels, 2 * n_channels)
        self.dec4 = Up(4 * n_channels, n_channels)
        self.dec5 = Up(2 * n_channels, n_channels)

        if self.isDeeply:
            self.out1 = Out(8 * n_channels, n_classes)
            self.out2 = Out(4 * n_channels, n_classes)
            self.out3 = Out(2 * n_channels, n_classes)
            self.out4 = Out(n_channels, n_classes)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        x6 = self.enc5(x5)
    
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