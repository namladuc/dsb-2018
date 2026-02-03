import torch.nn as nn
import torch
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
import torch.nn.functional as F
from functools import partial
from torch import Tensor

### SIMPLE UNET 2D MODULE ###
class DoubleConv(nn.Module):
    """(Conv2D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
          )     
        
    def forward(self,x):
        return self.double_conv(x)
   
class Down2D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.encoder(x)
    
class Down2DTiled(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2, 2),
        )
    def forward(self, x):
        return self.encoder(x)

class Up2D(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear='false'):
        super().__init__()
        
        if bilinear != 'false':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x       

class Out2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)

# https://www.youtube.com/watch?v=KOF38xAvo8I
class Attention2(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, in_channels):
        super(Attention2, self).__init__()
        self.conv_g = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_x = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=2)
        self.psi    = nn.Conv2d(in_channels * 2, 1, 1)
        self.sig    = nn.Sigmoid()
        self.re     = nn.ReLU(inplace=True)
        self.up_sam = nn.Upsample(scale_factor=2)

    def forward(self, g, x):
        '''
        g: low layer
        x: skip connection
        '''
        phi_g   = self.conv_g(g)
        theta_x = self.conv_x(x)
        out     = self.re(torch.cat([phi_g, theta_x], dim=1))
        out     = self.psi(out)
        out     = self.sig(out)
        out     = self.up_sam(out)
        return out

class UpAtt2D(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear='false'):
        super().__init__()
        
        if bilinear != 'false':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_channels, out_channels)
        self.att  = Attention2(in_channels // 2)
        
    def forward(self, x1, x2):
        x2 = self.att(x1, x2) * x2
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x        
### SIMPLE UNET MODULE ###

### Segmentation Model Pytorch ###
class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)

class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)

class Attention(nn.Module):
    def __init__(self, name, **params):
            super().__init__()

            if name is None:
                self.attention = nn.Identity(**params)
            elif name == "scse":
                self.attention = SCSEModule(**params)
            else:
                raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)
    
class Conv2dReLUFire(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_batchnorm=True,
    ):

        conv = FireModule(
            in_channels,
            out_channels // 2,
            out_channels
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLUFire, self).__init__(conv, bn, relu) 
### Segmentation Model Pytorch ###


### Fire Module Blocks ###
class DecoderFireBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLUFire(
            in_channels + skip_channels,
            out_channels,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLUFire(
            out_channels,
            out_channels,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze, expand):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze, expand // 2, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze, expand // 2, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class FireModuleV2(nn.Module):
    def __init__(self, in_channels, squeeze, expand, expand_ratio=2, expand_kernel=3):
        super(FireModuleV2, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze, expand - (expand // expand_ratio), kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze, expand // expand_ratio, kernel_size=expand_kernel, padding=expand_kernel//2)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)
        
class FireModuleSmall(nn.Module):
    def __init__(self, in_channels, squeeze, expand):
        super(FireModuleSmall, self).__init__()
        self.expand1x1 = nn.Conv2d(in_channels, expand // 2, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(in_channels, expand // 2, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)
    
class FireModule3D(nn.Module):
    def __init__(self, in_channels, squeeze, expand):
        super(FireModule3D, self).__init__()
        self.squeeze = nn.Conv3d(in_channels, squeeze, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv3d(squeeze, expand // 2, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv3d(squeeze, expand // 2, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class FireModuleNorm(nn.Module):
    def __init__(self, in_channels, squeeze, expand):
        super(FireModuleNorm, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze, expand // 2, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze, expand // 2, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(expand)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        left = self.expand1x1_activation(self.expand1x1(x))
        right = self.expand3x3_activation(self.expand3x3(x))
        return self.batchnorm(torch.cat([left, right], dim=1))
    
class Double2FireModule(nn.Module):
    def __init__(self, in_channels, out_Channel, fire_channel = 2):
        super().__init__()
        self.double_fire = nn.Sequential(
            FireModule(in_channels, out_Channel // fire_channel, out_Channel),
            nn.BatchNorm2d(out_Channel),
            nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost
            FireModule(out_Channel, out_Channel // fire_channel, out_Channel),
            nn.BatchNorm2d(out_Channel),
            nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost
          )     
    def forward(self, x):
        return self.double_fire(x)
    
class OneFireModule(nn.Module):
    def __init__(self, in_channels, out_Channel):
        super().__init__()
        self.double_fire = nn.Sequential(
            FireModule(in_channels, out_Channel // 2, out_Channel),
            nn.BatchNorm2d(out_Channel),
            nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost
          )     
    def forward(self, x):
        return self.double_fire(x)
    
class Down2FireModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.down_fire = Double2FireModule(in_channels, out_channels)
        
    def forward(self, x):
        return self.downsample(self.down_fire(x))
    
class DownFireModuleV2(nn.Module):
    def __init__(self, in_channels, out_channels, fire_channel = 4, expand_ratio=4, expand_kernel=5):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.fire_module = nn.Sequential(
            FireModuleV2(in_channels, out_channels // fire_channel, out_channels,
                         expand_ratio=expand_ratio, expand_kernel=expand_kernel),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost
        )
        
    def forward(self, x):
        return self.fire_module(self.downsample(x))
    
class DownFireModuleV2Res(nn.Module):
    def __init__(self, in_channels, out_channels, fire_channel = 4, expand_ratio=4, expand_kernel=5):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.fire_module = nn.Sequential(
            FireModuleV2(in_channels, out_channels // fire_channel, out_channels,
                         expand_ratio=expand_ratio, expand_kernel=expand_kernel),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost
        )
        
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),            
        )
        
    def forward(self, x):
        x = self.downsample(x)
        return self.fire_module(x) + self.identity(x)
    
class DownFireModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.fire_module = nn.Sequential(
            FireModule(in_channels, out_channels // 2, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost
        )
        
    def forward(self, x):
        return self.downsample(self.fire_module(x))

class DownRes1FireModule(nn.Module):
    def __init__(self, in_channels, out_channels, fire_channel = 2):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.fire_module = nn.Sequential(
            FireModule(in_channels, out_channels // fire_channel, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost
        )
        
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),            
        )
        
    def forward(self, x):
        return self.downsample(self.fire_module(x) + self.identity(x))
    
class DownRes3FireModule(nn.Module):
    def __init__(self, in_channels, out_channels, fire_channel = 2):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.fire_module = nn.Sequential(
            FireModule(in_channels, out_channels // fire_channel * 4, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost
        
            FireModule(out_channels, out_channels // fire_channel * 2, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost
        
            FireModule(out_channels, out_channels // fire_channel, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost
        )
        
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),            
        )
        
    def forward(self, x):
        return self.downsample(self.fire_module(x) + self.identity(x))

class Down2DResBottle(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.down_conv  = DoubleConv(in_channels, out_channels)
        
        self.identity_bottle = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),    
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),    
            nn.ReLU(inplace=True),
            nn.Conv2d(1, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),    
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.downsample(self.down_conv(x) + self.identity_bottle(x))

class Down2DRes(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.down_conv  = DoubleConv(in_channels, out_channels)
        
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),            
        )
    def forward(self, x):
        x = self.down_conv(x) + self.identity(x)
        return self.downsample(x)

class Down2FireModuleRes(nn.Module):
    def __init__(self, in_channels, out_channels, fire_channel = 2):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.down_fire = Double2FireModule(in_channels, out_channels, fire_channel=fire_channel)
        
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),            
        )
    def forward(self, x):
        x = self.down_fire(x) + self.identity(x)
        return self.downsample(x)
    
class Down2FireModuleResFix(nn.Module):
    def __init__(self, in_channels, out_channels, fire_channel = 2):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.down_fire = Double2FireModule(in_channels, out_channels, fire_channel=fire_channel)
        
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),            
        )
    def forward(self, x):
        x = self.downsample(x)
        return self.down_fire(x) + self.identity(x)

class Down1FireModuleFix(nn.Module):
    def __init__(self, in_channels, out_channels, fire_channel = 2):
        super().__init__()
        self.down_fire = nn.Sequential(
            nn.MaxPool2d(2, 2),
            FireModule(in_channels, out_channels // fire_channel, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )         

    def forward(self, x):
        return self.down_fire(x)

class Down1FireModuleResFix(nn.Module):
    def __init__(self, in_channels, out_channels, fire_channel = 2):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.down_fire = nn.Sequential(
            FireModule(in_channels, out_channels // fire_channel, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),            
        )
    def forward(self, x):
        x = self.downsample(x)
        return self.down_fire(x) + self.identity(x)
    
class DownConvFireModuleResFix(nn.Module):
    def __init__(self, in_channels, out_channels, fire_channel = 2):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.down_fire = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            FireModule(out_channels, out_channels // fire_channel, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
            
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),            
        )
    def forward(self, x):
        x = self.downsample(x)
        return self.down_fire(x) + self.identity(x)
    
class Down2FireSuperRes(nn.Module):
    def __init__(self, in_channels, out_channels, fire_channel = 2):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        
        self.down_fire_1 = nn.Sequential(
            FireModule(in_channels, out_channels // fire_channel, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost
        )
        
        self.down_fire_2 = nn.Sequential(
            FireModule(out_channels, out_channels // fire_channel, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost
        )
        
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),            
        )
    def forward(self, x):
        x1 = self.down_fire_1(x) + self.identity(x)
        x2 = self.down_fire_2(x1) + x1
        return self.downsample(x2 + self.identity(x))
    
class DownOneConvRes(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Conv2d(out_channels, out_channels,
                                      kernel_size=2, stride=2)
        self.down_fire = OneFireModule(in_channels, out_channels)
        
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),            
        )
    def forward(self, x):
        x = self.down_fire(x) + self.identity(x)
        return self.downsample(x)
    
class Down2FireModuleResAvg(nn.Module):
    def __init__(self, in_channels, out_channels, avg_conv_use=True):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.down_fire  = Double2FireModule(in_channels, out_channels)
        self.avg_conv_use = avg_conv_use
        
        self.avg_conv = nn.Conv2d(out_channels, out_channels,
                                      kernel_size=2, stride=2)
        
        self.identity   = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),            
        )
    def forward(self, x, y):
        if self.avg_conv_use:
            x = self.down_fire(x) + self.identity(y)
        else:   
            x = self.down_fire(x) + self.identity(x)
        return self.downsample(x), self.avg_conv(x) 
    
class Up1FireModule(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear='false', fire_split = 2):
        super().__init__()
        
        if bilinear != 'false':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = FireModule(in_channels, out_channels // fire_split, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x       

class Up1FireModuleV2(nn.Module):

    def __init__(self, in_channels, out_channels, fire_split = 2, expand_ratio=4, expand_kernel=5):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = FireModuleV2(in_channels, out_channels // fire_split, out_channels, expand_ratio, expand_kernel)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x       

class UpFireModule(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear='false'):
        super().__init__()
        
        if bilinear != 'false':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = Double2FireModule(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x       
### Fire Module Blocks ###
    
### Attention Module Blocks ###
#https://github.com/sfczekalski/attention_unet/blob/master/att_unet.ipynb
class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 2, 1)
        self.sig = nn.Sigmoid()

    def forward(self, decoder, encoder):
        max_pool = F.max_pool2d(encoder, kernel_size=encoder.size()[2:])
        decoder = self.sig(max_pool) * decoder
        return self.conv(decoder)

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class UpForChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        self.att = Attention_block(in_channels // 2, in_channels // 2, out_channels // 2)
        
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, skip_connect):
        x1 = self.up(x1)
        
        skip_connect = self.att(x1, skip_connect)
        
        diffY = skip_connect.size()[2] - x1.size()[2]
        diffX = skip_connect.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        
        x1 = torch.cat([skip_connect, x1], dim=1)
        return self.conv(x1)
    
class Up1FireModuleV2Att(nn.Module):

    def __init__(self, in_channels, out_channels, fire_split = 2, expand_ratio=4, expand_kernel=5):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.att = Attention_block(in_channels // 2, in_channels // 2, out_channels // 2)
            
        self.conv = FireModuleV2(in_channels, out_channels // fire_split, out_channels, expand_ratio, expand_kernel)
        
    def forward(self, x1, skip_connect):
        x1 = self.up(x1)
        
        skip_connect = self.att(x1, skip_connect)
        
        diffY = skip_connect.size()[2] - x1.size()[2]
        diffX = skip_connect.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        
        x1 = torch.cat([skip_connect, x1], dim=1)
        return self.conv(x1)
### Attention Module Blocks ###

### CCA Blocks ###
def get_activation(activation_type):
    
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out

class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv = 2, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)
    
class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):

        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            norm(out_features),
            act(),
        )

class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x += res
        return x

class FusedMBConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4, explain_kernel: int = 3):
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        expanded_features = in_features * expansion
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        partial(ConvNormAct, kernel_size=explain_kernel)
                            (
                                in_features, 
                                expanded_features, 
                                act=nn.ReLU6
                            ),
                        # here you can apply SE
                        # wide -> narrow
                        partial(ConvNormAct, kernel_size=1)(
                            expanded_features,
                            out_features,
                            act=nn.Identity
                        ),
                    ),
                ),
                nn.ReLU(),
            )
        )
    
# Ver 7
class DownClose7(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, squeeze_rate=2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            FusedMBConv(in_channels, out_channels, expansion=4, explain_kernel=3),
            
            nn.Conv2d(out_channels, in_channels // squeeze_rate, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // squeeze_rate),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels // squeeze_rate, in_channels // squeeze_rate, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
            nn.BatchNorm2d(in_channels // squeeze_rate),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels // squeeze_rate, out_channels, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.encoder(x)
    
class DownClose6(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, squeeze_rate=2):
        super().__init__()
        
        kernel_conv = None
        if kernel_size == 1:
            kernel_conv = nn.Conv2d(in_channels // squeeze_rate, out_channels, kernel_size=1, padding=1)
        if kernel_size == 3:
            kernel_conv = nn.Conv2d(in_channels // squeeze_rate, out_channels, kernel_size=3, stride=2, padding=1)
        if kernel_size == 5:
            kernel_conv = nn.Conv2d(in_channels // squeeze_rate, out_channels, kernel_size=5, stride=2, padding=2)
        if kernel_size == 7:
            kernel_conv = nn.Conv2d(in_channels // squeeze_rate, out_channels, kernel_size=7, stride=2, padding=3)
        if kernel_size == 9:
            kernel_conv = nn.Conv2d(in_channels // squeeze_rate, out_channels, kernel_size=9, stride=2, padding=4)
            
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // squeeze_rate, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // squeeze_rate),
            nn.ReLU(inplace=True),
            
            kernel_conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.encoder(x)

###