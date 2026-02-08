import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x


class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=True, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)

        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        img_patches = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)

        batch_size, tokens, _ = img_patches.shape

        project = self.projection(img_patches)
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size)

        patches = torch.cat([token, project], dim=1)
        patches += self.embedding[:tokens + 1, :]

        x = self.dropout(patches)
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]

        return x


class DecoderTransformer(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, CFG, decoder_channel=[48, 96, 192, 384, 768],
                 decoder_factor=[1, 2, 4, 8, 16], classification=False, num_classes=3):
        super().__init__()
        self.CFG = CFG
        self.trans = ViT(img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification)
        
        self.conv_filter = nn.ModuleList([
            nn.Conv2d(
                in_channels=decoder_channel[i],
                out_channels=num_classes,
                kernel_size=1
                )
            for i in range(len(decoder_channel))
        ])
        
        self.upsample_size = nn.ModuleList([
            nn.Upsample(scale_factor=decoder_factor[i])
            for i in range(len(decoder_factor))
        ])
    def forward(self, features):
        upSize = [
            self.upsample_size[i](features[i]) 
            for i in range(len(features))
            ]
        
        reduceChannel = [
            self.conv_filter[i](upSize[i])
            for i in range(len(upSize))
        ]
        
        x = reduceChannel[0]
        for i in range(1, len(reduceChannel)):
            x += reduceChannel[i]
            
        out = self.trans(x)
        out = rearrange(out ,
                        "b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)",
                        x       = self.CFG.img_size[0] // self.CFG.patch_dim,
                        y       = self.CFG.img_size[0] // self.CFG.patch_dim,
                        c       = self.CFG.num_classes, 
                        patch_x = self.CFG.patch_dim
                    )
        return out

if __name__ == '__main__':
    vit = ViT(img_dim=384,
              in_channels=3,
              embedding_dim=768,
              patch_dim=16,
              block_num=8,
              head_num=4,
              classification=False,
              num_classes=3,
              mlp_dim=512)
    print(sum(p.numel() for p in vit.parameters()))
    x = torch.rand(1, 3, 384, 384)
    a = vit(x)
    print(a.shape)
    print(rearrange(a , "b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)", x=384 // 16, y=384 // 16, c=3, patch_x=16).shape)
    print(rearrange(x, 'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=16, patch_y=16).shape)