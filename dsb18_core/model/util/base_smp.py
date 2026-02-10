from typing import List, Union, Sequence, Optional, Dict, Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoderBlock, UnetCenterBlock
from .modules import Activation


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.Upsample(mode="bilinear", scale_factor=upsampling, align_corners=True)
            if upsampling > 1
            else nn.Identity()
        )
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class UnetDecoder(nn.Module):
    """The decoder part of the U-Net architecture.

    Takes encoded features from different stages of the encoder and progressively upsamples them while
    combining with skip connections. This helps preserve fine-grained details in the final segmentation.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        n_blocks: int = 5,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        attention_type: Optional[str] = None,
        add_center_block: bool = False,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if add_center_block:
            self.center = UnetCenterBlock(
                head_channels,
                head_channels,
                use_norm=use_norm,
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        self.blocks = nn.ModuleList()
        for block_in_channels, block_skip_channels, block_out_channels in zip(
            in_channels, skip_channels, out_channels
        ):
            block = UnetDecoderBlock(
                block_in_channels,
                block_skip_channels,
                block_out_channels,
                use_norm=use_norm,
                attention_type=attention_type,
                interpolation_mode=interpolation_mode,
            )
            self.blocks.append(block)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # spatial shapes of features: [hw, hw/2, hw/4, hw/8, ...]
        spatial_shapes = [feature.shape[2:] for feature in features]
        spatial_shapes = spatial_shapes[::-1]

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skip_connections = features[1:]

        x = self.center(head)
        out = []
        for i, decoder_block in enumerate(self.blocks):
            # upsample to the next spatial shape
            height, width = spatial_shapes[i + 1]
            skip_connection = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, height, width, skip_connection=skip_connection)
            out.append(x)
        return out


class SegmentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._is_encoder_frozen = False

    def initialize(self):
        self.initialize_decoder(self.decoder)
        if self.isDeeply:
            for index in range(len(self.segmentation_heads)):
                self.initialize_head(self.segmentation_heads[index])
        else:
            self.initialize_head(self.segmentation_heads)

    def initialize_decoder(self, module):
        for m in module.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def initialize_head(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def load_state_dict(self, state_dict, **kwargs):
        # for compatibility of weights for
        # timm- ported encoders with TimmUniversalEncoder
        from segmentation_models_pytorch.encoders import TimmUniversalEncoder

        if isinstance(self.encoder, TimmUniversalEncoder):
            patterns = ["regnet", "res2", "resnest", "mobilenetv3", "gernet"]
            is_deprecated_encoder = any(
                self.encoder.name.startswith(pattern) for pattern in patterns
            )
            if is_deprecated_encoder:
                keys = list(state_dict.keys())
                for key in keys:
                    new_key = key
                    if key.startswith("encoder.") and not key.startswith("encoder.model."):
                        new_key = "encoder.model." + key.removeprefix("encoder.")
                    if "gernet" in self.encoder.name:
                        new_key = new_key.replace(".stages.", ".stages_")
                    state_dict[new_key] = state_dict.pop(key)

        # To be able to load weight with mismatched sizes
        # We are going to filter mismatched sizes as well if strict=False
        strict = kwargs.get("strict", True)
        if not strict:
            mismatched_keys = []
            model_state_dict = self.state_dict()
            common_keys = set(model_state_dict.keys()) & set(state_dict.keys())
            for key in common_keys:
                if model_state_dict[key].shape != state_dict[key].shape:
                    mismatched_keys.append(
                        (key, model_state_dict[key].shape, state_dict[key].shape)
                    )
                    state_dict.pop(key)

            if mismatched_keys:
                str_keys = "\n".join(
                    [f" - {key}: {s} (weights) -> {m} (model)" for key, m, s in mismatched_keys]
                )
                text = f"\n\n !!!!!! Mismatched keys !!!!!!\n\nYou should TRAIN the model to use it:\n{str_keys}\n"
                warnings.warn(text, stacklevel=-1)

        return super().load_state_dict(state_dict, **kwargs)

    def train(self, mode: bool = True):
        """Set the module in training mode.

        This method behaves like the standard :meth:`torch.nn.Module.train`,
        with one exception: if the encoder has been frozen via
        :meth:`freeze_encoder`, then its normalization layers are not affected
        by this call. In other words, calling ``model.train()`` will not
        re-enable updates to frozen encoder normalization layers
        (e.g., BatchNorm, InstanceNorm).

        To restore the encoder to normal training behavior, use
        :meth:`unfreeze_encoder`.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            # skip encoder if it is frozen
            if self._is_encoder_frozen and name == "encoder":
                continue
            module.train(mode)
        return self

    def _set_encoder_trainable(self, mode: bool):
        for param in self.encoder.parameters():
            param.requires_grad = mode

        for module in self.encoder.modules():
            # _NormBase is the common base of classes like _InstanceNorm
            # and _BatchNorm that track running stats
            if isinstance(module, torch.nn.modules.batchnorm._NormBase):
                module.train(mode)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(features)
        if self.isDeeply:
            out_masks = []
            for index, out_block in enumerate(self.segmentation_heads):
                x = out_block(decoder_output[index])
                out_masks.append(x)

            return out_masks[::-1]
        return self.segmentation_heads(decoder_output[-1])

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x

    def freeze_encoder(self):
        """
        Freeze encoder parameters and disable updates to normalization
        layer statistics.

        This method:
        - Sets ``requires_grad = False`` for all encoder parameters,
            preventing them from being updated during backpropagation.
        - Puts normalization layers that track running statistics
            (e.g., BatchNorm, InstanceNorm) into evaluation mode (``.eval()``),
            so their ``running_mean`` and ``running_var`` are no longer updated.
        """
        return self._set_encoder_trainable(False)

    def unfreeze_encoder(self):
        """
        Unfreeze encoder parameters and restore normalization layers to training mode.

        This method reverts the effect of :meth:`freeze_encoder`. Specifically:
        - Sets ``requires_grad=True`` for all encoder parameters.
        - Restores normalization layers (e.g. BatchNorm, InstanceNorm) to training mode,
        so their running statistics are updated again.
        """
        return self._set_encoder_trainable(True)
