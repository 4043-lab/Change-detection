import os, glob
from dataclasses import dataclass, asdict
from typing import Any, List, Optional, Tuple, Union

from PIL import Image

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as fn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class DownConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 4,
        stride: int = 2,
        padding: int = 1,
        activation: str = "leaky_relu",
        do_batch_norm: bool = True,
        num_groups: int = 32,
        negative_slope = 0.2,
    ):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups, in_channels) if do_batch_norm else None
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel, stride=stride, padding=padding
        )

        activation_lower = activation.lower()

        if activation_lower == "leaky_relu":
            self.act_fn = nn.LeakyReLU(negative_slope=negative_slope)
        elif activation_lower == "identity":
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError(f"`activation` must be `leaky_relu` or `identity`")

    def forward(self, x: Tensor) -> Tensor:
        if self.norm:
            x = self.norm(x)
        x = self.conv(x)
        x = self.act_fn(x)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        down_out_channels: Tuple[int],
        kernels: Union[int, Tuple[int]],
        strides: Union[int, Tuple[int]],
        paddings: Union[int, Tuple[int]],
        do_batch_norms: Union[bool, Tuple[bool]],
        activations: Union[str, Tuple[str]],
    ):
        super().__init__()

        # check inputs
        num_blocks = len(down_out_channels)
        if not isinstance(kernels, int) and len(kernels) != num_blocks:
            raise ValueError("`kernels` must have the same length as `down_out_channels`")
        if not isinstance(strides, int) and len(strides) != num_blocks:
            raise ValueError("`strides` must have the same length as `down_out_channels`")
        if not isinstance(paddings, int) and len(paddings) != num_blocks:
            raise ValueError("`paddings` must have the same length as `down_out_channels`")
        if not isinstance(do_batch_norms, bool) and len(do_batch_norms) != num_blocks:
            raise ValueError("`do_batch_norms` must have the same length as `down_out_channels`")
        if not isinstance(activations, str) and len(activations) != num_blocks:
            raise ValueError("`activations` must have the same length as `down_out_channels`")

        if isinstance(kernels, int):
            kernels = (kernels,) * num_blocks
        if isinstance(strides, int):
            strides = (strides,) * num_blocks
        if isinstance(paddings, int):
            paddings = (paddings,) * num_blocks
        if isinstance(do_batch_norms, bool):
            do_batch_norms = (do_batch_norms,) * num_blocks
        if isinstance(activations, str):
            activations = (activations,) * num_blocks

        self.down_blocks = nn.Sequential()
        for i in range(num_blocks):
            out_channels = down_out_channels[i]
            self.down_blocks.append(
                DownConv(
                    in_channels,
                    out_channels,
                    kernel=kernels[i],
                    stride=strides[i],
                    padding=paddings[i],
                    activation=activations[i],
                    do_batch_norm=do_batch_norms[i],
                )
            )
            in_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.down_blocks(x)
        return x

class UpConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 4,
        stride: int = 2,
        padding: int = 1,
        activation: str = "relu",
        do_batch_norm: bool = True,
        num_groups: int = 32,
    ):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups, in_channels) if do_batch_norm else None
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel, stride=stride, padding=padding
        )

        activation_lower = activation.lower()

        if activation_lower == "relu":
            self.act_fn = nn.ReLU()
        elif activation_lower == "tanh":
            self.act_fn = nn.Tanh()
        else:
            raise NotImplementedError(f"`activation` must be `relu` or `tanh`")

    def forward(self, x: Tensor) -> Tensor:
        if self.norm:
            x = self.norm(x)
        x = self.conv(x)
        x = self.act_fn(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        up_out_channels: Tuple[int],
        kernels: Union[int, Tuple[int]],
        strides: Union[int, Tuple[int]],
        paddings: Union[int, Tuple[int]],
        do_batch_norms: Union[bool, Tuple[bool]],
        activations: Union[str, Tuple[str]],
    ):
        super().__init__()

        # check inputs
        num_blocks = len(up_out_channels)
        if not isinstance(kernels, int) and len(kernels) != num_blocks:
            raise ValueError("`kernels` must have the same length as `up_out_channels`")
        if not isinstance(strides, int) and len(strides) != num_blocks:
            raise ValueError("`strides` must have the same length as `up_out_channels`")
        if not isinstance(paddings, int) and len(paddings) != num_blocks:
            raise ValueError("`paddings` must have the same length as `up_out_channels`")
        if not isinstance(do_batch_norms, int) and len(do_batch_norms) != num_blocks:
            raise ValueError("`do_batch_norms` must have the same length as `up_out_channels`")
        if not isinstance(activations, int) and len(activations) != num_blocks:
            raise ValueError("`activations` must have the same length as `up_out_channels`")

        if isinstance(kernels, int):
            kernels = (kernels,) * num_blocks
        if isinstance(strides, int):
            strides = (strides,) * num_blocks
        if isinstance(paddings, int):
            paddings = (paddings,) * num_blocks
        if isinstance(do_batch_norms, int):
            do_batch_norms = (do_batch_norms,) * num_blocks
        if isinstance(activations, int):
            activations = (activations,) * num_blocks

        self.up_blocks = nn.Sequential()
        for i in range(num_blocks):
            out_channels = up_out_channels[i]
            self.up_blocks.append(
                UpConv(
                    in_channels,
                    out_channels,
                    kernel=kernels[i],
                    stride=strides[i],
                    padding=paddings[i],
                    activation=activations[i],
                    do_batch_norm=do_batch_norms[i],
                )
            )
            in_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.up_blocks(x)
        return x

@dataclass
class EncoderConfig:
    in_channels: int = 3
    down_out_channels: Tuple[int] = (64, 128, 256, 512, 512)
    kernels: Union[int, Tuple[int]] = 4
    strides: Union[int, Tuple[int]] = (2, 2, 2, 2, 1)
    paddings: Union[int, Tuple[int]] = (1, 1, 1, 1, 0)
    do_batch_norms: Union[bool, Tuple[bool]] = (False, True, True, True, False)
    activations: Union[str, Tuple[str]] = (
        "leaky_relu",
        "leaky_relu",
        "leaky_relu",
        "leaky_relu",
        "identity",
    )

@dataclass
class DecoderConfig:
    in_channels: int = 512
    up_out_channels: Tuple[int] = (512, 256, 128, 64, 3)
    kernels: Union[int, Tuple[int]] = 4
    strides: Union[int, Tuple[int]] = (1, 2, 2, 2, 2)
    paddings: Union[int, Tuple[int]] = (0, 1, 1, 1, 1)
    do_batch_norms: Union[bool, Tuple[bool]] = (True, True, True, True, False)
    activations: Union[str, Tuple[str]] = (
        "relu",
        "relu",
        "relu",
        "relu",
        "tanh",
    )

class GeneratorNet(nn.Module):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: DecoderConfig,
        do_init: bool = True,
        init_mean: float = 0.0,
        init_std: float = 0.02,
        init_const: float = 0.0,
    ):
        super().__init__()

        if decoder_config.in_channels != encoder_config.down_out_channels[-1]:
            raise ValueError(
                "`in_channels` for decoder must be the same as the last element of `down_out_channels` for encoder"
            )

        self.encoder = Encoder(**asdict(encoder_config))
        self.decoder = Decoder(**asdict(decoder_config))

        if do_init:
            self._initialize_params(mean=init_mean, std=init_std, const=init_const)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def _initialize_params(self, mean: float, std: float, const: float):
        def init_params(module: nn.Module):
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.normal_(module.weight, mean=mean, std=std)
                torch.nn.init.constant_(module.bias, val=const)

        self.apply(init_params)

def collate_fn(image_batch: List[Tensor]) -> Tensor:
    pixel_values = torch.stack(image_batch)
    return pixel_values.to(memory_format=torch.contiguous_format).float()


def get_freq_means_and_stds(x: Tensor) -> Tuple[Tensor]:
    freq = torch.fft.fft2(x)
    real_mean = freq.real.mean(dim=0)
    real_std = freq.real.std(dim=0)
    imag_mean = freq.imag.mean(dim=0)
    imag_std = freq.imag.std(dim=0)
    return real_mean, real_std, imag_mean, imag_std

def get_noise(
    real_mean: Tensor,
    real_std: Tensor,
    imag_mean: Tensor,
    imag_std: Tensor,
) -> Tensor:
    freq_real = torch.normal(real_mean, real_std)
    freq_imag = torch.normal(imag_mean, imag_std)
    freq = freq_real + 1j * freq_imag
    noise = torch.fft.ifft2(freq)
    return noise.real

if __name__ == '__main__':
    enc_config = EncoderConfig()
    dec_config = DecoderConfig()
    net = GeneratorNet(enc_config, dec_config).cuda()
    print(net)
    # def getModelSize(model):
    #     param_size = 0
    #     param_sum = 0
    #     for param in model.parameters():
    #         param_size += param.nelement() * param.element_size()
    #         param_sum += param.nelement()
    #     buffer_size = 0
    #     buffer_sum = 0
    #     for buffer in model.buffers():
    #         buffer_size += buffer.nelement() * buffer.element_size()
    #         buffer_sum += buffer.nelement()
    #     all_size = (param_size + buffer_size) / 1024 / 1024
    #     print('模型总大小为：{:.3f}MB'.format(all_size))
    #     print('模型参数量为：{:.3f}M'.format(param_sum/1e6))
    #     # return (param_size, param_sum, buffer_size, buffer_sum, all_size)
    # getModelSize(net)