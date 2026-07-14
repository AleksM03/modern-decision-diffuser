import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
import math


def cosine_beta_schedule(
    timesteps: int, s: float = 0.008, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    taken from https://github.com/anuragajay/decision-diffuser/blob/main/code/diffuser/models/helpers.py#L80
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod: np.ndarray = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    x_shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    """
    Select one schedule value for each batch element and reshape it
    for broadcasting over x.

    values:      [T]
    timesteps:   [B]
    return:      [B, 1, ..., 1]
    """
    batch_size = timesteps.shape[0]

    selected = values.gather(0, timesteps.to(values.device),)

    return selected.reshape(batch_size, *((1,) * (len(x_shape) - 1)),)


def random_noise_like(
    tensor: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate standard Gaussian noise matching a tensor's shape,
    device and dtype.
    """
    return torch.randn(
        tensor.shape,
        device=tensor.device,
        dtype=tensor.dtype,
        generator=generator,
    )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> activation
    '''

    def __init__(
        self,
        inp_channels,
        out_channels,
        kernel_size,
        activation=nn.Mish,
        n_groups=8,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            activation(),
        )

    def forward(self, x):
        return self.block(x)

class EMA():
    '''
        exponential moving average
    '''
    def __init__(self, beta, step_start=2000):
        super().__init__()
        self.beta = beta
        self.step_start = step_start

    def update_model_average(self, ma_model, current_model):
        current = current_model._orig_mod if hasattr(current_model, "_orig_mod") else current_model
        for current_params, ma_params in zip(current.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def reset_parameters(self, current_model, ma_model):
        current = current_model._orig_mod if hasattr(current_model, "_orig_mod") else current_model
        ma_model.load_state_dict(current.state_dict())

    def step_ema(self, current_model, ma_model, step):
        if step < self.step_start:
            self.reset_parameters(current_model, ma_model)
            return
        self.update_model_average(ma_model, current_model)

