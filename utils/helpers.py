import torch
import numpy as np


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