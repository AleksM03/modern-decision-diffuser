from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.helpers import cosine_beta_schedule, extract


Denoiser = Callable[..., Tensor]
StepCallback = Callable[[Tensor, int], Tensor]


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        n_timesteps: int = 1000,
        *,
        clip_denoised: bool = False,
        predict_epsilon: bool = True,
    ) -> None:
        super().__init__()

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = n_timesteps
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.betas = nn.Buffer(betas)

        self.sqrt_alphas_cumprod = nn.Buffer(
            torch.sqrt(alphas_cumprod)
        )

        self.sqrt_one_minus_alphas_cumprod = nn.Buffer(
            torch.sqrt(1.0 - alphas_cumprod)
        )

        self.sqrt_recip_alphas_cumprod = nn.Buffer(
            torch.sqrt(1.0 / alphas_cumprod)
        )

        self.sqrt_recipm1_alphas_cumprod = nn.Buffer(
            torch.sqrt(
                1.0 / alphas_cumprod - 1.0
            )
        )

        posterior_variance = (
            betas
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - alphas_cumprod)
        )

        self.posterior_variance = nn.Buffer(
            posterior_variance
        )

        # self.posterior_log_variance = nn.Buffer(
        #     torch.log(
        #         posterior_variance.clamp(min=1e-20)
        #     )
        # )

        # self.posterior_mean_coef1 = nn.Buffer(
        #     (
        #         betas
        #         * torch.sqrt(alphas_cumprod_prev)
        #         / (1.0 - alphas_cumprod)
        #     )
        # )

        # self.posterior_mean_coef2 = nn.Buffer(
        #     (
        #         (1.0 - alphas_cumprod_prev)
        #         * torch.sqrt(alphas)
        #         / (1.0 - alphas_cumprod)
        #     )
        # )

    def forward_sample(
        self,
        x_0: Tensor,
        timesteps: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, timesteps, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_0.shape)

        return (
            sqrt_alphas_cumprod_t * x_0
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def predict_start_from_noise(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        model_output: Tensor,
    ) -> Tensor:
        if not self.predict_epsilon:
            return model_output

        return (
            extract(
                self.sqrt_recip_alphas_cumprod,
                timesteps,
                x_t.shape,
            )
            * x_t
            - extract(
                self.sqrt_recipm1_alphas_cumprod,
                timesteps,
                x_t.shape,
            )
            * model_output
        )

    def q_posterior(
        self,
        x_0: Tensor,
        x_t: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        posterior_mean = (
            extract(
                self.posterior_mean_coef1,
                timesteps,
                x_t.shape,
            )
            * x_0
            + extract(
                self.posterior_mean_coef2,
                timesteps,
                x_t.shape,
            )
            * x_t
        )

        posterior_variance = extract(
            self.posterior_variance,
            timesteps,
            x_t.shape,
        )

        posterior_log_variance = extract(
            self.posterior_log_variance,
            timesteps,
            x_t.shape,
        )

        return (
            posterior_mean,
            posterior_variance,
            posterior_log_variance,
        )

    def p_mean_variance(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        model_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        x_0 = self.predict_start_from_noise(
            x_t=x_t,
            timesteps=timesteps,
            model_output=model_output,
        )

        if self.clip_denoised:
            x_0 = x_0.clamp(-1.0, 1.0)

        return self.q_posterior(
            x_0=x_0,
            x_t=x_t,
            timesteps=timesteps,
        )

    @torch.no_grad()
    def sample(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        model_output: Tensor,
        *,
        noise_scale: float = 0.5,
    ) -> Tensor:
        model_mean, _, model_log_variance = (
            self.p_mean_variance(
                x_t=x_t,
                timesteps=timesteps,
                model_output=model_output,
            )
        )

        noise = noise_scale * torch.randn_like(x_t)

        nonzero_mask = (
            timesteps != 0
        ).to(x_t.dtype).reshape(
            x_t.shape[0],
            *((1,) * (x_t.ndim - 1)),
        )

        return (
            model_mean
            + nonzero_mask
            * torch.exp(0.5 * model_log_variance)
            * noise
        )

    @torch.no_grad()
    def sample_loop(
        self,
        denoiser: Denoiser,
        shape: tuple[int, ...],
        *,
        device: torch.device | str,
        initial_noise_scale: float = 0.5,
        noise_scale: float = 0.5,
        step_callback: StepCallback | None = None,
        **model_kwargs: Any,
    ) -> Tensor:
        x = initial_noise_scale * torch.randn(
            shape,
            device=device,
        )

        if step_callback is not None:
            x = step_callback(
                x,
                self.n_timesteps,
            )

        batch_size = shape[0]

        for step in reversed(range(self.n_timesteps)):
            timesteps = torch.full(
                (batch_size,),
                step,
                device=device,
                dtype=torch.long,
            )

            model_output = denoiser(
                x,
                timesteps,
                **model_kwargs,
            )

            x = self.p_sample(
                x_t=x,
                timesteps=timesteps,
                model_output=model_output,
                noise_scale=noise_scale,
            )

            if step_callback is not None:
                x = step_callback(x, step)

        return x

    def loss(
        self,
        denoiser: Denoiser,
        x_0: Tensor,
        timesteps: Tensor,
        *,
        noise: Tensor | None = None,
        reduction: str = "mean",
        **model_kwargs: Any,
    ) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)

        x_noisy = self.forward_sample(
            x_0=x_0,
            timesteps=timesteps,
            noise=noise,
        )

        model_output = denoiser(
            x_noisy,
            timesteps,
            **model_kwargs,
        )

        target = (
            noise
            if self.predict_epsilon
            else x_0
        )

        return F.mse_loss(
            model_output,
            target,
            reduction=reduction,
        )