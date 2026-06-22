import torch
import torch.nn.functional as F
from torch import nn

class DecisionDiffuser(nn.Module):
    def __init__(
        self,
        denoiser,
        diffusion,
        inverse_dynamics,
        horizon,
        observation_dim,
        action_dim,
        batch_size,
        device="cuda",
        *,
        returns_condition=False,
        condition_guidance_w=1.0,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.diffusion = diffusion
        self.inverse_dynamics = inverse_dynamics
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        self.device = device


    def condition_observations(self, conditions):
        def callback(x, timesteps):
            for step, value in conditions.items():
                x[:, step, :] = value.to(device=x.device, dtype=x.dtype)
            return x
        return callback

    def denoise(self, x, timesteps, *, returns=None):
        if not self.returns_condition:
            return self.denoiser(x, timesteps)

        cond = self.denoiser(
            x,
            timesteps,
            returns=returns,
            use_dropout=False,
        )
        uncond = self.denoiser(
            x,
            timesteps,
            returns=returns,
            force_dropout=True,
        )
        return cond + self.condition_guidance_w * (cond - uncond)

    def inverse_dynamics_loss(self, trajectories):
        observations = trajectories[:, :, self.action_dim:]
        actions = trajectories[:, :, :self.action_dim]

        x_t = observations[:, :-1]
        x_t_1 = observations[:, 1:]
        action_t = actions[:, :-1]

        state_pairs = torch.cat([x_t, x_t_1], dim=-1)
        state_pairs = state_pairs.reshape(-1, 2 * self.observation_dim)
        action_t = action_t.reshape(-1, self.action_dim)

        if hasattr(self.inverse_dynamics, "calc_loss"):
            return self.inverse_dynamics.calc_loss(state_pairs, action_t)

        predicted_actions = self.inverse_dynamics(state_pairs)
        return F.mse_loss(predicted_actions, action_t)

    def diffusion_loss(self, trajectories, conditions, returns=None):
        observations = trajectories[:, :, self.action_dim:]
        batch_size = observations.shape[0]
        timesteps = torch.randint(
            0,
            self.diffusion.n_timesteps,
            (batch_size,),
            device=observations.device,
            dtype=torch.long,
        )

        return self.diffusion.loss(
            self.denoise,
            observations,
            timesteps,
            conditioning_callback=self.condition_observations(conditions),
            returns=returns,
        )

    def loss(self, trajectories, conditions, returns=None):
        inverse_loss = self.inverse_dynamics_loss(trajectories)
        diffusion_loss = self.diffusion_loss(trajectories, conditions, returns)
        loss = 0.5 * (diffusion_loss + inverse_loss)

        return loss, {
            "loss": loss.detach(),
            "diffusion_loss": diffusion_loss.detach(),
            "inverse_dynamics_loss": inverse_loss.detach(),
        }

    def sample(self, conditions, *, returns=None, horizon=None):
        horizon = horizon or self.horizon
        batch_size = self.batch_size
        if conditions:
            first_condition = next(iter(conditions.values()))
            batch_size = first_condition.shape[0]

        return self.diffusion.sample_loop(
            self.denoise,
            shape=(batch_size, horizon, self.observation_dim),
            device=self.device,
            conditioning_callback=self.condition_observations(conditions),
            returns=returns,
        )

    def forward(self, conditions, *, returns=None, horizon=None):
        return self.sample(conditions, returns=returns, horizon=horizon)
