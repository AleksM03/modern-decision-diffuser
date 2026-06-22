import torch
from torch import nn, Tensor
from typing import Any
from torch.distributions import Bernoulli
from einops.layers.torch import Rearrange
from einops import rearrange

from utils.model_helpers import SinusoidalPosEmb, Downsample1d, Upsample1d, Conv1dBlock


class ResidualTemporalBlock(nn.Module):

    def __init__(
        self,
        inp_channels,
        out_channels,
        embed_dim,
        horizon,
        kernel_size=5,
        activation=nn.Mish,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, activation),
            Conv1dBlock(out_channels, out_channels, kernel_size, activation),
        ])

        self.time_mlp = nn.Sequential(
            activation(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)

class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        returns_condition=False,
        condition_dropout=0.1,
        activation=nn.Mish,
        kernel_size=5,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            activation(),
            nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                        nn.Linear(1, dim),
                        activation(),
                        nn.Linear(dim, dim * 4),
                        activation(),
                        nn.Linear(dim * 4, dim),
                    )
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
            embed_dim = 2*dim
        else:
            embed_dim = dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, activation=activation),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, activation=activation),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, activation=activation)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, activation=activation)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, activation=activation),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, activation=activation),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, activation=activation),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        x = rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = rearrange(x, 'b t h -> b h t')

        return x

    def get_pred(self, x, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        x = rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = rearrange(x, 'b t h -> b h t')

        return x

class EnergyWrapper(nn.Module):
    """
    Wrap a denoiser so it returns an input gradient instead of its raw output.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        **model_kwargs: Any,
    ) -> Tensor:
        with torch.enable_grad():
            x_inp = x.detach().requires_grad_(True)

            prediction = self.model(
                x_inp,
                timesteps,
                **model_kwargs,
            )

            energy = ((prediction - x_inp) ** 2).mean()
            (gradient,) = torch.autograd.grad(
                outputs=energy,
                inputs=x_inp,
                create_graph=self.training,
            )

        return gradient
