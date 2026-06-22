import torch
import torch.nn as nn


class InverseDynamics(nn.Module):
    def __init__(
        self, hidden_dim, observation_dim, action_dim, low_act=-1.0, up_act=1.0
    ):
        super(ARInvModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.action_embed_hid = 128
        self.out_lin = 128
        self.num_bins = 80

        self.up_act = up_act
        self.low_act = low_act
        self.bin_size = (self.up_act - self.low_act) / self.num_bins
        self.ce_loss = nn.CrossEntropyLoss()

        self.state_embed = nn.Sequential(
            nn.Linear(2 * self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lin_mod = nn.ModuleList(
            [nn.Linear(i, self.out_lin) for i in range(1, self.action_dim)]
        )
        self.act_mod = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, self.action_embed_hid),
                    nn.ReLU(),
                    nn.Linear(self.action_embed_hid, self.num_bins),
                )
            ]
        )

        for _ in range(1, self.action_dim):
            self.act_mod.append(
                nn.Sequential(
                    nn.Linear(hidden_dim + self.out_lin, self.action_embed_hid),
                    nn.ReLU(),
                    nn.Linear(self.action_embed_hid, self.num_bins),
                )
            )

    def forward(self, comb_state, deterministic=False):
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        lp_0 = self.act_mod[0](state_d)
        l_0 = torch.distributions.Categorical(logits=lp_0).sample()

        if deterministic:
            a_0 = self.low_act + (l_0 + 0.5) * self.bin_size
        else:
            a_0 = torch.distributions.Uniform(
                self.low_act + l_0 * self.bin_size,
                self.low_act + (l_0 + 1) * self.bin_size,
            ).sample()

        a = [a_0.unsqueeze(1)]

        for i in range(1, self.action_dim):
            lp_i = self.act_mod[i](
                torch.cat([state_d, self.lin_mod[i - 1](torch.cat(a, dim=1))], dim=1)
            )
            l_i = torch.distributions.Categorical(logits=lp_i).sample()

            if deterministic:
                a_i = self.low_act + (l_i + 0.5) * self.bin_size
            else:
                a_i = torch.distributions.Uniform(
                    self.low_act + l_i * self.bin_size,
                    self.low_act + (l_i + 1) * self.bin_size,
                ).sample()

            a.append(a_i.unsqueeze(1))

        return torch.cat(a, dim=1)

    def calc_loss(self, comb_state, action):
        eps = 1e-8
        action = torch.clamp(action, min=self.low_act + eps, max=self.up_act - eps)
        l_action = torch.div(
            (action - self.low_act), self.bin_size, rounding_mode="floor"
        ).long()
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        loss = self.ce_loss(self.act_mod[0](state_d), l_action[:, 0])

        for i in range(1, self.action_dim):
            loss += self.ce_loss(
                self.act_mod[i](
                    torch.cat([state_d, self.lin_mod[i - 1](action[:, :i])], dim=1)
                ),
                l_action[:, i],
            )

        return loss / self.action_dim
