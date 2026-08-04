"""Signal-conditioned RSSM world model + baselines for CAV trajectory prediction.

State s_t:  [y_rel, v_f, a_f, gap, dv]  (5D, normalized)
Action a_t: [v_lead, a_lead]            (2D, normalized, proxy for CAV action)
Signal φ_t: [G, Y, R]                   (3D one-hot, exogenous SPaT)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def gaussian_kl(mu1, logs1, mu2, logs2):
    """KL(N(mu1,s1) || N(mu2,s2)) per element, summed over last dim."""
    v1 = 2 * logs1
    v2 = 2 * logs2
    kl = 0.5 * (v2 - v1 + (torch.exp(v1) + (mu1 - mu2) ** 2) / torch.exp(v2) - 1)
    return kl.sum(-1)


class MLP(nn.Module):
    def __init__(self, dims, act=F.elu):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RSSM(nn.Module):
    """Signal-conditioned Recurrent State-Space Model.

    h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}, φ_{t-1}])   (signal enters transition)
    z_t ~ q(z|h_t, s_t) (posterior) | p(z|h_t) (prior)
    ŝ_{t+1} = dec(h_t, z_t, a_t, φ_t)                   (signal enters decoder)
    """

    def __init__(self, s_dim=5, a_dim=2, phi_dim=3, h_dim=64, z_dim=16, hidden=128):
        super().__init__()
        self.h_dim, self.z_dim = h_dim, z_dim
        self.gru = nn.GRUCell(z_dim + a_dim + phi_dim, h_dim)
        self.prior = MLP([h_dim, hidden, 2 * z_dim])
        self.posterior = MLP([h_dim + s_dim, hidden, 2 * z_dim])
        self.decoder = MLP([h_dim + z_dim + a_dim + phi_dim, hidden, hidden, s_dim])

    def _split(self, x):
        return x.chunk(2, dim=-1)

    def forward(self, s_seq, a_seq, phi_seq, mask):
        """Teacher-forced forward. Returns preds (B,T,s_dim), kl (B,T)."""
        B, T, _ = s_seq.shape
        h = s_seq.new_zeros(B, self.h_dim)
        z = s_seq.new_zeros(B, self.z_dim)
        preds, kls = [], []
        for t in range(T):
            inp = torch.cat([z, a_seq[:, t], phi_seq[:, t]], dim=-1)
            h = self.gru(inp, h)
            post_mu, post_ls = self._split(self.posterior(torch.cat([h, s_seq[:, t]], -1)))
            pri_mu, pri_ls = self._split(self.prior(h))
            z = post_mu + torch.randn_like(post_mu) * torch.exp(post_ls)
            kls.append(gaussian_kl(post_mu, post_ls, pri_mu, pri_ls))
            dec_inp = torch.cat([h, z, a_seq[:, t], phi_seq[:, t]], -1)
            preds.append(self.decoder(dec_inp))
        return torch.stack(preds, 1), torch.stack(kls, 1)

    @torch.no_grad()
    def rollout(self, s0, a_seq, phi_seq):
        """Open-loop multi-step rollout using prior only (except t=0 posterior)."""
        B, T, _ = a_seq.shape
        h = s0.new_zeros(B, self.h_dim)
        # t=0: use posterior since we have s0
        post_mu, post_ls = self._split(self.posterior(torch.cat([h, s0], -1)))
        z = post_mu
        preds = []
        for t in range(T):
            inp = torch.cat([z, a_seq[:, t], phi_seq[:, t]], -1)
            h = self.gru(inp, h)
            pri_mu, pri_ls = self._split(self.prior(h))
            z = pri_mu  # deterministic: use mean
            dec_inp = torch.cat([h, z, a_seq[:, t], phi_seq[:, t]], -1)
            preds.append(self.decoder(dec_inp))
        return torch.stack(preds, 1)


class LSTMBase(nn.Module):
    """Plain LSTM baseline. signal=True uses φ, False ablates it."""

    def __init__(self, s_dim=5, a_dim=2, phi_dim=3, hidden=64, signal=True):
        super().__init__()
        self.signal = signal
        inp_dim = s_dim + a_dim + (phi_dim if signal else 0)
        self.lstm = nn.LSTM(inp_dim, hidden, batch_first=True)
        self.head = MLP([hidden, hidden, s_dim])

    def forward(self, s_seq, a_seq, phi_seq, mask=None):
        if self.signal:
            x = torch.cat([s_seq, a_seq, phi_seq], -1)
        else:
            x = torch.cat([s_seq, a_seq], -1)
        h, _ = self.lstm(x)
        return self.head(h)  # (B,T,s_dim) — predicts s_{t+1} from s_t

    @torch.no_grad()
    def rollout(self, s0, a_seq, phi_seq):
        B, T, _ = a_seq.shape
        s = s0.unsqueeze(1)
        h = None
        preds = []
        for t in range(T):
            if self.signal:
                x = torch.cat([s, a_seq[:, t:t + 1], phi_seq[:, t:t + 1]], -1)
            else:
                x = torch.cat([s, a_seq[:, t:t + 1]], -1)
            out, h = self.lstm(x, h)
            s_next = self.head(out).squeeze(1)  # (B,s_dim)
            preds.append(s_next)
            s = s_next.unsqueeze(1)
        return torch.stack(preds, 1)  # (B,T,s_dim)


class IDM:
    """Intelligent Driver Model baseline (calibrated, no signal)."""

    def __init__(self):
        self.v0 = 12.0   # desired speed m/s
        self.T = 1.5     # time headway s
        self.a = 1.5     # max accel m/s2
        self.b = 2.0     # comfortable decel
        self.s0 = 2.0    # min gap m
        self.delta = 4.0

    def fit(self, vf, gap, dvl, af_true, dt=0.1):
        """Grid-search calibrate on training data (unnormalized m/s)."""
        best = (1e9, None)
        for v0 in [8, 10, 12, 14]:
            for T in [0.8, 1.0, 1.5, 2.0]:
                for a in [1.0, 1.5, 2.0]:
                    for b in [1.5, 2.0, 3.0]:
                        for s0 in [1.0, 2.0, 3.0]:
                            s_star = s0 + np.maximum(0, vf * T + vf * dvl / (2 * np.sqrt(a * b)))
                            a_idm = a * (1 - (vf / v0) ** 4 - (s_star / np.maximum(gap, 0.1)) ** 2)
                            err = np.mean((a_idm - af_true) ** 2)
                            if err < best[0]:
                                best = (err, (v0, T, a, b, s0))
        self.v0, self.T, self.a, self.b, self.s0 = best[1]

    def step(self, vf, gap, dvl, dt=0.1):
        s_star = self.s0 + np.maximum(0, vf * self.T + vf * dvl / (2 * np.sqrt(self.a * self.b)))
        a_idm = self.a * (1 - (vf / self.v0) ** 4 - (s_star / np.maximum(gap, 0.1)) ** 2)
        return np.clip(a_idm, -6, 4)
