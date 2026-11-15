"""Variational latent Neural ODE with backward GRU encoder.

Baseline follows Chen et al. (2018) in spirit, not a byte-for-byte recreation.
Conditioned version augments the ODE state with a constant trajectory-specific
conditioning code, permitting gradients to flow through its initial value.
"""
import torch
from torch import nn
from .solver import integrate


class RecognitionGRU(nn.Module):
    def __init__(self, latent_dim=4, hidden_dim=25, cond_dim=0):
        super().__init__()
        self.latent_dim, self.cond_dim = latent_dim, cond_dim
        self.gru = nn.GRU(input_size=4, hidden_size=hidden_dim, batch_first=True)
        self.posterior = nn.Linear(hidden_dim, 2 * latent_dim)
        self.cond = nn.Linear(hidden_dim, cond_dim) if cond_dim else None

    def forward(self, observations, times, total_time):
        if observations.ndim != 3 or observations.shape[-1] != 2:
            raise ValueError('observations must have [batch, n_obs, 2] shape')
        if times.shape != observations.shape[:2]:
            raise ValueError('Each observed point must have its timestamp')
        rev_xy = torch.flip(observations, [1])
        rev_t = torch.flip(times, [1])
        gap = torch.zeros_like(rev_t)
        gap[:, 1:] = (rev_t[:, :-1] - rev_t[:, 1:])
        feats = torch.cat([rev_xy, (rev_t / total_time).unsqueeze(-1),
                           (gap / total_time).unsqueeze(-1)], dim=-1)
        _, hidden = self.gru(feats)
        stats = self.posterior(hidden[-1])
        mu, logvar = torch.chunk(stats, 2, dim=-1)
        logvar = logvar.clamp(-8, 5)
        cond = torch.tanh(self.cond(hidden[-1])) if self.cond is not None else None
        return mu, logvar, cond


def mlp(in_dim, hidden_dim, out_dim, n_hidden_layers=1, activation=nn.Tanh):
    """MLP with `n_hidden_layers` hidden layers. n_hidden_layers=1 reproduces the
    original Linear-Tanh-Linear layout (identical state_dict keys)."""
    layers, d = [], in_dim
    for _ in range(max(1, int(n_hidden_layers))):
        layers += [nn.Linear(d, hidden_dim), activation()]
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class LatentDynamics(nn.Module):
    def __init__(self, latent_dim=4, hidden_dim=20, cond_dim=0, n_layers=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.nfe = 0
        self.network = mlp(latent_dim + cond_dim, hidden_dim, latent_dim, n_layers)

    def forward(self, t, full_state):
        self.nfe += 1
        dz = self.network(full_state)
        if self.cond_dim:
            zeros = torch.zeros_like(full_state[..., self.latent_dim:])
            return torch.cat([dz, zeros], dim=-1)
        return dz


class LatentODE(nn.Module):
    def __init__(self, latent_dim=4, encoder_hidden=25, dynamics_hidden=20,
                 decoder_hidden=20, cond_dim=0, solver='dopri5', adjoint=False,
                 dynamics_layers=1, decoder_layers=1, rtol=1e-4, atol=1e-5,
                 step_size=None):
        super().__init__()
        self.latent_dim, self.cond_dim = latent_dim, cond_dim
        self.solver, self.adjoint = solver, adjoint
        self.rtol, self.atol, self.step_size = rtol, atol, step_size
        self.encoder = RecognitionGRU(latent_dim, encoder_hidden, cond_dim)
        self.dynamics = LatentDynamics(latent_dim, dynamics_hidden, cond_dim, dynamics_layers)
        self.decoder = mlp(latent_dim, decoder_hidden, 2, decoder_layers)

    def forward(self, obs_xy, obs_t, query_t, sample=None):
        if sample is None:
            sample = self.training
        mu, logvar, cond = self.encoder(obs_xy, obs_t, query_t[-1].clamp_min(1e-6))
        z0 = mu + torch.randn_like(mu) * torch.exp(.5 * logvar) if sample else mu
        full_state = torch.cat([z0, cond], -1) if cond is not None else z0
        zt = integrate(self.dynamics, full_state, query_t, method=self.solver,
                       adjoint=self.adjoint, rtol=self.rtol, atol=self.atol,
                       step_size=self.step_size)
        pred = self.decoder(zt[..., :self.latent_dim]).transpose(0, 1)
        return {'prediction': pred, 'mu': mu, 'logvar': logvar, 'condition': cond}


def kl_standard_normal(mu, logvar):
    return .5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar, dim=-1).mean()
