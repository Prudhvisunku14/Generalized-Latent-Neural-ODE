"""Planar normalizing flow and continuous normalizing flow building blocks."""

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint

LOG_2PI = math.log(2.0 * math.pi)


def diagonal_gaussian_logprob(z, mean, log_std):
    return (-0.5 * (((z - mean) / torch.exp(log_std)) ** 2 + 2.0 * log_std + LOG_2PI)).sum(-1)


class PlanarLayer(nn.Module):
    """Invertible planar layer z -> z + u_hat tanh(w^T z + b)."""

    def __init__(self, dim=2):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim) * 0.05)
        self.w = nn.Parameter(torch.randn(dim) * 0.05)
        self.b = nn.Parameter(torch.zeros(()))

    def u_hat(self):
        wtu = torch.dot(self.w, self.u)
        m = -1.0 + F.softplus(wtu)
        return self.u + ((m - wtu) * self.w / (torch.dot(self.w, self.w) + 1e-8))

    def forward(self, z):
        u = self.u_hat()
        inner = z @ self.w + self.b
        h = torch.tanh(inner)
        out = z + h[:, None] * u[None, :]
        psi = (1.0 - h.square())[:, None] * self.w[None, :]
        det = 1.0 + psi @ u[:, None]
        logabsdet = torch.log(det.abs().squeeze(-1) + 1e-8)
        return out, logabsdet

    @torch.no_grad()
    def inverse(self, y, iters=60):
        """Exact inverse of z -> z + u_hat tanh(w^T z + b).

        With s = w^T z + b and c = w^T u_hat > -1 the scalar map s -> s + c tanh(s) is strictly
        increasing, and |c tanh(s)| <= |c|, so the root lies in [d-|c|, d+|c|] (d = w^T y + b).
        Bisection is used instead of Newton because it cannot diverge when c -> -1.
        """
        u = self.u_hat()
        c = torch.dot(self.w, u)
        d = y @ self.w + self.b
        lo, hi = d - c.abs(), d + c.abs()
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            g = mid + c * torch.tanh(mid) - d
            pos = g > 0
            hi = torch.where(pos, mid, hi)
            lo = torch.where(pos, lo, mid)
        s = 0.5 * (lo + hi)
        return y - torch.tanh(s)[:, None] * u[None, :]


class PlanarDensityFlow(nn.Module):
    """Base -> target planar flow used for KL(q||p) density matching (Figure 4)."""

    def __init__(self, n_layers, dim=2):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(dim))
        self.log_std = nn.Parameter(torch.zeros(dim))
        self.layers = nn.ModuleList([PlanarLayer(dim) for _ in range(n_layers)])

    def sample_and_logq(self, n):
        eps = torch.randn(n, self.mean.numel(), device=self.mean.device)
        z = self.mean + torch.exp(self.log_std) * eps
        logq = diagonal_gaussian_logprob(z, self.mean, self.log_std)
        for layer in self.layers:
            z, logdet = layer(z)
            logq = logq - logdet
        return z, logq

    @torch.no_grad()
    def sample(self, n):
        return self.sample_and_logq(n)[0]

    @torch.no_grad()
    def log_prob(self, x):
        """Exact log q(x): invert the K planar layers (bisection) and subtract the log-dets."""
        z, total = x, torch.zeros(x.shape[0], device=x.device)
        for layer in reversed(self.layers):
            z_prev = layer.inverse(z)
            total = total + layer(z_prev)[1]
            z = z_prev
        return diagonal_gaussian_logprob(z, self.mean, self.log_std) - total


class PlanarMLEFlow(nn.Module):
    """Data -> Gaussian planar flow used for maximum-likelihood training (Figure 5).

    Generation applies the exact same learned bijections in reverse using a stable
    one-dimensional Newton solve for each planar layer.
    """

    def __init__(self, n_layers=64, dim=2, base_variance=0.1):
        super().__init__()
        self.layers = nn.ModuleList([PlanarLayer(dim) for _ in range(n_layers)])
        self.register_buffer("base_mean", torch.zeros(dim))
        self.register_buffer("base_log_std", torch.ones(dim) * (0.5 * math.log(base_variance)))

    def log_prob(self, x):
        z = x
        logdet_sum = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            z, logdet = layer(z)
            logdet_sum = logdet_sum + logdet
        return diagonal_gaussian_logprob(z, self.base_mean, self.base_log_std) + logdet_sum

    @torch.no_grad()
    def base_sample(self, n):
        eps = torch.randn(n, self.base_mean.numel(), device=self.base_mean.device)
        return self.base_mean + torch.exp(self.base_log_std) * eps

    @torch.no_grad()
    def sample_stages(self, n, fractions: Iterable[float]):
        z = self.base_sample(n)
        k = len(self.layers)
        requested = [self.stage_layers(f) for f in fractions]
        result = []
        applied = 0
        # Generative direction is inverse layers from last to first.
        for layer in reversed(self.layers):
            z = layer.inverse(z)
            applied += 1
            while len(result) < len(requested) and applied >= requested[len(result)]:
                result.append(z.clone())
        while len(result) < len(requested):
            result.append(z.clone())
        return result


    def stage_layers(self, fraction):
        """Number of inverse layers applied after `fraction` of the generative direction."""
        k = len(self.layers)
        return max(1, min(k, int(round(float(fraction) * k))))

    @torch.no_grad()
    def stage_density(self, grid, fractions):
        """Exact density of the intermediate generative state (paper Fig. 5, 'NF' row).

        After inverting the last n layers, the state is y = f_{K-n}...f_1(x).  Its density is
        p(y) = p_base(f_K...f_{K-n+1}(y)) * prod_{i>K-n} |det J_i|, evaluated by pushing the
        grid forward through layers K-n+1..K.
        """
        out = []
        k = len(self.layers)
        for f in fractions:
            n = self.stage_layers(f)
            y = grid
            logdet_sum = torch.zeros(grid.shape[0], device=grid.device)
            for layer in self.layers[k - n :]:
                y, ld = layer(y)
                logdet_sum = logdet_sum + ld
            out.append((diagonal_gaussian_logprob(y, self.base_mean, self.base_log_std) + logdet_sum).exp())
        return out


class HyperNetwork(nn.Module):
    """Time-dependent hypernetwork producing the parameters of M planar units at time t.

    Paper (Sec. 4, "Time-dependent dynamics"): parameters are a function of t
    (hypernetwork, Ha et al. 2016) and every hidden unit n has a scalar gate
    sigma_n(t) in (0,1).  Per unit: W_n in R^dim, U_n in R^dim, B_n in R, G_n in R.
    """

    def __init__(self, dim=2, hidden_dim=32, width=64):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.width = width
        block = width * dim
        self.block = block
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # outputs: W (M*dim), U (M*dim), B (M), G (M)
        self.fc3 = nn.Linear(hidden_dim, 2 * block + 2 * width)
        # ASSUMPTION: not specified in the paper.  Eq. (10) *sums* the M units, so the
        # initial velocity would scale with M; we shrink the last layer by 1/M so that
        # every width starts with a comparable, gentle flow.
        with torch.no_grad():
            self.fc3.weight.mul_(1.0 / width)
            self.fc3.bias.mul_(1.0 / width)

    def forward(self, t):
        p = t.reshape(1, 1)
        p = torch.tanh(self.fc1(p))
        p = torch.tanh(self.fc2(p))
        p = self.fc3(p).reshape(-1)
        M, D = self.width, self.dim
        W = p[: self.block].reshape(M, D)
        U = p[self.block : 2 * self.block].reshape(M, D)
        B = p[2 * self.block : 2 * self.block + M]
        gate = torch.sigmoid(p[2 * self.block + M :])  # sigma_n(t) in (0, 1)
        return W, B, U, gate


class CNFDynamics(nn.Module):
    """Planar CNF:  dz/dt = sum_n sigma_n(t) u_n(t) tanh(w_n(t)^T z + b_n(t))   (eq. 9-10)

    d log p(z(t))/dt = -tr(df/dz) = - sum_n sigma_n u_n^T w_n (1 - tanh^2(w_n^T z + b_n)).
    The trace of each planar unit is exact and closed-form, so the cost is linear in M.
    `trace_mode="autograd"` recomputes the same trace with autograd (used in tests).
    """

    def __init__(self, dim=2, hidden_dim=32, width=64, trace_mode="analytic"):
        super().__init__()
        self.dim = dim
        self.width = width
        self.trace_mode = trace_mode
        self.hyper = HyperNetwork(dim, hidden_dim, width)
        self.nfe = 0

    def velocity_and_trace(self, t, z):
        W, B, U, gate = self.hyper(t)
        a = torch.tanh(z @ W.t() + B)               # [N, M]
        dz = (a * gate) @ U                         # [N, D]  = sum_n sigma_n u_n h_n
        wu = (W * U).sum(-1)                        # [M]     = u_n^T w_n
        trace = ((1.0 - a.square()) * (gate * wu)).sum(-1)  # [N]
        return dz, trace

    def velocity(self, t, z):
        return self.velocity_and_trace(t, z)[0]

    def forward(self, t, states):
        self.nfe += 1
        z, logp = states
        if self.trace_mode == "autograd":
            with torch.set_grad_enabled(True):
                z = z if z.requires_grad else z.requires_grad_(True)
                dz = self.velocity(t, z)
                trace = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
                for i in range(z.shape[1]):
                    trace = trace + torch.autograd.grad(dz[:, i].sum(), z, create_graph=True)[0][:, i]
        else:
            dz, trace = self.velocity_and_trace(t, z)
        return dz, -trace[:, None]                  # Theorem 1: d log p / dt = -tr(df/dz)


class VelocityOnly(nn.Module):
    def __init__(self, dynamics):
        super().__init__()
        self.dynamics = dynamics

    def forward(self, t, z):
        return self.dynamics.velocity(t, z)


class ContinuousNormalizingFlow(nn.Module):
    """Base density at t=0, data/target at t=T.

    * sampling / density matching:  z(0)~p0  --ODE (0 -> T)-->  z(T)
    * likelihood:                   x=z(T)   --ODE (T -> 0)-->  z(0),  log p(x) = log p0(z(0)) - int_0^T tr(df/dz) dt
    """

    def __init__(
        self,
        width=64,
        hidden_dim=32,
        dim=2,
        integration_time=10.0,   # ASSUMPTION: T is not stated in the paper (10 follows the public CNF demo)
        rtol=1e-5,               # paper Sec. 6: tolerance 1e-5 for density estimation
        atol=1e-5,
        learn_base=False,
        base_variance=0.1,       # ASSUMPTION: base variance not stated (public CNF demo uses 0.1 I)
        adjoint=True,
        trace_mode="analytic",
    ):
        super().__init__()
        self.dynamics = CNFDynamics(dim, hidden_dim, width, trace_mode)
        self.integration_time = float(integration_time)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.adjoint = bool(adjoint)
        if learn_base:
            self.base_mean = nn.Parameter(torch.zeros(dim))
            self.base_log_std = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("base_mean", torch.zeros(dim))
            self.register_buffer("base_log_std", torch.ones(dim) * (0.5 * math.log(base_variance)))

    def base_sample(self, n):
        eps = torch.randn(n, self.base_mean.numel(), device=self.base_mean.device)
        z = self.base_mean + torch.exp(self.base_log_std) * eps
        logp = diagonal_gaussian_logprob(z, self.base_mean, self.base_log_std)
        return z, logp

    def base_logprob(self, z):
        return diagonal_gaussian_logprob(z, self.base_mean, self.base_log_std)

    def _solver(self):
        return odeint_adjoint if self.adjoint else odeint

    def density_matching_sample(self, n):
        """Reparameterised samples z(T) and log q(z(T)) for the KL(q||p) loss."""
        z0, logq0 = self.base_sample(n)
        t = torch.tensor([0.0, self.integration_time], device=z0.device, dtype=z0.dtype)
        zt, logqt = self._solver()(
            self.dynamics, (z0, logq0[:, None]), t,
            rtol=self.rtol, atol=self.atol, method="dopri5",
        )
        return zt[-1], logqt[-1, :, 0]

    def log_prob(self, x, t_end=None):
        """log q(z(t_end) = x); t_end defaults to T (the data end of the flow)."""
        t_end = self.integration_time if t_end is None else float(t_end)
        delta1 = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        t = torch.tensor([t_end, 0.0], device=x.device, dtype=x.dtype)
        zt, deltat = self._solver()(
            self.dynamics, (x, delta1), t,
            rtol=self.rtol, atol=self.atol, method="dopri5",
        )
        z0 = zt[-1]
        delta0 = deltat[-1, :, 0]        # = int_0^{t_end} tr(df/dz) dt
        return self.base_logprob(z0) - delta0

    @torch.no_grad()
    def sample_stages(self, n, fractions):
        z0, _ = self.base_sample(n)
        fracs = [float(f) for f in fractions]
        times = torch.tensor([0.0] + [self.integration_time * f for f in fracs], device=z0.device, dtype=z0.dtype)
        zt = odeint(VelocityOnly(self.dynamics), z0, times, rtol=self.rtol, atol=self.atol, method="dopri5")
        return [z.detach() for z in zt[1:]]

    @torch.no_grad()
    def sample(self, n):
        return self.sample_stages(n, [1.0])[-1]

    @torch.no_grad()
    def density_stages(self, grid, fractions, chunk=4096):
        """Exact model density p(z(t)) on `grid` [N,2] at t = frac*T (paper Fig. 5 'Density' row)."""
        out = []
        for f in fractions:
            vals = []
            for i in range(0, grid.shape[0], chunk):
                vals.append(self.log_prob(grid[i : i + chunk], t_end=self.integration_time * float(f)))
            out.append(torch.cat(vals).exp())
        return out
