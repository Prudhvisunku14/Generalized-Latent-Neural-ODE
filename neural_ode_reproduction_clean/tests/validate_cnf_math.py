#!/usr/bin/env python3
"""Mathematical sanity checks for the NF/CNF code (run: python tests/validate_cnf_math.py)."""
import sys
from pathlib import Path

import torch
from torchdiffeq import odeint

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "02_cnf"))
sys.path.insert(0, str(ROOT))
from flows import ContinuousNormalizingFlow, PlanarDensityFlow, PlanarMLEFlow, VelocityOnly  # noqa: E402
from common import set_seed  # noqa: E402

set_seed(0)
torch.set_default_dtype(torch.float64)
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")


def grid(lim, n):
    xs = torch.linspace(-lim, lim, n)
    Y, X = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], 1), (2 * lim / (n - 1)) ** 2


# ---- CNF -------------------------------------------------------------------------------------------
cnf = ContinuousNormalizingFlow(width=8, integration_time=2.0, rtol=1e-7, atol=1e-7, adjoint=False)
with torch.no_grad():  # make the (tiny-init) flow non-trivial
    cnf.dynamics.hyper.fc3.weight.mul_(20.0)
    cnf.dynamics.hyper.fc3.bias.mul_(20.0)
z = torch.randn(64, 2) * 0.3
t = torch.tensor(0.7)
dz, tr = cnf.dynamics.velocity_and_trace(t, z)
zz = z.clone().requires_grad_(True)
v = cnf.dynamics.velocity(t, zz)
tr_ag = sum(torch.autograd.grad(v[:, i].sum(), zz, retain_graph=True)[0][:, i] for i in range(2))
check("closed-form trace == autograd trace (eq. 9/10)", torch.allclose(tr, tr_ag, atol=1e-10), f"max err {(tr-tr_ag).abs().max():.2e}")
cnf_ag = ContinuousNormalizingFlow(width=8, integration_time=2.0, rtol=1e-8, atol=1e-8, adjoint=False, trace_mode="autograd")
cnf_ag.load_state_dict(cnf.state_dict())
x = torch.randn(32, 2) * 0.4
check("autograd-trace CNF log_prob == analytic CNF log_prob",
      torch.allclose(cnf.log_prob(x), cnf_ag.log_prob(x), atol=1e-6))

# forward/reverse consistency and sign of d logp/dt
z0, logq0 = cnf.base_sample(256)
zT, logqT = cnf.density_matching_sample(256)
torch.manual_seed(1)
z0, logq0 = cnf.base_sample(128)
t01 = torch.tensor([0.0, 2.0])
zt, lq = odeint(cnf.dynamics, (z0, logq0[:, None]), t01, rtol=1e-9, atol=1e-9, method="dopri5")
lp_rev = cnf.log_prob(zt[-1])
check("log q from forward ODE == log_prob from reverse ODE (data->base)", torch.allclose(lq[-1, :, 0], lp_rev, atol=1e-5),
      f"max err {(lq[-1,:,0]-lp_rev).abs().max():.2e}")
zback = odeint(VelocityOnly(cnf.dynamics), zt[-1],
               torch.tensor([2.0, 0.0]), rtol=1e-9, atol=1e-9)[-1]
check("reverse consistency z_T -> z_0", torch.allclose(zback, z0, atol=1e-5), f"max err {(zback-z0).abs().max():.2e}")
g, dA = grid(4.0, 161)
dens = cnf.log_prob(g).exp()
big, _ = cnf.base_sample(200000)
with torch.no_grad():
    zbig = odeint(VelocityOnly(cnf.dynamics), big, t01, rtol=1e-7, atol=1e-7)[-1]
frac = ((zbig.abs() <= 4.0).all(1)).double().mean()   # mass of the true flow inside the grid window
check("CNF grid mass == Monte-Carlo mass inside window", abs(dens.sum() * dA - frac) < 5e-3,
      f"grid mass={dens.sum()*dA:.4f} MC mass={frac:.4f}")

# gradients through the adjoint are finite
cnf_adj = ContinuousNormalizingFlow(width=8, integration_time=2.0, rtol=1e-6, atol=1e-6, adjoint=True)
loss = -cnf_adj.log_prob(torch.randn(32, 2) * 0.3).mean()
loss.backward()
check("adjoint gradients finite", all(torch.isfinite(p.grad).all() for p in cnf_adj.parameters() if p.grad is not None) and torch.isfinite(loss))
z_, lq_ = cnf_adj.density_matching_sample(32)
check("density-matching sample/logq finite", torch.isfinite(z_).all() and torch.isfinite(lq_).all())

# ---- NF --------------------------------------------------------------------------------------------
nf = PlanarDensityFlow(8)
with torch.no_grad():
    for L in nf.layers:
        L.u.normal_(0, 0.8); L.w.normal_(0, 0.8); L.b.normal_(0, 0.3)
zs, lq = nf.sample_and_logq(500)
check("NF: log_prob(sample) == logq from forward pass", torch.allclose(nf.log_prob(zs), lq, atol=1e-6), f"max err {(nf.log_prob(zs)-lq).abs().max():.2e}")
g, dA = grid(7.0, 281)
m = nf.log_prob(g).exp().sum() * dA
zbig, _ = nf.sample_and_logq(400000)
frac = ((zbig.abs() <= 7.0).all(1)).double().mean()
check("NF grid mass == Monte-Carlo mass inside window", abs(m - frac) < 1.5e-2, f"grid mass={m:.4f} MC mass={frac:.4f}")
mle = PlanarMLEFlow(16)
with torch.no_grad():
    for L in mle.layers:
        L.u.normal_(0, 0.5); L.w.normal_(0, 0.5)
x = torch.randn(200, 2) * 0.5
z = x
for L in mle.layers:
    z, _ = L(z)
back = z
for L in reversed(mle.layers):
    back = L.inverse(back)
check("MLE-NF inverse(forward(x)) == x", torch.allclose(back, x, atol=1e-8), f"max err {(back-x).abs().max():.2e}")
nll = -mle.log_prob(x).mean(); nll.backward()
check("MLE-NF NLL/gradients finite", torch.isfinite(nll) and all(torch.isfinite(p.grad).all() for p in mle.parameters()))
dens = mle.stage_density(g, [1.0])[0]
check("MLE-NF stage_density(100%) is the generative density: total mass ~1", abs(dens.sum() * dA - 1) < 5e-2, f"mass={dens.sum()*dA:.4f}")
print("ALL PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
