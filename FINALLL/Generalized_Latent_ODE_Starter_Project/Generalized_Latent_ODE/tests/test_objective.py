"""Validation-ELBO implementation and sign convention (used for checkpoint selection)."""
import math
import torch
from glo.pipeline import batch_objective
from glo.model import kl_standard_normal


def _prepared(B=3, T=12, n_obs=5, scale=.9):
    g = torch.Generator().manual_seed(0)
    idx = torch.sort(torch.randperm(T, generator=g)[:n_obs]).values[None].repeat(B, 1)
    return {'obs_indices': idx, 'obs_xy': torch.randn(B, n_obs, 2, generator=g),
            'truth': torch.randn(B, T, 2, generator=g), 'scale': scale}


def test_objective_is_negative_elbo_by_hand():
    torch.manual_seed(1)
    P = _prepared(); B, T = 3, 12
    out = {'prediction': torch.randn(B, T, 2), 'mu': torch.randn(B, 4), 'logvar': torch.randn(B, 4) * .3}
    loss, parts = batch_objective(out, P, obs_sigma=.05, kl_weight=1.)
    sigma = .05 / P['scale']
    loglik = torch.distributions.Normal(out['prediction'][torch.arange(B)[:, None], P['obs_indices']], sigma)\
        .log_prob(P['obs_xy']).sum(dim=(1, 2))                       # log p(x | z) per trajectory
    q = torch.distributions.Normal(out['mu'], (.5 * out['logvar']).exp())
    p = torch.distributions.Normal(torch.zeros(4), torch.ones(4))
    kl = torch.distributions.kl_divergence(q, p).sum(-1)              # KL(q || N(0, I)) per trajectory
    elbo = (loglik - kl).mean()
    assert torch.allclose(loss, -elbo, rtol=1e-5), (float(loss), float(-elbo))
    assert math.isclose(parts['nll'], float(-loglik.mean()), rel_tol=1e-5)
    assert math.isclose(parts['kl'], float(kl.mean()), rel_tol=1e-5)
    assert float(kl_standard_normal(out['mu'], out['logvar'])) >= 0


def test_lower_objective_means_better_fit():
    """Checkpoint selection keeps the MINIMUM: a better fit must give a lower value, and it may be
    negative because the Gaussian density with sigma < 1/sqrt(2*pi*e) exceeds 1 near the mean."""
    P = _prepared(); B, T = 3, 12
    perfect = torch.zeros(B, T, 2)
    perfect[torch.arange(B)[:, None], P['obs_indices']] = P['obs_xy']
    zero_kl = {'mu': torch.zeros(B, 4), 'logvar': torch.zeros(B, 4)}
    good, _ = batch_objective({'prediction': perfect, **zero_kl}, P)
    bad, _ = batch_objective({'prediction': perfect + .3, **zero_kl}, P)
    assert good < bad
    n_scalar = P['obs_xy'][0].numel()
    sigma = .05 / P['scale']
    assert math.isclose(float(good), n_scalar * (math.log(sigma) + .5 * math.log(2 * math.pi)), rel_tol=1e-5)
    assert float(good) < 0
