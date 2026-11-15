"""M3 (dynamics-conditioned Latent ODE): dz/dt = f(z, c) with c constant along the trajectory."""
import inspect
import torch
from glo.model import LatentODE
from glo.solver import integrate

CFG = dict(latent_dim=8, encoder_hidden=16, dynamics_hidden=16, decoder_hidden=16,
           dynamics_layers=2, decoder_layers=2, cond_dim=4, solver='rk4', step_size=0.1)


def _inputs(B=3):
    times = torch.linspace(0, 10, 40)
    obs_t = times[torch.tensor([0, 3, 7, 12, 20, 27])][None].repeat(B, 1)
    return torch.randn(B, 6, 2), obs_t, times


def test_code_is_constant_through_integration_and_bounded():
    torch.manual_seed(0)
    m = LatentODE(**CFG); obs, ot, t = _inputs()
    mu, lv, c = m.encoder(obs, ot, t[-1])
    assert c.shape == (3, 4) and torch.all(c.abs() <= 1)
    traj = integrate(m.dynamics, torch.cat([mu, c], -1), t, method='rk4', step_size=0.1)
    assert torch.allclose(traj[..., 8:], c.expand_as(traj[..., 8:]))        # dc/dt = 0 exactly


def test_code_changes_the_dynamics_and_gets_gradients():
    torch.manual_seed(0)
    m = LatentODE(**CFG); obs, ot, t = _inputs()
    out = m(obs, ot, t, sample=False)
    z = out['mu']
    a = m.dynamics(0., torch.cat([z, torch.full((3, 4), -.5)], -1))[..., :8]
    b = m.dynamics(0., torch.cat([z, torch.full((3, 4), .5)], -1))[..., :8]
    assert not torch.allclose(a, b)                                           # f depends on c
    out['prediction'].square().mean().backward()
    assert m.encoder.cond.weight.grad.abs().sum() > 0                         # c is trained through the ODE


def test_no_label_or_future_inputs():
    """The model's forward pass only receives observed coordinates/times and the query grid."""
    params = list(inspect.signature(LatentODE.forward).parameters)
    assert params == ['self', 'obs_xy', 'obs_t', 'query_t', 'sample']
    assert LatentODE(**CFG).encoder.gru.input_size == 4                       # (x, y, t/T, dt/T) only
