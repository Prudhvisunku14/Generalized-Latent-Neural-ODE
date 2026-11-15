import torch
from glo.model import LatentODE
from glo.solver import integrate
from glo.pipeline import infer_test_metrics


def test_ode_solver_exponential():
    class Dynamics(torch.nn.Module):
        def forward(self,t,z): return -z
    t=torch.linspace(0,2,9)
    z=integrate(Dynamics(),torch.tensor([[2.]]),t,method='rk4')
    assert torch.allclose(z[:,0,0],2*torch.exp(-t),atol=3e-4)


def test_backward_through_dynamics_and_encoder():
    torch.manual_seed(0)
    model=LatentODE(latent_dim=4,encoder_hidden=8,dynamics_hidden=10,decoder_hidden=8,
                    cond_dim=4,solver='rk4')
    times=torch.linspace(0,2,12)
    obs_t=times[torch.tensor([0,2,4,6,8])][None,:].repeat(2,1)
    obs_xy=torch.randn(2,5,2)
    out=model(obs_xy,obs_t,times,sample=False)
    assert out['prediction'].shape==(2,12,2)
    out['prediction'].square().mean().backward()
    assert model.dynamics.network[0].weight.grad is not None
    assert model.encoder.cond.weight.grad is not None
    assert model.encoder.posterior.weight.grad is not None


def test_metrics_zero_for_perfect_prediction():
    t=torch.linspace(0,10,20)
    pred=torch.zeros(2,20,2)
    pred[:,:,0]=t
    raw=pred.clone()
    obs_mask=torch.zeros((2,20),dtype=torch.bool)
    obs_mask[:,:8]=True
    prepared={'raw_truth':raw,'mean':torch.zeros(2),'scale':1.,'query_t':t,'obs_mask':obs_mask}
    met=infer_test_metrics(pred,prepared,cutoff=7.)
    assert all(torch.allclose(v,torch.zeros_like(v),atol=1e-6) for v in met.values())


def test_harmonic_oscillator_analytic_dopri5_and_rk4():
    """Rotation dz/dt = A z (exact solution: rotation by omega*t) integrated on [0, 10]."""
    omega = 1.3
    A = torch.tensor([[0., -omega], [omega, 0.]])

    class Rot(torch.nn.Module):
        def forward(self, t, z): return z @ A.T
    t = torch.linspace(0, 10, 200)
    z0 = torch.tensor([[1., 0.], [0., 2.]])
    exact = torch.stack([torch.stack([z0[:, 0] * torch.cos(omega * s) - z0[:, 1] * torch.sin(omega * s),
                                      z0[:, 0] * torch.sin(omega * s) + z0[:, 1] * torch.cos(omega * s)], -1)
                         for s in t])
    for method, kw in (('dopri5', {'rtol': 1e-6, 'atol': 1e-7}), ('rk4', {'step_size': .02})):
        z = integrate(Rot(), z0, t, method=method, **kw)
        assert z.shape == (200, 2, 2)
        assert torch.allclose(z, exact, atol=1e-3), method
    # integration must start at times[0], not at 0, when the grid is shifted
    z = integrate(Rot(), z0, t + 3., method='dopri5', rtol=1e-6, atol=1e-7)
    assert torch.allclose(z, exact, atol=1e-3)


def test_gradients_reach_every_module_through_dopri5():
    torch.manual_seed(1)
    model = LatentODE(latent_dim=8, encoder_hidden=16, dynamics_hidden=16, decoder_hidden=16,
                      dynamics_layers=2, decoder_layers=2, solver='dopri5')
    times = torch.linspace(0, 10, 50)
    obs_t = times[torch.tensor([0, 5, 10, 20, 30])][None].repeat(3, 1)
    out = model(torch.randn(3, 5, 2), obs_t, times, sample=True)
    out['prediction'].square().mean().backward()
    for name, prm in model.named_parameters():
        assert prm.grad is not None and prm.grad.abs().sum() > 0, name


def test_depth_one_matches_legacy_layout():
    m = LatentODE(latent_dim=4, encoder_hidden=25, dynamics_hidden=20, decoder_hidden=20)
    keys = set(m.state_dict())
    assert {'dynamics.network.0.weight', 'dynamics.network.2.weight', 'decoder.0.weight',
            'decoder.2.weight'} <= keys
    assert not any(k.startswith('dynamics.network.4') or k.startswith('decoder.4') for k in keys)
