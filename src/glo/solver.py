"""torchdiffeq interface, plus clearly labelled offline RK4 test fallback.

The fixed-step fallback differentiates directly through its RK4 operations. It is
not the torchdiffeq adaptive solver or its adjoint method, and should not be
used to make reproducibility claims about the original paper.
"""
import math
import torch
try:
    from torchdiffeq import odeint, odeint_adjoint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


def _offline_rk4(func, y0, times, max_step=.18):
    if times.ndim != 1 or len(times) < 2 or not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError('times must be a strictly increasing 1-D tensor')
    state = y0
    states = [state]
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        steps = max(1, math.ceil(float(dt.detach().cpu()) / max_step))
        h = dt / steps
        tau = times[i - 1]
        for _ in range(steps):
            k1 = func(tau, state)
            k2 = func(tau + h / 2, state + h * k1 / 2)
            k3 = func(tau + h / 2, state + h * k2 / 2)
            k4 = func(tau + h, state + h * k3)
            state = state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            tau = tau + h
        states.append(state)
    return torch.stack(states, 0)


def integrate(func, y0, times, method='dopri5', adjoint=False, rtol=1e-4, atol=1e-5,
              step_size=None):
    """Integrate dz/dt = func(t, z) from times[0]; returns [len(times), *y0.shape].

    `step_size` only applies to fixed-step methods (rk4/euler/midpoint); it caps
    the internal step so sparse query grids are still integrated accurately.
    """
    if HAS_TORCHDIFFEQ:
        solver = odeint_adjoint if adjoint else odeint
        options = None
        if method in ('rk4', 'euler', 'midpoint', 'explicit_adams') and step_size:
            options = {'step_size': float(step_size)}
        return solver(func, y0, times, method=method, rtol=rtol, atol=atol, options=options)
    if adjoint:
        raise RuntimeError('Adjoint integration requires installed torchdiffeq')
    return _offline_rk4(func, y0, times, max_step=step_size or .18)


def solver_status():
    return ('torchdiffeq (dopri5/adjoint available)' if HAS_TORCHDIFFEQ else
            'OFFLINE DIFFERENTIABLE RK4 FALLBACK (torchdiffeq unavailable)')
