"""Flow matching utilities (linear path, conditional OT-style).

The model learns the velocity field of a linear interpolation between Gaussian
noise and the data:

    x_t   = (1 - t) * eps + t * x0,     eps ~ N(0, I)
    v*    = x0 - eps
    loss  = E || v_theta(x_t, t, cond) - v* ||^2

At inference, integrate the learned velocity with a few Euler steps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_noise_like(x: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(x)


def interpolate(x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x_t, target_velocity) for a batch of clean targets x0 and times t."""
    eps = torch.randn_like(x0)
    t = t.view(-1, *([1] * (x0.ndim - 1)))
    x_t = (1 - t) * eps + t * x0
    v = x0 - eps
    return x_t, v


def flow_matching_loss(velocity: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(velocity, target_velocity)


@torch.no_grad()
def euler_sample(velocity_fn, shape: tuple, cond_fn, n_steps: int = 4,
                 device: str = "cpu", x0: torch.Tensor | None = None) -> torch.Tensor:
    """Integrate the learned velocity field from noise with Euler steps.

    ``velocity_fn(x, t) -> velocity`` is a callable returning the predicted
    velocity for the current point x and scalar time t (batched).  ``cond_fn``
    is ignored here but kept for API symmetry; conditioning must be baked into
    ``velocity_fn`` by the caller via a closure.
    """
    x = x0 if x0 is not None else torch.randn(shape, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((shape[0],), i / n_steps, device=device)
        v = velocity_fn(x, t)
        x = x + v * dt
    return x
