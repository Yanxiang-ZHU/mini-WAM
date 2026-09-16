import torch

from models.flow_matching import interpolate, flow_matching_loss, euler_sample


def test_interpolate_endpoints():
    x0 = torch.ones(4, 3)
    x_t, v = interpolate(x0, torch.tensor([1.0, 1.0, 1.0, 1.0]))
    # at t=1, x_t should equal x0 (noise term vanishes)
    assert torch.allclose(x_t, x0, atol=1e-4)


def test_target_velocity():
    x0 = torch.ones(4, 3)
    x_t, v = interpolate(x0, torch.zeros(4))
    # at t=0, x_t = eps, v = x0 - eps = x_t_complement... just check v = x0 - eps
    eps = x_t  # at t=0, x_t = eps (since (1-0)*eps + 0*x0)
    assert torch.allclose(v, x0 - eps, atol=1e-6)


def test_loss_zero_for_perfect():
    v = torch.zeros(4, 3)
    assert flow_matching_loss(v, torch.zeros(4, 3)).item() == 0.0


def test_euler_sample_shape():
    def vel_fn(x, t):
        return -x  # pushes toward 0

    out = euler_sample(vel_fn, (2, 5), None, n_steps=4, device="cpu")
    assert out.shape == (2, 5)
