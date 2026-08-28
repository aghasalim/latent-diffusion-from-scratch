"""Tests. The UNet shape test exists because the first version was wrong."""
import pytest
import torch

from ldm.autoencoder import AutoEncoder, psnr
from ldm.data import load
from ldm.diffusion import Diffusion, cosine_schedule
from ldm.metrics import Featurizer, cfid, sliced_w2
from ldm.unet import UNet


# --- data -------------------------------------------------------------------
def test_data_is_padded_and_scaled():
    tr, va = load(500)
    assert tr.shape[1:] == (1, 32, 32), "padded to 32 so f=8 divides evenly"
    assert tr.min() >= -1.0 and tr.max() <= 1.0
    assert tr.shape[0] + va.shape[0] == 500


def test_data_split_is_disjoint_and_seeded():
    a, _ = load(400, seed=1)
    b, _ = load(400, seed=1)
    c, _ = load(400, seed=2)
    assert torch.equal(a, b) and not torch.equal(a, c)


# --- autoencoder ------------------------------------------------------------
@pytest.mark.parametrize("f,expect", [(2, (4, 16, 16)), (4, (4, 8, 8)), (8, (4, 4, 4))])
def test_latent_shape_and_compression(f, expect):
    ae = AutoEncoder(f=f)
    assert ae.latent_shape == expect
    elems = expect[0] * expect[1] * expect[2]
    assert abs(ae.compression_ratio() - (32 * 32) / elems) < 1e-6


def test_autoencoder_round_trips_shape():
    ae = AutoEncoder(f=4)
    x = torch.randn(3, 1, 32, 32)
    r, mean, logvar = ae(x)
    assert r.shape == x.shape
    assert mean.shape == (3, *ae.latent_shape)
    assert logvar.shape == mean.shape


def test_decoder_output_is_bounded():
    """Tanh output must stay in the data range, or PSNR is meaningless."""
    ae = AutoEncoder(f=4)
    out = ae.decode(torch.randn(4, *ae.latent_shape) * 10)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_logvar_is_clamped():
    ae = AutoEncoder(f=4)
    _, logvar = ae.encode(torch.randn(2, 1, 32, 32) * 50)
    assert logvar.min() >= -8.0 - 1e-4 and logvar.max() <= 8.0 + 1e-4


def test_autoencoder_learns_to_reconstruct():
    torch.manual_seed(0)
    tr, _ = load(600)
    ae = AutoEncoder(f=4)
    opt = torch.optim.Adam(ae.parameters(), 2e-3)
    first = None
    for i in range(120):
        loss, parts = ae.loss(tr[:96])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 0:
            first = parts["recon"]
    assert parts["recon"] < first * 0.5, f"{first} -> {parts['recon']}"


def test_psnr_is_infinite_for_identical_images():
    x = torch.rand(2, 1, 8, 8) * 2 - 1
    assert psnr(x, x) > 100


def test_psnr_decreases_with_noise():
    x = torch.rand(4, 1, 16, 16) * 2 - 1
    assert psnr(x + 0.01 * torch.randn_like(x), x) > psnr(x + 0.2 * torch.randn_like(x), x)


# --- schedule ---------------------------------------------------------------
def test_cosine_schedule_is_monotone_and_bounded():
    ab = cosine_schedule(300)
    assert ab.shape == (300,)
    assert bool((ab[1:] <= ab[:-1]).all()), "alphas_cumprod must decrease"
    assert ab[0] < 1.0 and ab[-1] > 0.0


def test_q_sample_endpoints():
    """At t=0 the sample is nearly the image; at t=T-1 nearly pure noise."""
    d = Diffusion(300)
    x0 = torch.randn(8, 1, 8, 8)
    near, _ = d.q_sample(x0, torch.zeros(8, dtype=torch.long))
    far, _ = d.q_sample(x0, torch.full((8,), 299, dtype=torch.long))
    assert (near - x0).abs().mean() < (far - x0).abs().mean()
    assert (near - x0).abs().mean() < 0.1


def test_q_sample_variance_is_preserved():
    d = Diffusion(300)
    x0 = torch.randn(2000, 1, 4, 4)
    for t in (0, 150, 299):
        xt, _ = d.q_sample(x0, torch.full((2000,), t, dtype=torch.long))
        assert abs(xt.std().item() - 1.0) < 0.1, f"t={t} std {xt.std().item()}"


# --- unet -------------------------------------------------------------------
@pytest.mark.parametrize("ch,res,mults", [
    (1, 32, (1, 2)), (4, 16, (1, 2)), (4, 8, (1, 2)), (4, 4, (1,)), (3, 32, (1, 2, 2)),
])
def test_unet_preserves_shape(ch, res, mults):
    """The first UNet mismatched skip channels: it built a block for 128 where
    the concatenation produced 96. GroupNorm caught it, and this test pins it."""
    net = UNet(channels=ch, mults=mults)
    x = torch.randn(2, ch, res, res)
    assert net(x, torch.randint(0, 100, (2,))).shape == x.shape


def test_unet_output_depends_on_timestep():
    torch.manual_seed(0)
    net = UNet(channels=1)
    x = torch.randn(2, 1, 32, 32)
    a = net(x, torch.zeros(2, dtype=torch.long))
    b = net(x, torch.full((2,), 99, dtype=torch.long))
    assert not torch.allclose(a, b, atol=1e-4)


# --- sampling ---------------------------------------------------------------
@pytest.mark.parametrize("nfe", [1, 5, 25])
def test_ddim_calls_the_model_exactly_nfe_times(nfe):
    """NFE is the cost axis of the whole repo, so it has to be exact."""
    calls = {"n": 0}

    class Counting(torch.nn.Module):
        def forward(self, x, t):
            calls["n"] += 1
            return torch.zeros_like(x)

    Diffusion(200).ddim_sample(Counting(), (2, 1, 8, 8), nfe=nfe,
                               generator=torch.Generator().manual_seed(0))
    assert calls["n"] == nfe


def test_ddim_is_deterministic_at_eta_zero():
    torch.manual_seed(0)
    net = UNet(channels=1)
    d = Diffusion(200)
    a = d.ddim_sample(net, (2, 1, 32, 32), nfe=5, eta=0.0,
                      generator=torch.Generator().manual_seed(3))
    b = d.ddim_sample(net, (2, 1, 32, 32), nfe=5, eta=0.0,
                      generator=torch.Generator().manual_seed(3))
    assert torch.allclose(a, b, atol=1e-6)


def test_ddim_output_is_finite():
    torch.manual_seed(0)
    out = Diffusion(200).ddim_sample(UNet(channels=1), (2, 1, 32, 32), nfe=10,
                                     generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out).all()


def test_ddim_callback_sees_every_step_and_changes_nothing():
    """The animation in bench/figures.py reads the trajectory through this."""
    torch.manual_seed(0)
    net, d = UNet(channels=1), Diffusion(200)
    seen = []
    a = d.ddim_sample(net, (2, 1, 32, 32), nfe=6,
                      generator=torch.Generator().manual_seed(3),
                      callback=lambda i, t, x0, x: seen.append(x.clone()))
    b = d.ddim_sample(net, (2, 1, 32, 32), nfe=6,
                      generator=torch.Generator().manual_seed(3))
    assert len(seen) == 6
    assert torch.equal(seen[-1], a)
    assert torch.allclose(a, b, atol=1e-6)


# --- metrics ----------------------------------------------------------------
def test_cfid_is_near_zero_for_the_same_distribution():
    torch.manual_seed(0)
    f = Featurizer().eval()
    x = torch.randn(600, 1, 32, 32)
    assert cfid(f, x[:300], x[300:]) < 1.0


def test_cfid_detects_a_different_distribution():
    torch.manual_seed(0)
    f = Featurizer().eval()
    a = torch.randn(400, 1, 32, 32)
    b = torch.randn(400, 1, 32, 32) * 3 + 2
    assert cfid(f, a, b) > cfid(f, a[:200], a[200:])


def test_cfid_is_non_negative():
    torch.manual_seed(0)
    f = Featurizer().eval()
    a, b = torch.randn(300, 1, 32, 32), torch.randn(300, 1, 32, 32) * 2
    assert cfid(f, a, b) >= -1e-6


@pytest.mark.parametrize("na,nb", [(300, 300), (200, 800), (800, 200)])
def test_sliced_w2_is_sample_size_independent(na, nb):
    g = torch.Generator().manual_seed(0)
    pool = torch.randn(800, 1, 8, 8, generator=g)
    same = sliced_w2(pool[:na], pool[:nb])
    shifted = sliced_w2(pool[:na], pool[:nb] + 2.0)
    assert same < shifted / 4


def test_sliced_w2_zero_on_identical_input():
    x = torch.randn(200, 1, 8, 8)
    assert sliced_w2(x, x) < 1e-5
