"""LayoutCross on a synthetic oracle: a smooth angular flux psi[mu, gamma, s, E]
whose fibers are returned exactly, so every error is the cross's own."""

import pytest
import torch

from ttnte.cross import CrossConfig, LayoutCross

torch.set_default_dtype(torch.float64)

NM, NG, G = 4, 8, 24
# Two patches: a 3 x 4 grid and a 6 x 3 grid of dofs
SHAPES = [[3, 4], [6, 3]]
OFFSETS = [0, 12]
NS = 30


def _psi():
    mu = torch.linspace(0.1, 1.0, NM)
    ga = torch.linspace(0.0, 6.0, NG)
    x = torch.linspace(0.0, 1.0, NS)
    sig = 1.0 + 3.0 * torch.linspace(0.0, 1.0, G)
    path = mu[:, None] * (1.5 + torch.cos(ga))[None, :]
    return torch.exp(-sig * x[:, None] / path[:, :, None, None])  # (NM, NG, NS, G)


PSI = _psi()
W = torch.outer(torch.linspace(0.5, 1.5, NM), torch.full((NG,), 1.0 / NG)).flatten()
Y = torch.stack(
    [torch.ones(NM * NG), torch.linspace(-1, 1, NM).repeat_interleave(NG)], 1
)
REF = PSI.reshape(NM * NG, NS, G).permute(0, 2, 1)  # (Nd, G, NS)


def _sweep(blocks):
    out = [PSI[d // NG, d % NG, :, g] for dirs, gs in blocks for d in dirs for g in gs]
    return torch.stack(out)


def _cross(layout, tol=1e-6, **kw):
    mass = kw.pop("mass_bands", [])
    cfg = CrossConfig(layout=layout, tol=tol, **kw)
    return LayoutCross(
        2,
        NM,
        NG,
        G,
        NS,
        W,
        Y,
        [0, NG - 1, NM * NG - 1, (NM - 1) * NG],
        _sweep,
        cfg,
        OFFSETS,
        SHAPES,
        mass,
    )


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


LAYOUTS = [
    "polar|azimuthal|space|energy",
    "polar,azimuthal,space|energy",
    "polar,azimuthal|space|energy",
    "energy|space|azimuthal|polar",
    "polar|azimuthal|space,energy",
    "azimuthal,polar|energy|space",
]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_layouts_reproduce_flux_and_moments(layout):
    c = _cross(layout)
    c.run()
    assert c.converged
    full = c.evaluate(list(range(NM * NG)), list(range(G)), list(range(NS)))
    assert _rel(full, REF) < 20 * 1e-6
    mref = torch.einsum("dm,d,dgs->mgs", Y, W, REF)
    assert _rel(c.moments(), mref) < 20 * 1e-6
    assert c.num_fibers <= NM * NG * G


def test_split_layout_samples_fewer_fibers_than_all():
    c = _cross("polar|azimuthal|space|energy", tol=1e-4)
    c.run()
    assert c.num_fibers < 0.5 * NM * NG * G


def test_tight_tolerance_is_exact():
    c = _cross("polar|azimuthal|space|energy", tol=1e-13)
    c.run()
    full = c.evaluate(list(range(NM * NG)), list(range(G)), list(range(NS)))
    assert _rel(full, REF) < 1e-10


def test_evaluate_subsets_and_global_cores():
    c = _cross("polar|azimuthal|space|energy")
    c.run()
    dirs, groups, space = [3, 17, 30], [0, 5, 23], [1, 12, 29]
    sub = c.evaluate(dirs, groups, space)
    full = c.evaluate(list(range(NM * NG)), list(range(G)), list(range(NS)))
    assert torch.allclose(sub, full[dirs][:, groups][:, :, space])
    cores = c.cores
    assert [k.shape[1] for k in cores] == [NM, NG, NS, G]
    assert cores[0].shape[0] == 1 and cores[-1].shape[2] == 1


def _dense(cores):
    t = cores[0]
    for k in cores[1:]:
        t = torch.tensordot(t, k, ([t.dim() - 1], [0]))
    return t.squeeze(0).squeeze(-1)


@pytest.mark.parametrize(
    "layout, order",
    [
        ("polar|azimuthal|x|y|energy", "mgxye"),
        ("polar|y|x|azimuthal|energy", "myxge"),
        ("energy|x,y|polar,azimuthal", "exymg"),
        ("polar,azimuthal|x,y,energy", "mgxye"),
    ],
)
def test_patch_tts_in_user_layout(layout, order):
    c = _cross(layout, tol=1e-7)
    c.run()
    for p, (shape, off) in enumerate(zip(SHAPES, OFFSETS)):
        n = shape[0] * shape[1]
        ref = PSI[:, :, off : off + n].reshape(NM, NG, shape[0], shape[1], G)
        ref = ref.permute(["mgxye".index(a) for a in order])
        cores = c.patch_tt(p)
        dense = _dense(cores).reshape(ref.shape)
        assert _rel(dense, ref) < 50 * 1e-7


def test_weighting_options_agree():
    # A tridiagonal SPD mass matrix per patch, and lethargy widths
    bands = []
    for shape in SHAPES:
        n = shape[0] * shape[1]
        band = torch.zeros(n, 2)
        band[:, 0] = 4.0
        band[:-1, 1] = 1.0
        bands.append(band)
    du = torch.linspace(0.5, 2.0, G).tolist()
    plain = _cross("polar|azimuthal|space|energy", weighted=False)
    plain.run()
    weighted = _cross(
        "polar|azimuthal|space|energy", lethargy_widths=du, mass_bands=bands
    )
    weighted.run()
    for c in (plain, weighted):
        full = c.evaluate(list(range(NM * NG)), list(range(G)), list(range(NS)))
        assert _rel(full, REF) < 20 * 1e-6


def test_deterministic():
    a = _cross("polar|azimuthal|space|energy")
    b = _cross("polar|azimuthal|space|energy")
    a.run()
    b.run()
    assert a.ranks == b.ranks and a.num_fibers == b.num_fibers
    assert torch.equal(a.moments(), b.moments())


def test_honest_error_report():
    """A layout that converges slowly must not claim more than it achieves."""
    c = _cross("space|energy|azimuthal|polar", tol=1e-8, max_sweeps=1)
    c.run()
    full = c.evaluate(list(range(NM * NG)), list(range(G)), list(range(NS)))
    if c.converged:
        assert _rel(full, REF) < 100 * 1e-8


def test_invalid_configurations():
    with pytest.raises(RuntimeError):
        _cross("polar|space|energy")  # no azimuthal in 2-D
    with pytest.raises(RuntimeError):
        _cross("polar|azimuthal|space|energy", tol=0.0)
