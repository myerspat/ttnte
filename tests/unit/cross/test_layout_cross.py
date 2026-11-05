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
    widths = kw.pop("lethargy_widths", [])
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
        widths,
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
    "space,azimuthal,polar|energy",  # angles not leading: per-moment path
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
    true = _rel(full, REF)
    # Converged or not, the reported global error does not understate it
    assert c.global_error >= true / 5
    if c.converged:
        assert true < 100 * 1e-8


def test_invalid_configurations():
    with pytest.raises(RuntimeError):
        _cross("polar|space|energy")  # no azimuthal in 2-D
    with pytest.raises(RuntimeError):
        _cross("polar|azimuthal|space|energy", tol=0.0)


def test_invalid_problem_geometry():
    """The fiber layout is checked before any raw-pointer weighting: offsets
    and shapes must tile the fiber, mass bands match their patch, and initial
    directions be in range."""
    cfg = CrossConfig(layout="polar|azimuthal|space|energy", tol=1e-4)
    dirs = [0, NG - 1, NM * NG - 1, (NM - 1) * NG]

    def make(offsets=OFFSETS, shapes=SHAPES, mass=(), init=dirs):
        return LayoutCross(
            2, NM, NG, G, NS, W, Y, init, _sweep, cfg, offsets, shapes, list(mass)
        )

    with pytest.raises(RuntimeError, match="disagree"):
        make(offsets=[0, 10])
    with pytest.raises(RuntimeError, match="start at 0"):
        make(offsets=[1, 12])
    with pytest.raises(RuntimeError, match="mass band"):
        make(mass=[torch.ones(12, 2), torch.ones(17, 2)])
    with pytest.raises(RuntimeError, match="Initial direction"):
        make(init=[NM * NG])
    with pytest.raises(RuntimeError, match="at least one direction"):
        make(init=[])


def test_selection_options():
    """select_cond and select_near are validated and reach the selection: a
    stricter conditioning target, or ignoring the cost of new sweeps
    (select_near = 1), takes more fibers."""
    cfg = CrossConfig(select_cond=0.05, select_near=1.0)
    assert cfg.select_cond == 0.05 and cfg.select_near == 1.0
    lay = "polar|azimuthal|space|energy"
    runs = {}
    for name, kw in (
        ("default", {}),
        ("strict", {"select_cond": 0.6}),
        ("greedy", {"select_near": 1.0}),
    ):
        runs[name] = _cross(lay, tol=1e-4, **kw)
        runs[name].run()
        assert runs[name].converged
    assert runs["strict"].num_fibers > runs["default"].num_fibers
    assert runs["greedy"].num_fibers > runs["default"].num_fibers
    for kw in (
        {"select_cond": 0.0},
        {"select_cond": 1.0},
        {"select_near": 0.0},
        {"select_near": 1.5},
    ):
        with pytest.raises(RuntimeError):
            _cross("polar|azimuthal|space|energy", **kw)


@pytest.mark.parametrize(
    "layout", ["polar,azimuthal,space|energy", "polar|azimuthal|space|energy"]
)
def test_phase_timings_are_disjoint(layout):
    """Every phase is timed once: sweeps inside superblocks, the initial
    sketch and the hold-out check are not counted again, so the phases sum
    to at most the wall time of the run."""
    c = _cross(layout, tol=1e-4)
    c.run()
    t = c.timings
    for key in (
        "sweep",
        "weight_cpu",
        "gather",
        "project",
        "svd",
        "select",
        "init",
        "check",
    ):
        assert key in t and t[key] >= 0.0
    assert "superblock" not in t
    # weight_cpu is CPU time inside the sweeps, not a phase of its own
    phases = {k: v for k, v in t.items() if k != "weight_cpu"}
    assert sum(phases.values()) <= c.seconds * 1.001 + 1e-4


@pytest.mark.parametrize(
    "layout", ["polar,azimuthal,space|energy", "polar|azimuthal|space|energy"]
)
def test_stall_exit(layout):
    """A rank-capped cross cannot reach its tolerance: with stall_passes it
    stops once two passes leave the ranks unchanged and the hold-out error
    falls by less than 10%, instead of using every pass."""
    full = _cross(layout, tol=1e-8, max_rank=2, stall_passes=0)
    full.run()
    assert not full.converged and not full.stalled and full.sweeps_done == 8
    early = _cross(layout, tol=1e-8, max_rank=2)
    early.run()
    assert not early.converged and early.stalled and early.sweeps_done < 8
    # The stall rule runs on the error accepted on (global by default)
    errs = [h["global_error"] for h in early.history if h["phase"] == "backward"]
    assert errs[-1] > 0.9 * errs[-2] and errs[-2] > 0.9 * errs[-3]
    # A converging cross is unaffected
    c = _cross(layout)
    c.run()
    assert c.converged and not c.stalled
    with pytest.raises(RuntimeError):
        _cross(layout, stall_passes=-1)
    with pytest.raises(RuntimeError):
        _cross(layout, stall_improvement=1.0)


def test_norm_diagnostics():
    """Backward records carry the RMS norms of the hold-out pool and of the swept fibers
    (unweighted here, so they lie within the fiber norms)."""
    c = _cross("polar,azimuthal,space|energy", tol=1e-4, weighted=False)
    c.run()
    norms = REF.norm(dim=2)
    lo, hi = norms.min().item(), norms.max().item()
    for h in c.history:
        if h["phase"] == "forward":
            assert h["holdout_rms"] != h["holdout_rms"]  # NaN
            continue
        assert lo <= h["holdout_rms"] <= hi and lo <= h["swept_rms"] <= hi


@pytest.mark.parametrize(
    "layout", ["space|energy|azimuthal|polar", "energy|space|azimuthal|polar"]
)
def test_global_error_estimate(layout):
    """The recorded global estimate follows the true global relative error of the TT
    after every pass (within a factor of 5 here), and global_holdout accepts on it."""
    for passes in range(1, 5):
        c = _cross(layout, tol=1e-8, weighted=False, max_sweeps=passes, stall_passes=0)
        c.run()
        full = c.evaluate(list(range(NM * NG)), list(range(G)), list(range(NS)))
        est = [h for h in c.history if h["phase"] == "backward"][-1]["global_error"]
        assert est == c.global_error
        true = _rel(full, REF)
        assert true / 5 <= est <= 5 * true
    g = _cross(layout, tol=1e-6, global_holdout=True)
    g.run()
    assert g.converged and g.holdout_error <= 10 * 1e-6


def test_two_stage_check():
    """A pool within the acceptance accepts without the global estimate (NaN); a failing
    pool is judged on the global estimate, and holdout_error is the error each pass was
    accepted or rejected on."""
    c = _cross("polar,azimuthal,space|energy", tol=1e-4)
    c.run()
    back = [h for h in c.history if h["phase"] == "backward"]
    assert c.converged
    for h in back:
        pool_ok = h["holdout_error"] <= 10 * 1e-4
        assert (h["global_error"] != h["global_error"]) == pool_ok  # NaN iff pool ok
    last = back[-1]
    accepted = (
        last["holdout_error"]
        if last["global_error"] != last["global_error"]
        else last["global_error"]
    )
    assert c.holdout_error == accepted
    # Failing pools: the run's error is the global estimate
    f = _cross("space|energy|azimuthal|polar", tol=1e-8, max_sweeps=2, stall_passes=0)
    f.run()
    last = [h for h in f.history if h["phase"] == "backward"][-1]
    assert last["holdout_error"] > 10 * 1e-8
    assert f.holdout_error == last["global_error"] == f.global_error
    # Without global_holdout only the pool decides, and no estimate is formed
    p = _cross(
        "space|energy|azimuthal|polar",
        tol=1e-8,
        max_sweeps=2,
        stall_passes=0,
        global_holdout=False,
    )
    p.run()
    last = [h for h in p.history if h["phase"] == "backward"][-1]
    assert p.holdout_error == last["holdout_error"]
    assert last["global_error"] != last["global_error"]


def test_sweep_calls_recorded():
    """Every sweep request is recorded with its phase, and the requests add up to the
    fibers swept."""
    c = _cross("polar|azimuthal|space|energy", tol=1e-4)
    c.run()
    calls = c.sweep_calls
    assert {s["phase"] for s in calls} <= {"init", "superblock", "check"}
    assert sum(s["fibers"] for s in calls) == c.num_fibers
    assert all(
        s["blocks"] >= 1 and s["directions"] >= 1 and s["seconds"] >= 0 for s in calls
    )
    assert abs(sum(s["seconds"] for s in calls) - c.timings["sweep"]) < 1e-6


@pytest.mark.parametrize("layout", LAYOUTS)
def test_evaluate_pairs_matches_evaluate(layout):
    """The pair evaluator used by the hold-out check returns the same fibers as
    evaluate() at arbitrary (direction, group) pairs."""
    c = _cross(layout, tol=1e-4)
    c.run()
    pairs = [3, 17, 140, 141, 500, NM * NG * G - 1, 0, 30 * G + 5]
    got = c.evaluate_pairs(pairs)
    want = torch.stack(
        [c.evaluate([p // G], [p % G], list(range(NS)))[0, 0] for p in pairs]
    )
    assert torch.allclose(got, want, rtol=1e-11, atol=1e-13 * want.abs().max().item())


@pytest.mark.parametrize("layout", LAYOUTS[:3])
def test_qr_svd_path_agrees(layout):
    """gram_svd=False (the QR-based truncated SVD) reaches the same accuracy."""
    for gram in (True, False):
        c = _cross(layout, tol=1e-6, gram_svd=gram)
        c.run()
        full = c.evaluate(list(range(NM * NG)), list(range(G)), list(range(NS)))
        assert c.converged and _rel(full, REF) < 1e-4


def test_one_dimensional():
    """1-D: one angle mode (polar), so the azimuthal mode has size 1."""
    nm = 8
    mu = torch.linspace(-0.95, 0.95, nm)
    x = torch.linspace(0.0, 1.0, NS)
    sig = 1.0 + 3.0 * torch.linspace(0.0, 1.0, G)
    psi = torch.exp(-sig * x[:, None] / (1.2 + mu[:, None, None]))  # (nm, NS, G)

    def sweep(blocks):
        return torch.stack(
            [psi[d, :, g] for dirs, gs in blocks for d in dirs for g in gs]
        )

    w = torch.full((nm,), 2.0 / nm)
    y = torch.stack([torch.ones(nm), mu], 1)
    cfg = CrossConfig(layout="polar|space|energy", tol=1e-6)
    c = LayoutCross(1, nm, 1, G, NS, w, y, [0, nm - 1], sweep, cfg, OFFSETS, SHAPES, [])
    c.run()
    full = c.evaluate(list(range(nm)), list(range(G)), list(range(NS)))
    assert c.converged and _rel(full, psi.permute(0, 2, 1)) < 1e-4


def test_banded_mass_weighting():
    """A patch above the dense-factor limit (3000 dofs) is weighted and
    unweighted through its banded Cholesky factor: the weighted cross still
    reproduces the flux, as the unweighted one does."""
    shape = [60, 60]
    ns = shape[0] * shape[1]
    mu = torch.linspace(0.1, 1.0, NM)
    ga = torch.linspace(0.0, 6.0, NG)
    x = torch.linspace(0.0, 1.0, ns)
    sig = 1.0 + 3.0 * torch.linspace(0.0, 1.0, G)
    path = mu[:, None] * (1.5 + torch.cos(ga))[None, :]
    psi = torch.exp(-sig * x[:, None] / path[:, :, None, None])  # (NM, NG, ns, G)
    ref = psi.reshape(NM * NG, ns, G).permute(0, 2, 1)

    def sweep(blocks):
        return torch.stack(
            [
                psi[d // NG, d % NG, :, g]
                for dirs, gs in blocks
                for d in dirs
                for g in gs
            ]
        )

    band = torch.zeros(ns, 3)
    band[:, 0] = 6.0
    band[:-1, 1] = 1.0
    band[:-2, 2] = 0.5
    dirs = [0, NG - 1, NM * NG - 1, (NM - 1) * NG]
    for mass in ([], [band]):
        cfg = CrossConfig(
            layout="polar|azimuthal|space|energy", tol=1e-6, weighted=bool(mass)
        )
        c = LayoutCross(2, NM, NG, G, ns, W, Y, dirs, sweep, cfg, [0], [shape], mass)
        c.run()
        full = c.evaluate(list(range(NM * NG)), list(range(G)), list(range(ns)))
        assert c.converged and _rel(full, ref) < 1e-4


def test_lethargy_widths_validated():
    """Lethargy widths belong to the problem: one per group, all positive."""
    with pytest.raises(RuntimeError, match="one entry per group"):
        _cross("polar|azimuthal|space|energy", lethargy_widths=[1.0] * (G - 1))
    with pytest.raises(RuntimeError, match="positive"):
        _cross("polar|azimuthal|space|energy", lethargy_widths=[0.0] + [1.0] * (G - 1))


@pytest.mark.parametrize(
    "field, bad",
    [
        ("forward_truncation_share", 0.0),
        ("forward_truncation_share", 1.5),
        ("growth_share", 0.0),
        ("gram_solve_min_ratio", 1.0),
        ("gram_floor", 0.0),
        ("resampled_pool", 0),
        ("swept_sample", 0),
        ("forced_per_check", 0),
        ("growth_max_blocks", 0),
        ("sketch_rows", 0),
    ],
)
def test_tuning_fields_validated(field, bad):
    with pytest.raises(RuntimeError):
        _cross("polar|azimuthal|space|energy", **{field: bad})


def test_tuning_fields_take_effect():
    """The former constants are CrossConfig fields with the same defaults, and changing
    one changes the run (a coarser forward truncation keeps fewer forward ranks)."""
    c = CrossConfig()
    assert (c.forward_truncation_share, c.gram_solve_min_ratio, c.gram_floor) == (
        0.1,
        1e-4,
        1e-7,
    )
    assert (c.resampled_pool, c.swept_sample, c.forced_per_check) == (4, 2, 4)
    assert (c.growth_share, c.growth_max_blocks, c.sketch_rows) == (0.5, 4, 32)
    a = _cross("polar|azimuthal|space|energy", tol=1e-6)
    a.run()
    b = _cross("polar|azimuthal|space|energy", tol=1e-6, forward_truncation_share=1.0)
    b.run()
    fwd = lambda c: [h["ranks"] for h in c.history if h["phase"] == "forward"][0]
    assert fwd(b) != fwd(a) or b.num_fibers != a.num_fibers
