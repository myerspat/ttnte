import pytest
import torch
from igakit.cad import refine, line

from ttnte import mpi_context
from ttnte.xs import Server, Material
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    DGTransportAssemblerConfig,
    DIGAFirstOrderTransportAssembler1D,
    FixedSource,
)
from ttnte.math import QuadratureSet1D

test_params = [
    ("cpu", torch.float64),
]


def _pure_absorber(device, dtype):
    """A 1-group, non-fissile material -- fixed sources on a fissile fill are rejected
    at assemble() time (combined fixed-source + fission/subcritical- multiplication
    problems aren't supported yet), so these tests need a genuinely non-fissile material
    rather than the fissile pu239 benchmark."""
    server = Server()
    mat = Material("Absorber")
    mat.chi = torch.zeros(1, dtype=dtype, device=device)
    mat.total = torch.tensor([0.32640], dtype=dtype, device=device)
    mat.nu_fission = torch.zeros(1, dtype=dtype, device=device)
    mat.fission = torch.zeros(1, dtype=dtype, device=device)
    mat.absorption = torch.tensor([0.101184], dtype=dtype, device=device)
    mat.scatter_gtg = torch.tensor([[[0.225216]]], dtype=dtype, device=device)
    mat.finalize()
    server.add_material(mat)
    server.finalize()
    return mat.label, server


def _make_slab_assembler(device, dtype, source, qset=None):
    """Build a single-patch 1-D slab (purely-absorbing, 1-group) with a FixedSource
    attached, and return (assembler, half_length) after assemble()."""
    mpi_context.init()
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)

    fill, xs_server = _pure_absorber(device, dtype)

    rc = 2.256751
    patch = Patch.from_igakit(
        refine(line((-rc, 0), (rc, 0)), 10, 3),
        device=device,
        dtype=dtype,
        fill=fill,
    )
    patch.source = source

    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()

    if qset is None:
        qset = QuadratureSet1D.gauss_legendre(64)
        qset.to_(device, dtype)

    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-10
    config.cross.eps = config.rounding.eps
    config.max_dense_size = 0
    config.cross_jacobian_inverse = True

    assembler = DIGAFirstOrderTransportAssembler1D(patch, qset, xs_server, config)
    assembler.assemble()
    return assembler, qset, rc


@pytest.mark.parametrize("device, dtype", test_params)
def test_isotropic_source_matches_physical_strength(device, dtype):
    """An isotropic source of strength Q, integrated over angle and summed over the DG
    basis, must recover exactly Q * volume -- the defining property of specifying source
    strength in physical units."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device)

    Q = 3.7
    source = FixedSource(isotropic_strength=torch.tensor([Q], dtype=dtype))
    assembler, qset, rc = _make_slab_assembler(device, dtype, source)

    assert assembler.source.is_tt

    scalar_source = qset.integrate(assembler.source, 1e-12, 10**9)
    total = scalar_source.to_dense().sum().item()

    expected = Q * (2 * rc)
    assert total == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("device, dtype", test_params)
def test_function_source_matches_isotropic_for_constant_function(device, dtype):
    """A cross-check on the MMS/arbitrary-function branch's much more involved
    composition (coordinate-grid TT-cross + basis projection):

    if the supplied function is the same constant everywhere, the result must
    match the isotropic branch's output pointwise, since both represent the
    exact same physical source.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device)

    Q = 3.7
    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(device, dtype)

    iso_source = FixedSource(isotropic_strength=torch.tensor([Q], dtype=dtype))
    iso_assembler, _, _ = _make_slab_assembler(device, dtype, iso_source, qset)

    def const_func(coords):
        x = coords[1]
        return torch.full_like(x, Q)

    func_source = FixedSource(function=const_func)
    func_assembler, _, _ = _make_slab_assembler(device, dtype, func_source, qset)

    dense_iso = iso_assembler.source.to_dense()
    dense_func = func_assembler.source.to_dense()

    torch.testing.assert_close(dense_func, dense_iso, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("device, dtype", test_params)
def test_no_source_leaves_source_state_undefined(device, dtype):
    """A patch with no FixedSource attached must leave the assembler's source undefined
    -- the existing eigenvalue path (EigenSource) must be unaffected by this feature
    when no fixed source is present."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device)

    mpi_context.init()
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)

    fill, xs_server = _pure_absorber(device, dtype)
    rc = 2.256751
    patch = Patch.from_igakit(
        refine(line((-rc, 0), (rc, 0)), 10, 3),
        device=device,
        dtype=dtype,
        fill=fill,
    )
    assert patch.source is None

    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()

    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(device, dtype)
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-10
    config.cross.eps = config.rounding.eps
    config.max_dense_size = 0
    config.cross_jacobian_inverse = True

    assembler = DIGAFirstOrderTransportAssembler1D(patch, qset, xs_server, config)
    assembler.assemble()

    assert not assembler.source.defined()


@pytest.mark.parametrize("device, dtype", test_params)
def test_fixed_source_on_fissile_fill_raises(device, dtype):
    """A fixed source attached to a fissile fill must fail loudly at assemble() time --
    combined fixed-source + fission (subcritical multiplication) problems aren't
    supported yet, and silently dropping the fixed source (or solving with a never-
    updated EigenSource) would be much worse than an explicit error."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device)

    from ttnte.xs.benchmarks import pu239

    fills, xs_server = pu239(num_groups=1, device=device, dtype=dtype)
    rc = 2.256751
    patch = Patch.from_igakit(
        refine(line((-rc, 0), (rc, 0)), 10, 3),
        device=device,
        dtype=dtype,
        fill=fills[0],
    )
    patch.source = FixedSource(isotropic_strength=torch.tensor([1.0], dtype=dtype))

    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()

    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(device, dtype)
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-10
    config.cross.eps = config.rounding.eps
    config.max_dense_size = 0
    config.cross_jacobian_inverse = True

    assembler = DIGAFirstOrderTransportAssembler1D(patch, qset, xs_server, config)
    with pytest.raises(RuntimeError, match="fissile"):
        assembler.assemble()


# ---------------------------------------------------------------------------
# projected_isotropic_source: a pre-projected, angle-independent load
# ---------------------------------------------------------------------------
def _projected_from_source(source_state, num_angular):
    """Extract the dense (space x NumDim, energy) load from an assembled isotropic
    source State: the value on ordinate 0 (every ordinate carries the same load)."""
    num_cores = len(source_state.as_tt().cores)
    # to_dense() gives the row modes followed by a size-1 column mode per core.
    dense = source_state.to_dense()
    dense = dense.reshape(dense.shape[:num_cores])
    return dense[(0,) * num_angular].clone()


def _two_group_scatterer(device, dtype):
    server = Server()
    mat = Material("TwoGroup")
    mat.chi = torch.zeros(2, dtype=dtype, device=device)
    mat.total = torch.tensor([0.5, 1.2], dtype=dtype, device=device)
    mat.nu_fission = torch.zeros(2, dtype=dtype, device=device)
    mat.fission = torch.zeros(2, dtype=dtype, device=device)
    mat.absorption = torch.tensor([0.1, 0.6], dtype=dtype, device=device)
    mat.scatter_gtg = torch.tensor(
        [[[0.3, 0.1], [0.0, 0.6]]], dtype=dtype, device=device
    )
    mat.finalize()
    server.add_material(mat)
    server.finalize()
    return mat.label, server


def _make_square_assembler(device, dtype, source):
    """Single-patch 2-D square, 2-group non-fissile material, with `source` attached;
    returns the assembled DIGAFirstOrderTransportAssembler2D."""
    from igakit.cad import ruled
    from ttnte.math import ProductQuadrature
    from ttnte.physics import DIGAFirstOrderTransportAssembler2D

    mpi_context.init()
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)

    fill, xs_server = _two_group_scatterer(device, dtype)
    L = 2.0
    patch = Patch.from_igakit(
        refine(ruled(line((0, 0), (L, 0)), line((0, L), (L, L))), [4, 5], [2, 2]),
        device=device,
        dtype=dtype,
        fill=fill,
    )
    patch.source = source

    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()

    qset = ProductQuadrature.gauss_legendre_chebyshev(2, 4, 2)
    qset.to_(device, dtype)

    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-12
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10)
    config.cross_jacobian_inverse = True

    assembler = DIGAFirstOrderTransportAssembler2D(patch, qset, xs_server, config)
    assembler.assemble()
    return assembler, patch


def _rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("device, dtype", test_params)
def test_projected_isotropic_source_round_trip_1d(device, dtype):
    """The load extracted from an isotropic_strength source (ordinate 0), fed back via
    projected_isotropic_source on an otherwise identical block, must reproduce the same
    assembled source."""
    device = torch.device(device)
    Q = 3.7
    qset = QuadratureSet1D.gauss_legendre(16)
    qset.to_(device, dtype)

    iso, _, _ = _make_slab_assembler(
        device, dtype, FixedSource(isotropic_strength=torch.tensor([Q])), qset
    )
    projected = _projected_from_source(iso.source, num_angular=1)
    assert projected.dim() == 2

    load, _, _ = _make_slab_assembler(
        device, dtype, FixedSource(projected_isotropic_source=projected), qset
    )
    assert _rel_err(load.source.to_dense(), iso.source.to_dense()) < 1e-12


@pytest.mark.parametrize("device, dtype", test_params)
def test_projected_isotropic_source_round_trip_2d_spatially_varying(device, dtype):
    """2-D, 2-group, spatially varying (in x and y, different per group) but angle-
    independent `function` source: its load round-trips through
    projected_isotropic_source to the same assembled source, so a genuinely non-constant
    load (not just a constant per group) is carried through."""
    device = torch.device(device)

    def q(coords):
        # coords: [polar, azimuthal, x, y, group]
        x, y, g = coords[2], coords[3], coords[4]
        return (1.0 + 0.5 * x + 0.25 * x * y) * (1.0 + 2.0 * g)

    func, patch = _make_square_assembler(device, dtype, FixedSource(function=q))
    dense = func.source.to_dense()

    # Sanity: the load really varies in space, and is identical on every ordinate.
    spatial = dense[0, 0, :, :, 0]
    assert (spatial.max() - spatial.min()) / spatial.abs().max() > 0.1
    assert _rel_err(dense, dense[:1, :1].expand_as(dense)) < 1e-10

    projected = _projected_from_source(func.source, num_angular=2)
    assert list(projected.shape) == [dense.shape[2], dense.shape[3], 2]

    load, _ = _make_square_assembler(
        device, dtype, FixedSource(projected_isotropic_source=projected)
    )
    assert _rel_err(load.source.to_dense(), dense) < 1e-12

    # Both kinds of source on one block sum.
    both, _ = _make_square_assembler(
        device, dtype, FixedSource(function=q, projected_isotropic_source=projected)
    )
    assert _rel_err(both.source.to_dense(), 2.0 * dense) < 1e-10


@pytest.mark.parametrize("device, dtype", test_params)
def test_projected_isotropic_source_property_round_trip(device, dtype):
    src = FixedSource(projected_isotropic_source=torch.ones(3, 2))
    assert src.defined
    assert list(src.projected_isotropic_source.shape) == [3, 2]
    src.projected_isotropic_source = None
    assert not src.defined


@pytest.mark.parametrize("device, dtype", test_params)
@pytest.mark.parametrize(
    "case, match",
    [
        ("dims", "NumDim \\+ 1"),
        ("mode", "control points"),
        ("energy", "energy groups"),
    ],
)
def test_projected_isotropic_source_validation(device, dtype, case, match):
    device = torch.device(device)
    qset = QuadratureSet1D.gauss_legendre(8)
    qset.to_(device, dtype)
    iso, _, _ = _make_slab_assembler(
        device, dtype, FixedSource(isotropic_strength=torch.tensor([1.0])), qset
    )
    good = _projected_from_source(iso.source, num_angular=1)
    n, G = good.shape

    if case == "dims":
        projected = good.reshape(n, G, 1)
    elif case == "mode":
        projected = torch.ones(n + 1, G)
    elif case == "energy":
        projected = torch.ones(n, G + 1)

    with pytest.raises(RuntimeError, match=match):
        _make_slab_assembler(
            device, dtype, FixedSource(projected_isotropic_source=projected), qset
        )


@pytest.mark.parametrize("device, dtype", test_params)
def test_projected_isotropic_source_rejected_on_incident_face(device, dtype):
    device = torch.device(device)
    mpi_context.init()
    torch.set_default_dtype(dtype)
    fill, xs_server = _pure_absorber(device, dtype)
    patch = Patch.from_igakit(
        refine(line((0, 0), (1, 0)), 4, 3), device=device, dtype=dtype, fill=fill
    )
    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()
    patch.set_boundary_source(
        0, False, FixedSource(projected_isotropic_source=torch.ones(1, 1))
    )
    qset = QuadratureSet1D.gauss_legendre(8)
    qset.to_(device, dtype)
    config = DGTransportAssemblerConfig()
    config.max_dense_size = 0
    assembler = DIGAFirstOrderTransportAssembler1D(patch, qset, xs_server, config)
    with pytest.raises(RuntimeError, match="INCIDENT"):
        assembler.assemble()
