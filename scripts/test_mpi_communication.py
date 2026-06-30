import torch
import numpy as np
from igakit import cad

from ttnte import mpi_context
from ttnte.xs import Material, Server
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import BoundaryType, BCPlane
from ttnte.parallel import IGALoadBalancer, IGADofHeuristic
from ttnte.driver import IGATransportDriver


def create_xs_server(device, dtype):
    # Source material
    source = Material("source")
    source.total = torch.tensor([1.0], device=device, dtype=dtype)
    source.scatter_gtg = torch.tensor([[[1.0 * 0.9]]], device=device, dtype=dtype)
    source.finalize()

    # Void material
    void = Material("void")
    void.total = torch.tensor([0.0], device=device, dtype=dtype)
    void.scatter_gtg = torch.tensor([[[0.0]]], device=device, dtype=dtype)
    void.finalize()

    # Create server and add materials
    server = Server()
    server.add_material(source)
    server.add_material(void)
    server.finalize()

    return [source.label, void.label], server


def create_coarse_patches(device, dtype):
    # Create quarter circle NURBS surface
    radius = 5  # cm
    pitch = 12  # cm
    c0 = cad.circle(radius=radius, angle=np.pi / 2)
    c1 = c0.slice(0, 0, 0.5)
    c2 = c0.slice(0, 0.5, 1)
    l0 = cad.line(p0=(0, 0), p1=(0, 0))

    # Create water patch
    l1 = cad.line(p0=(pitch / 2, 0), p1=(pitch / 2, pitch / 2))
    l2 = cad.line(p0=(pitch / 2, pitch / 2), p1=(0, pitch / 2))

    # Create igakit volumes
    source = [cad.ruled(l0, c1), cad.ruled(l0, c2)]
    void = [cad.ruled(c1, l1), cad.ruled(c2, l2)]

    # Extrude all surface
    zdelta = 10  # cm
    source = [cad.extrude(s, displ=zdelta, axis=2) for s in source]
    void = [cad.extrude(s, displ=zdelta, axis=2) for s in void]

    # Convert to ttnte patches
    source = [Patch.from_igakit(s, device=device, dtype=dtype) for s in source]
    void = [Patch.from_igakit(s, device=device, dtype=dtype) for s in void]

    return source, void


if __name__ == "__main__":
    # Machine parameters
    num_cpus = int(384 * 2)
    num_gpus = int(4)

    # Initialize MPI
    mpi_context.init()

    # Set defaults for PyTorch
    torch.set_default_dtype(torch.float64)
    torch.autograd.set_grad_enabled(False)
    torch.set_num_threads(int(num_cpus / mpi_context.world_size))
    print(f"Rank {mpi_context.rank} initialized with {torch.get_num_threads()} threads")

    device = torch.get_default_device()
    dtype = torch.float64

    # Get XS info and patches
    (slabel, vlabel), server = create_xs_server(device, dtype)
    spatch, vpatch = create_coarse_patches(device, dtype)

    # Set the material type for the patches
    for s, v in zip(spatch, vpatch):
        s.fill = slabel
        v.fill = vlabel
    patches = spatch + vpatch

    # Create mesh object
    mesh = IGAMesh(mpi_context)
    mesh.reserve(len(patches))
    for p in patches:
        mesh.add_block(p)

    # Mark internal interfaces and connect them
    mesh.connect()

    # Set reflective boundary conditions
    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True, y_min=True, z_min=True, z_max=True),
        BoundaryType.REFLECTIVE,
        tol=1e-5,
    )

    # Finalize the mesh
    mesh.finalize()

    # Create the driver
    driver = IGATransportDriver(mesh, server, mpi_context)

    driver.distribute([IGADofHeuristic()])

    print(f"Rank {mpi_context.rank} mesh:\n{driver.mesh}")
