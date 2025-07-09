import numpy as np
from igakit import cad

from ttnte.iga import IGAMesh


def test_inverse_map():
    # Create quarter circle NURBS surface
    radius = 0.54  # cm
    pitch = 1.26  # cm
    c0 = cad.circle(radius=radius, angle=np.pi / 2)
    c1 = c0.slice(0, 0, 0.5)
    c2 = c0.slice(0, 0.5, 1)
    l0 = cad.line(p0=(0, 0), p1=(0, 0))
    fuel1 = cad.ruled(l0, c1)
    fuel2 = cad.ruled(l0, c2)

    # Create water patch
    l1 = cad.line(p0=(pitch / 2, 0), p1=(pitch / 2, pitch / 2))
    l2 = cad.line(p0=(pitch / 2, pitch / 2), p1=(0, pitch / 2))
    mod1 = cad.ruled(c1, l1)
    mod2 = cad.ruled(c2, l2)

    # NURBS surfaces
    patches = {}
    patches[fuel1] = "UO2"
    patches[fuel2] = "UO2"
    patches[mod1] = "Water"
    patches[mod2] = "Water"

    # Create IGA mesh object
    mesh = IGAMesh(patches)

    # Refine mesh resolution
    mesh.refine(0, 2, degree=2)

    # Finalize mesh
    mesh.connect()

    # Finalize mesh
    mesh.finalize()

    # Iterate through each patch and sample random parametric coordinates
    tol = 1e-8
    for p in range(mesh.num_patches):
        # Create random parametric coordinates
        rand_coords = np.random.rand(10, 2)

        # Evaluate physical coordinates and inverse map to get parametric coords
        pids, coords = mesh.inverse_map(
            np.array(mesh.patches[p].evaluate_list(rand_coords.tolist()))[:, :-1].T,
            tol=tol,
        )

        # Check tolerances
        assert (pids == p).all()
        assert (
            np.sqrt(
                np.sum(
                    (
                        np.array(mesh.patches[p].evaluate_list(rand_coords.tolist()))[
                            :, :-1
                        ].T
                        - np.array(mesh.patches[p].evaluate_list(coords.T.tolist()))[
                            :, :-1
                        ].T
                    )
                    ** 2,
                    axis=-1,
                )
            )
            < tol * 5
        ).all()
