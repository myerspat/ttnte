import pickle
from pathlib import Path

import numpy as np
from igakit import cad

from ttnte.iga import IGAMesh
from ttnte.cad.curves import qtrlobe


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

    # ================================================
    # Lightbridge problem
    D = 1.26  # Fuel width
    D2 = D * 0.5
    X = 1.36  # Channel pitch
    delta = 0.306  # Width of lobes
    y2 = delta * 0.5
    d = 0.04  # Thickness of cladding at valleys
    dmax = 0.102  # Thickness of cladding at ends of the lobes
    R = 0.297  # Radius defining outer curve of valleys
    a = 0.156  # Displacer width

    y1 = y2 - d  # Half of width of inner lobe
    x1 = D2 - R - y2 - dmax  # Portrusion of innerlobe
    x2 = x1 + dmax  # Portrusion of outer lobe

    # NURBS curves
    origin = cad.line(p0=(0, 0), p1=(0, 0))
    burn = cad.line(p1=(a / (2**0.5), 0), p0=(0, a / (2**0.5)))
    fuel = qtrlobe(outrad=R + d, portrs=x1, hfwidth=y1)
    clad = qtrlobe(outrad=R, portrs=x2, hfwidth=y2)
    topedge = cad.line(p0=(0, X / 2), p1=(X / 2, X / 2))
    corner = cad.line(p1=(X / 2, X / 2), p0=(X / 2, X / 2))
    rightedge = cad.line(p1=(X / 2, 0), p0=(X / 2, X / 2))

    # NURBS patches
    patches = {}
    sections = [0, 1 / 3, 2 / 3, 1]
    edges = [topedge, corner, rightedge]

    for i in range(len(sections) - 1):
        # Line sections
        osec = origin.slice(0, sections[i], sections[i + 1])
        bsec = burn.slice(0, sections[i], sections[i + 1])
        fsec = fuel.slice(0, sections[i], sections[i + 1])
        csec = clad.slice(0, sections[i], sections[i + 1])

        # Create patches
        patches[cad.ruled(osec, bsec)] = "BA (UO2 FA)"
        patches[cad.ruled(bsec, fsec)] = "UO2 3%"
        patches[cad.ruled(fsec, csec)] = "Guide Tube"
        patches[cad.ruled(csec, edges[i])] = "Water"

    # Create IGA mesh object
    mesh = IGAMesh(patches)

    # Refine mesh
    for p in range(mesh.num_patches):
        mesh.refine(p, 3, 2)

    # Finalize mesh
    mesh.connect()

    # Set reflective boundary conditions
    mesh.set_reflective_condition(("left", "bottom", "top", "right"))

    # Finalize mesh
    mesh.finalize()

    # Test points
    pids, _ = mesh.inverse_map(
        np.array([[0, 0.53125], [0, 0.5525], [0.57375, 0]]).T,
        tol=tol,
    )
    assert (pids[:-1] == 2).all()
    assert pids[-1] == 10


def test_normals():
    # Read in problematic patch
    with open(Path(__file__).parent / "supporting/patch.pkl", "rb") as f:
        patches = {pickle.load(f): "None"}

    # Create IGA mesh object
    mesh = IGAMesh(patches)

    # Finalize mesh
    mesh.connect()

    # Finalize mesh
    mesh.finalize()

    # Test vector
    i = np.array([1, 0])

    # Check the normal for the NURBS(xhat, 0)
    vec = mesh.normal(0, np.array([[0.5], [0]]))[1].flatten()

    # Check the vectors point in -i
    assert 1 == np.dot(-i, vec)
