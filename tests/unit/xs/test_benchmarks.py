import math

import torch

from ttnte.xs.benchmarks import kaist


def test_kaist():
    # Read XS data
    labels, xs_server = kaist(problem="2B")

    # Check materials
    names = {label.to_string() for label in labels}
    assert names == {
        "MOX 7%",
        "UO2 2%",
        "UO2 3%",
        "BA (UO2 FA)",
        "BA (MOX FA)",
        "Control Rod",
        "Guide Tube",
        "Gas",
        "Water",
        "Baffle",
        "Reflector",
    }

    def get(name):
        label = next(l for l in labels if l.to_string() == name)
        return xs_server.get_material(label)

    # Group structure: the library's edges, the 0 eV edge as the cutoff
    bounds = [2.0e7, 1.353e6, 9.119e3, 3.928, 0.6251, 0.1457, 0.05692, 1e-5]
    assert xs_server.group_bounds == bounds
    widths = xs_server.lethargy_widths
    assert len(widths) == 7
    assert all(
        abs(w - math.log(bounds[g] / bounds[g + 1])) < 1e-12
        for g, w in enumerate(widths)
    )

    # Check some XS data
    assert torch.allclose(
        get("UO2 2%").chi,
        torch.tensor(
            [5.9252e-01, 4.0714e-01, 3.3193e-04, 0.0, 0.0, 0.0, 0.0],
            dtype=get("UO2 2%").chi.dtype,
        ),
    )
    assert torch.allclose(
        get("UO2 2%").nu_fission[3:7],
        torch.tensor(
            [4.1982e-02, 1.8488e-01, 3.0967e-01, 6.2433e-01],
            dtype=get("UO2 2%").nu_fission.dtype,
        ),
    )
    # Non-fissile materials have all-zero chi/nu_fission, which Material
    # treats as a no-op (leaving the underlying tensor unset -> None)
    assert get("Gas").nu_fission is None
    assert get("Gas").chi is None
    assert not get("Gas").is_fissile()
    assert get("UO2 2%").is_fissile()

    assert torch.equal(
        get("Reflector").scatter_gtg[0,],
        torch.tensor(
            [
                [
                    8.2716e-02,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    8.1963e-02,
                    4.7143e-01,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    5.1642e-04,
                    9.9730e-02,
                    9.5552e-01,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    0.0000e00,
                    0.0000e00,
                    1.1014e-01,
                    7.0714e-01,
                    2.3861e-03,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    0.0000e00,
                    0.0000e00,
                    1.5751e-02,
                    3.5409e-01,
                    8.9203e-01,
                    2.1673e-01,
                    9.8237e-02,
                ],
                [
                    0.0000e00,
                    0.0000e00,
                    2.8683e-03,
                    4.9958e-02,
                    4.2845e-01,
                    1.1939e00,
                    6.4545e-01,
                ],
                [
                    0.0000e00,
                    0.0000e00,
                    1.7862e-03,
                    2.2326e-02,
                    1.3581e-01,
                    4.3269e-01,
                    2.0233e00,
                ],
            ],
            dtype=get("Reflector").scatter_gtg.dtype,
        ),
    )
    assert torch.equal(
        get("Control Rod").scatter_gtg[1,],
        torch.tensor(
            [
                [
                    4.8620e-03,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    -1.3112e-03,
                    6.7646e-03,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    0.0000e00,
                    -4.4387e-04,
                    5.4865e-03,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    0.0000e00,
                    0.0000e00,
                    -1.4496e-04,
                    6.8217e-03,
                    3.2670e-05,
                    0.0000e00,
                    0.0000e00,
                ],
                [
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    -1.4191e-03,
                    7.4662e-03,
                    -1.1229e-03,
                    -6.2904e-04,
                ],
                [
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    -1.8585e-03,
                    8.7547e-03,
                    -2.0484e-03,
                ],
                [
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    -1.2332e-04,
                    -2.0733e-03,
                    7.9118e-03,
                ],
            ],
            dtype=get("Control Rod").scatter_gtg.dtype,
        ),
    )
