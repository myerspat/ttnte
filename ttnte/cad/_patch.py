from typing import Optional

import torch
from igakit.cad import NURBS

from ttnte.cpp.ttnte_python.xs import MaterialLabel
from ttnte.cpp.ttnte_python.cad import BSplineBasis, Patch


@staticmethod
def from_igakit(
    kitpatch: NURBS,
    label: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    fill: Optional[MaterialLabel] = None,
):
    """Build a ``ttnte.cad.Patch`` from an ``igakit.cad.NURBS``.

    Parameters
    ----------
    kitpatch: igakit.cad.NURBS
        Input geometric patch.
    label: str, optional
        Name of the patch.
    device: torch.device, default=torch.get_default_device()
        Device to initialize the patch on.
    dtype: torch.dtype, default=torch.get_default_dtype()
        Data type to initialize the control points and knot vectors as.
    fill: torch.xs.MaterialLabel, optional
        Fill of the patch.

    Return
    ------
    patch: ttnte.cad.Patch
        Valid patch.
    """
    # Evaluate dynamic defaults
    device = device if device is not None else torch.get_default_device()
    dtype = dtype if dtype is not None else torch.get_default_dtype()

    patch = Patch(label)

    # Create the basis
    basis = [
        BSplineBasis(
            torch.tensor(kitpatch.knots[i], device=device, dtype=dtype),
            kitpatch.degree[i],
        )
        for i in range(kitpatch.dim)
    ]
    patch.set_basis(basis, False)

    # If not all weights are one then we make a NURBS else B-Spline
    if (kitpatch.control[..., -1] != 1.0).any():
        patch.set_ctrlptsw(
            torch.tensor(kitpatch.control, device=device, dtype=dtype), False
        )
    else:
        patch.set_ctrlpts(
            torch.tensor(kitpatch.control[..., :-1], device=device, dtype=dtype), False
        )

    # Add fill
    if fill is not None:
        patch.fill = fill

    # Validate the patch
    patch.validate()

    return patch


# Add methods to the patch class
Patch.from_igakit = from_igakit
