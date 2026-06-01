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
    drop_invariant_dims: bool = True,
    tol: float = 1e-10,
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
    drop_invariant_dims: bool, default=True
        Drop spatial dimensions that do not vary by control point.
    tol: float, default=1e-10
        The tolerance for checking if this is a B-spline or NURBS and
        which dimensions do not very.

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
    patch.set_basis(basis)

    # Convert the control points to a pytorch tensor
    ctrlptsw = torch.tensor(kitpatch.control, device=device, dtype=dtype)

    if drop_invariant_dims:
        # Separate and un-weight control points from their weights
        weights = ctrlptsw[..., -1:]
        ctrlpts = ctrlptsw[..., :-1] / weights

        # Flatten and check variation
        flat_ctrlpts = ctrlpts.reshape((-1, ctrlpts.shape[-1]))
        variation = flat_ctrlpts.max(dim=0).values - flat_ctrlpts.min(dim=0).values
        active_dims = variation > tol

        # Check this is not a single point patch
        if not active_dims.any():
            active_dims[0] = True

        # Append True to keep the weight dimension at the end
        keep_indices = torch.cat([active_dims, torch.tensor([True], device=device)])

        # Filter the control tensor
        ctrlptsw = ctrlptsw[..., keep_indices]

    # Set control points for B-spline or NURBS
    if (torch.abs(ctrlptsw[..., -1] - 1.0) < tol).all():
        # B-spline
        patch.set_ctrlpts(ctrlptsw[..., :-1])
    else:
        # NURBS
        patch.set_ctrlptsw(ctrlptsw)

    # Add fill
    if fill is not None:
        patch.fill = fill

    # Validate the patch
    patch.finalize()

    return patch


# Add methods to the patch class
Patch.from_igakit = from_igakit
