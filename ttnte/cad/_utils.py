import cotengra as ctg
import numpy as np
from igakit.nurbs import NURBS as IgakitNURBS
from numba import njit, prange

# Import geomdl
try:
    from geomdl.core import NURBS as GeomdlNURBS
except ImportError:
    print(
        "Geomdl was not compiled with cython. For best performance refer to"
        + "https://nurbs-python.readthedocs.io/en/5.x/install.html#compile"
        + "-with-cython"
    )
    from geomdl import NURBS as GeomdlNURBS


def _igakit2geomdl(igakit_nurbs: IgakitNURBS):
    """Convert ``igakit.nurbs.NURBS`` to ``geomdl.NURBS.Surface``."""
    # Initialize geomdl NURBS object
    geomdl_nurbs = GeomdlNURBS.Surface()

    # Set degree
    geomdl_nurbs.degree_u, geomdl_nurbs.degree_v = igakit_nurbs.degree

    # Set control points
    geomdl_nurbs.ctrlpts_size_u = igakit_nurbs.points.shape[0]
    geomdl_nurbs.ctrlpts_size_v = igakit_nurbs.points.shape[1]

    # Set control points
    geomdl_nurbs.ctrlpts = igakit_nurbs.points.reshape((-1, 3)).tolist()

    # Set weights
    geomdl_nurbs.weights = igakit_nurbs.weights.flatten().tolist()

    # Set knot vectors
    geomdl_nurbs.knotvector_u = igakit_nurbs.knots[0].tolist()
    geomdl_nurbs.knotvector_v = igakit_nurbs.knots[1].tolist()

    return geomdl_nurbs


# ==========================================================================
# Base NURBS evaluate functions
@njit(cache=True, nogil=True)
def find_span_binsearch(degree, knot_vector, num_ctrlpts, knot, tol):
    # In The NURBS Book; number of knots = m + 1, number of control points = n + 1, p = degree
    # All knot vectors should follow the rule: m = p + n + 1
    n = num_ctrlpts - 1
    if abs(knot_vector[int(n + 1)] - knot) <= tol:
        return int(n)

    # Set max and min positions of the array to be searched
    low = degree
    high = num_ctrlpts

    # The division could return a float value which makes it impossible to use as an array index
    mid = (low + high) / 2
    # Direct int casting would cause numerical errors due to discarding the significand figures (digits after the dot)
    # The round function could return unexpected results, so we add the floating point with some small number
    # This addition would solve the issues caused by the division operation and how Python stores float numbers.
    # E.g. round(13/2) = 6 (expected to see 7)
    mid = int(round(mid + tol))

    # Search for the span
    while (knot < knot_vector[int(mid)]) or (knot >= knot_vector[int(mid + 1)]):
        if knot < knot_vector[int(mid)]:
            high = mid
        else:
            low = mid
        mid = int((low + high) / 2)

    return mid


@njit(cache=True, nogil=True)
def find_span_linear(degree, knot_vector, num_ctrlpts, knot):
    span = degree + 1  # Knot span index starts from zero
    while span < num_ctrlpts and knot_vector[span] <= knot:
        span += 1
    return int(span - 1)


@njit(parallel=True, cache=True, nogil=True)
def find_spans(degree, knot_vector, num_ctrlpts, knots, use_binary=True, tol=1e-5):
    # Number of knots to look at
    n = int(len(knots))

    # Fill array
    spans = np.empty(n, dtype=np.int32)

    if use_binary:
        for i in prange(n):
            spans[i] = find_span_binsearch(
                degree, knot_vector, num_ctrlpts, knots[i], tol
            )
    else:
        for i in prange(n):
            spans[i] = find_span_linear(degree, knot_vector, num_ctrlpts, knots[i])

    return spans


@njit(cache=True, nogil=True)
def basis_function(degree, knot_vector, span, knot):
    """Computes the non-vanishing basis functions for a single parameter.

    Implementation of Algorithm A2.2 from The NURBS Book by Piegl & Tiller.
    Uses recurrence to compute the basis functions, also known as Cox - de
    Boor recursion formula.

    :param degree: degree, :math:`p`
    :type degree: int
    :param knot_vector: knot vector, :math:`U`
    :type knot_vector: list, tuple
    :param span: knot span, :math:`i`
    :type span: int
    :param knot: knot or parameter, :math:`u`
    :type knot: float
    :return: basis functions
    :rtype: list
    """
    left = np.zeros(degree + 1)
    right = np.zeros(degree + 1)
    N = np.ones(degree + 1)

    for j in range(1, degree + 1):
        left[j] = knot - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - knot
        saved = 0.0
        for r in range(0, j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved

    return N


@njit(parallel=True, cache=True, nogil=True)
def basis_functions(degree, knot_vector, spans, knots):
    # Number of knots
    n = len(knots)

    # Fill array
    basis = np.empty((n, degree + 1), dtype=np.float64)

    # Run through all points
    for i in prange(n):
        basis[i, :] = basis_function(degree, knot_vector, spans[i], knots[i])
    return basis
