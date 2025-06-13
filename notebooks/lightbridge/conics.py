import numpy as np
from igakit.nurbs import NURBS
from igakit.transform import transform

Pi = np.pi
radians = np.radians
degrees = np.degrees


def ezellipse(center=None, angle=None, xstretch=1, ystretch=1):
    """
    Construct a NURBS circular arc or full circle

    Parameters
    ----------
    xstretch, ystretch : float, optional
    center : array_like, optional
    angle : float or 2-tuple of floats, optional

    Examples
    --------

    >>> crv = circle()
    >>> crv.shape
    (9,)
    >>> P = crv([0, 0.25, 0.5, 0.75, 1])
    >>> assert np.allclose(P[0], ( 1,  0, 0))
    >>> assert np.allclose(P[1], ( 0,  1, 0))
    >>> assert np.allclose(P[2], (-1,  0, 0))
    >>> assert np.allclose(P[3], ( 0, -1, 0))
    >>> assert np.allclose(P[4], ( 1,  0, 0))

    >>> crv = circle(angle=3*Pi/2)
    >>> crv.shape
    (7,)
    >>> P = crv([0, 1/3., 2/3., 1])
    >>> assert np.allclose(P[0], ( 1,  0, 0))
    >>> assert np.allclose(P[1], ( 0,  1, 0))
    >>> assert np.allclose(P[2], (-1,  0, 0))
    >>> assert np.allclose(P[3], ( 0, -1, 0))

    >>> crv = circle(radius=2, center=(1,1), angle=(Pi/2,-Pi/2))
    >>> crv.shape
    (5,)
    >>> P = crv([0, 0.5, 1])
    >>> assert np.allclose(P[0], (1,  3, 0))
    >>> assert np.allclose(P[1], (3,  1, 0))
    >>> assert np.allclose(P[2], (1, -1, 0))

    >>> crv = circle(radius=3, center=2, angle=Pi/2)
    >>> crv.shape
    (3,)
    >>> P = crv([0, 1])
    >>> assert np.allclose(P[0], ( 5, 0, 0))
    >>> assert np.allclose(P[1], ( 2, 3, 0))

    """
    radius = 1 #to bring in circle code
    if angle is None:
        # Full circle, 4 knot spans, 9 control points
        spans = 4
        Cw = np.zeros((9,4), dtype='d')
        Cw[:,:2] = [[ 1, 0], [ 1, 1], [ 0, 1],
                    [-1, 1], [-1, 0], [-1,-1],
                    [ 0,-1], [ 1,-1], [ 1, 0]]
        Cw[:,:2] *= radius
        wm = np.sqrt(2)/2
        Cw[:,3] = 1; Cw[1::2,:] *= wm
    else:
        Pi = np.pi # inline numpy.pi
        # Determine start and end angles
        if isinstance(angle, (tuple, list)):
            start, end = angle
            if start is None: start = 0
            if end is None: end = 2*Pi
        else:
            start, end = 0, angle
        # Compute sweep and number knot spans
        sweep = end - start
        quadrants = (0.0, Pi/2, Pi, 3*Pi/2)
        spans = np.searchsorted(quadrants, abs(sweep))
        # Construct a single-segment NURBS circular arc
        # centered at the origin and bisected by +X axis
        alpha = sweep/(2*spans)
        sin_a = np.sin(alpha)
        cos_a = np.cos(alpha)
        tan_a = np.tan(alpha)
        x = radius*cos_a
        y = radius*sin_a
        wm = cos_a
        xm = x + y*tan_a
        Ca = [[    x, -y, 0,  1],
              [wm*xm,  0, 0, wm],
              [    x,  y, 0,  1]]
        # Compute control points by successive rotation
        # of the controls points in the first segment
        Cw = np.empty((2*spans+1,4), dtype='d')
        R = transform().rotate(alpha+start, 2)
        Cw[0:3,:] = R(Ca)
        if spans > 1:
            R = transform().rotate(2*alpha, 2)
            for i in range(1, spans):
                n = 2*i+1
                Cw[n:n+2,:] = R(Cw[n-2:n,:])
    #stretch points
    if xstretch is not None:
        T = transform().scale(xstretch,0)
        Cw = T(Cw)
    if ystretch is not None:
        T = transform().scale(ystretch,1)
        Cw = T(Cw)
    # Translate control points to center
    if center is not None:
        T = transform().translate(center)
        Cw = T(Cw)
    # Compute knot vector in the range [0,1]
    a, b = 0, 1
    U = np.empty(2*(spans+1)+2, dtype='d')
    U[0], U[-1] = a, b
    U[1:-1] = np.linspace(a,b,spans+1).repeat(2)
    # Return the new NURBS object
    return NURBS([U], Cw)

def tstellipse(input=np.array([0]), xstretch=1, ystretch=1):
    """Take in a vector of inputs and return the elliptical function of those values."""
    output = []  # Initialize an empty list to store results
    
    for x in input:
        # Calculate y using the ellipse formula, ensure correct square root handling
        inside_sqrt = 1 - ((x / xstretch) ** 2)
        if inside_sqrt >= 0:
            y = np.sqrt(inside_sqrt) * ystretch
        else:
            y = 0.0  # Or handle as appropriate if negative
        
        output.append(y)  # Append y to the list
    
    return np.array(output)  # Convert list to a NumPy array