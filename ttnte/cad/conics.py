import numpy as np
from igakit.nurbs import NURBS
from igakit.transform import transform

Pi = np.pi
radians = np.radians
degrees = np.degrees


def ellipse(center=None, angle=None, xstretch=1, ystretch=1):
    """
    Construct a NURBS ellipse with semi and major axes along x and y axes. 
    Mirroring "https://github.com/dalcinl/igakit/blob/master/igakit/cad.py"

    Parameters
    ----------
    xstretch, ystretch : float, optional
        half the diameter along x and y axis respectively
    center : array_like, optional
        center of the ellipse
    angle : float or 2-tuple of floats, optional
        angular cut about the center

    Returns
    --------
    ellipse: NURBS surface

    """
    if angle is None:
        # Full circle, 4 knot spans, 9 control points
        spans = 4
        Cw = np.zeros((9,4), dtype='d')
        Cw[:,:2] = [[ 1, 0], [ 1, 1], [ 0, 1],
                    [-1, 1], [-1, 0], [-1,-1],
                    [ 0,-1], [ 1,-1], [ 1, 0]]
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
        x = cos_a
        y = sin_a
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


def qtrlobe(outrad=0.5, portrs=0.1, hfwidth=0.1):
    """
    Construct a quarter of NURBS surface used in the lightbridge fuel problem
    to describe outside edge of the fuel and the cladding. 

    
    Parameters
    --------
    outrad: float, necessary
        describes radius of circle defining edge between lobes
    portrs: float, necessary
        distance lobes extend from their neck
    hfwidth: float, necessary
        half of width at the neck of the lobes

    Returns
    --------
    qtrlobe: NURBS surface
    
    """

    Pi = np.pi
    fuelrad = outrad + portrs + hfwidth #radius of fuel cell should be sum of parameters

    # Compute sweep and number knot spans
    sweep = Pi/2
    spans = 3

    # Construct a single-segment NURBS circular arc
    # centered at the origin and bisected by +X axis
    alpha = sweep/(2)
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)
    tan_a = np.tan(alpha)
    x = cos_a
    y = sin_a
    wm = cos_a
    xm = x + y*tan_a
    Ca = [[    x, -y, 0,  1],
          [wm*xm,  0, 0, wm],
          [    x,  y, 0,  1]]
    Cw = np.empty((2*spans+1,4), dtype='d')
    R = transform().rotate(alpha, 2)
    Ctop, Cmiddle, Cbottom = R(Ca), R(Ca), R(Ca)

    # Stretch control points to match desired shape
    Ptop1 = transform().scale(hfwidth,0)
    Ptop2 = transform().scale(portrs,1)
    Pbottom1 = transform().scale(portrs,0)
    Pbottom2 = transform().scale(hfwidth, 1)
    Pmiddle1 = transform().scale(outrad,0)
    Pmiddle2 = transform().scale(outrad,1)
    Ctop, Cmiddle, Cbottom = Ptop1(Ptop2((Ctop))), Pmiddle1(Pmiddle2((Cmiddle))), Pbottom1(Pbottom2(Cbottom))

    # Move control points to rights spots to define surface
    oneeighty = transform().rotate(Pi, 2)
    Ttop = transform().translate([0,fuelrad-portrs,0])
    Tbottom = transform().translate([fuelrad-portrs,0,0])
    Tmiddle = transform().translate([1*(fuelrad-portrs),1*(fuelrad-portrs),0])
    Ctop, Cmiddle, Cbottom = Ttop(Ctop), Tmiddle(oneeighty(Cmiddle)), Tbottom(Cbottom)

    Cw[0,:] = Ctop[2,:]
    Cw[1,:] = Ctop[1,:]
    Cw[2,:] = Ctop[0,:]
    Cw[3,:] = Cmiddle[1,:]
    Cw[4,:] = Cmiddle[2,:]
    Cw[5,:] = Cbottom[1,:]
    Cw[6,:] = Cbottom[0,:]

    # Compute knot vector in the range [0,1]
    a, b = 0, 1
    U = np.empty(2*(spans+1)+2, dtype='d')
    U[0], U[-1] = a, b
    U[1:-1] = np.linspace(a,b,spans+1).repeat(2)

    return NURBS([U], Cw)