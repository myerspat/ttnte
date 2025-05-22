from typing import Literal, Union


class Boundary(object):
    """
    Patch boundary object.

    Attributes
    ----------
    p: int or None
        Patch index.
    orientation: 1 or -1
        Mapping orientation.
    boundary_idx: int
        Index into boundaries.
    """

    def __init__(
        self,
        p: Union[None, int],
        orientation: Literal[-1, 1],
        boundary_idx: int,
    ):
        """
        Create Boundary object.

        Parameters
        ----------
        p: int or None
            Patch index.
        orientation: 1 or -1
            Mapping orientation.
        boundary_idx: int
            Index into boundaries.
        """
        self.p = p
        self.orientation = orientation
        self.boundary_idx = boundary_idx
