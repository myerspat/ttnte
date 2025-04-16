from typing import Literal, Union


class IncidentBoundary(object):
    def __init__(
        self,
        from_patch: Union[int, str],
        orientation: Literal[-1, 1],
    ):
        """"""
        self.from_patch = from_patch
        self.orientation = orientation
