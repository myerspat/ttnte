from ._dmrg import DMRG


class ALS(DMRG):
    """Simple alias for 1-site DMRG."""

    def __init__(self, **dmrg_opts):
        super().__init__(num_sites=1, **dmrg_opts)
