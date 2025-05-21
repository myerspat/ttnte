from ._dmrg import DMRG


class MALS(DMRG):
    """Simple alias for 2-site DMRG."""

    def __init__(self, **dmrg_opts):
        super().__init__(num_sites=2, **dmrg_opts)
