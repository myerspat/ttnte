import functools

from autoray import do
from cupyx.scipy.sparse.linalg import LinearOperator as CupyLinearOperator
from quimb.tensor.tensor_core import (
    Tensor,
    TensorNetwork,
    get_tensor_linop_backend,
    oset_union,
    prod,
    tensor_contract,
    tensor_split,
)
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator


class TNLinearOperator(CupyLinearOperator, ScipyLinearOperator):
    r"""Get a linear operator - something that replicates the matrix-vector
    operation - for an arbitrary uncontracted TensorNetwork, e.g::

                 : --O--O--+ +-- :                 --+
                 :   |     | |   :                   |
                 : --O--O--O-O-- :    acting on    --V
                 :   |     |     :                   |
                 : --+     +---- :                 --+
        left_inds^               ^right_inds

    This can then be supplied to scipy's sparse linear algebra routines.
    The ``left_inds`` / ``right_inds`` convention is that the linear operator
    will have shape matching ``(*left_inds, *right_inds)``, so that the
    ``right_inds`` are those that will be contracted in a normal
    matvec / matmat operation::

        _matvec =    --0--v    , _rmatvec =     v--0--

    Parameters
    ----------
    tns : sequence of Tensors or TensorNetwork
        A representation of the hamiltonian
    left_inds : sequence of str
        The 'left' inds of the effective hamiltonian network.
    right_inds : sequence of str
        The 'right' inds of the effective hamiltonian network. These should be
        ordered the same way as ``left_inds``.
    ldims : tuple of int, or None
        The dimensions corresponding to left_inds. Will figure out if None.
    rdims : tuple of int, or None
        The dimensions corresponding to right_inds. Will figure out if None.
    optimize : str, optional
        The path optimizer to use for the 'matrix-vector' contraction.
    backend : str, optional
        The array backend to use for the 'matrix-vector' contraction.
    is_conj : bool, optional
        Whether this object should represent the *adjoint* operator.

    See Also
    --------
    TNLinearOperator1D
    """

    def __init__(
        self,
        tns,
        left_inds,
        right_inds,
        ldims=None,
        rdims=None,
        optimize=None,
        backend=None,
        is_conj=False,
    ):
        if backend is None:
            self.backend = get_tensor_linop_backend()
        else:
            self.backend = backend
        self.optimize = optimize

        if isinstance(tns, TensorNetwork):
            self._tensors = tns.tensors

            if ldims is None or rdims is None:
                ix_sz = tns.ind_sizes()
                ldims = tuple(ix_sz[i] for i in left_inds)
                rdims = tuple(ix_sz[i] for i in right_inds)

        else:
            self._tensors = tuple(tns)

            if ldims is None or rdims is None:
                ix_sz = dict(concat((zip(t.inds, t.shape) for t in tns)))
                ldims = tuple(ix_sz[i] for i in left_inds)
                rdims = tuple(ix_sz[i] for i in right_inds)

        self.left_inds, self.right_inds = left_inds, right_inds
        self.ldims, ld = ldims, prod(ldims)
        self.rdims, rd = rdims, prod(rdims)
        self.tags = oset_union(t.tags for t in self._tensors)

        self._kws = {
            "get": "expression",
            "constants": range(len(self._tensors)),
        }
        self._ins = ()

        # conjugate inputs/ouputs rather all tensors if necessary
        self.is_conj = is_conj
        self._conj_linop = None
        self._adjoint_linop = None
        self._transpose_linop = None
        self._contractors = dict()

        super().__init__(dtype=self._tensors[0].dtype, shape=(ld, rd))

    def _matvec(self, vec):
        in_data = do("reshape", vec, self.rdims)

        if self.is_conj:
            in_data = conj(in_data)

        # cache the contractor
        if "matvec" not in self._contractors:
            # generate a expression that acts directly on the data
            iT = Tensor(in_data, inds=self.right_inds)
            self._contractors["matvec"] = tensor_contract(
                *self._tensors,
                iT,
                output_inds=self.left_inds,
                optimize=self.optimize,
                **self._kws,
            )

        fn = self._contractors["matvec"]
        out_data = fn(*self._ins, in_data, backend=self.backend)

        if self.is_conj:
            out_data = conj(out_data)

        return out_data.ravel()

    def _matmat(self, mat):
        d = mat.shape[-1]
        in_data = do("reshape", mat, (*self.rdims, d))

        if self.is_conj:
            in_data = conj(in_data)

        # for matmat need different contraction scheme for different d sizes
        key = f"matmat_{d}"

        # cache the contractor
        if key not in self._contractors:
            # generate a expression that acts directly on the data
            iT = Tensor(in_data, inds=(*self.right_inds, "_mat_ix"))
            o_ix = (*self.left_inds, "_mat_ix")
            self._contractors[key] = tensor_contract(
                *self._tensors,
                iT,
                output_inds=o_ix,
                optimize=self.optimize,
                **self._kws,
            )

        fn = self._contractors[key]
        out_data = fn(*self._ins, in_data, backend=self.backend)

        if self.is_conj:
            out_data = conj(out_data)

        return do("reshape", out_data, (-1, d))

    def trace(self):
        if "trace" not in self._contractors:
            tn = TensorNetwork(self._tensors)
            self._contractors["trace"] = tn.trace(
                self.left_inds, self.right_inds, optimize=self.optimize
            )
        return self._contractors["trace"]

    def copy(self, conj=False, transpose=False):
        if transpose:
            inds = self.right_inds, self.left_inds
            dims = self.rdims, self.ldims
        else:
            inds = self.left_inds, self.right_inds
            dims = self.ldims, self.rdims

        if conj:
            is_conj = not self.is_conj
        else:
            is_conj = self.is_conj

        return TNLinearOperator(
            self._tensors,
            *inds,
            *dims,
            is_conj=is_conj,
            optimize=self.optimize,
            backend=self.backend,
        )

    def conj(self):
        if self._conj_linop is None:
            self._conj_linop = self.copy(conj=True)
        return self._conj_linop

    def _transpose(self):
        if self._transpose_linop is None:
            self._transpose_linop = self.copy(transpose=True)
        return self._transpose_linop

    def _adjoint(self):
        """Hermitian conjugate of this TNLO."""
        # cache the adjoint
        if self._adjoint_linop is None:
            self._adjoint_linop = self.copy(conj=True, transpose=True)
        return self._adjoint_linop

    def to_dense(self, *inds_seq, to_qarray=False, **contract_opts):
        """Convert this TNLinearOperator into a dense array, defaulting to
        grouping the left and right indices respectively.
        """
        contract_opts.setdefault("optimize", self.optimize)

        if self.is_conj:
            ts = (t.conj() for t in self._tensors)
        else:
            ts = self._tensors

        if not inds_seq:
            inds_seq = self.left_inds, self.right_inds

        return tensor_contract(*ts, **contract_opts).to_dense(
            *inds_seq,
            to_qarray=to_qarray,
        )

    toarray = to_dense
    to_qarray = functools.partialmethod(to_dense, to_qarray=True)

    @functools.wraps(tensor_split)
    def split(self, **split_opts):
        return tensor_split(
            self,
            left_inds=self.left_inds,
            right_inds=self.right_inds,
            **split_opts,
        )

    @property
    def A(self):
        return self.to_dense()

    def astype(self, dtype):
        """Convert this ``TNLinearOperator`` to type ``dtype``."""
        return TNLinearOperator(
            (t.astype(dtype) for t in self._tensors),
            left_inds=self.left_inds,
            right_inds=self.right_inds,
            ldims=self.ldims,
            rdims=self.rdims,
            optimize=self.optimize,
            backend=self.backend,
        )

    def __array_function__(self, func, types, args, kwargs):
        if (func not in TNLO_HANDLED_FUNCTIONS) or (
            not all(issubclass(t, self.__class__) for t in types)
        ):
            return NotImplemented
        return TNLO_HANDLED_FUNCTIONS[func](*args, **kwargs)
