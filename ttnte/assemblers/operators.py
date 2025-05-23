import cotengra as ctg
import torch as tn


class SparseOperator(object):
    def __init__(self, A):
        self._A = A

    def __matmul__(self, x):
        return (self._A @ x.clone().flatten()).reshape(x.shape)

    def __neg__(self):
        return SparseOperator(-self._A)

    def __add__(self, B):
        return SparseOperator(self._A + B._A)

    def __sub__(self, B):
        return SparseOperator(self._A - B._A)

    def cuda(self, device):
        self._A = self._A.cuda(device)
        return self

    def cpu(self):
        self._A = self._A.cpu()
        return self

    def to_dense(self):
        return self._A.to_dense()

    @property
    def shape(self):
        return self._A.shape


class ScatteringOperator(SparseOperator):
    def __init__(self, Y, S, w_mu, w_eta):
        self._w_mu = tn.tensor(w_mu)
        self._w_eta = tn.tensor(w_eta)
        self._Y = Y
        self._S = S

    def __matmul__(self, x):
        assert x.ndim >= 4 and x.shape[:3] == (
            4,
            self._w_mu.shape[0],
            self._w_eta.shape[0],
        )

        # Apply spherical harmonics and angular integration
        result = ctg.einsum(
            "abcd,labc,b,c->ld",
            x.reshape((*x.shape[:3], -1)),
            self._Y,
            self._w_mu,
            self._w_eta,
        )

        # Apply XS
        result[0,] = self._S[0,] @ result[0,]

        idx = 1
        for n in range(1, self._S.shape[0]):
            for m in range(n + 1):
                result[idx] = self._S[n,] @ result[idx,]
                idx += 1

        # Outer product with spherical harmonics
        return ctg.einsum("ld,labc->abcd", result, self._Y).reshape(x.shape)

    def __neg__(self):
        return ScatteringOperator(
            self._Y, -self._S, self._w_mu.numpy(), self._w_eta.numpy()
        )

    def cuda(self, device):
        self._w_mu = self._w_mu.cuda(device)
        self._w_eta = self._w_eta.cuda(device)
        self._Y = self._Y.cuda(device)
        self._S = self._S.cuda(device)
        return self

    def cpu(self):
        self._w_mu = self._w_mu.cpu()
        self._w_eta = self._w_eta.cpu()
        self._Y = self._Y.cpu()
        self._S = self._S.cpu()
        return self


class FissionOperator(SparseOperator):
    def __init__(self, F, w_mu, w_eta):
        self._w_mu = tn.tensor(w_mu)
        self._w_eta = tn.tensor(w_eta)
        self._F = F

    def __matmul__(self, x):
        assert x.ndim >= 4 and x.shape[:3] == (
            4,
            self._w_mu.shape[0],
            self._w_eta.shape[0],
        )

        # Angular integration
        result = ctg.einsum(
            "abcd,b,c->d",
            x.reshape((*x.shape[:3], -1)),
            self._w_mu,
            self._w_eta,
        )

        # Apply fission operator
        return ctg.einsum(
            "abc,d->abcd", tn.ones(x.shape[:3], device=x.device), self._F @ result
        ).reshape(x.shape)

    def cuda(self, device):
        self._w_mu = self._w_mu.cuda(device)
        self._w_eta = self._w_eta.cuda(device)
        self._F = self._F.cuda(device)
        return self

    def cpu(self):
        self._w_mu = self._w_mu.cpu()
        self._w_eta = self._w_eta.cpu()
        self._F = self._F.cpu()
        return self


class AngularOperator(SparseOperator):
    def __init__(self, w_mu, w_eta):
        """"""
        self._w_mu = tn.tensor(w_mu)
        self._w_eta = tn.tensor(w_eta)

    def __matmul__(self, x):
        """"""
        assert x.ndim >= 4 and x.shape[:3] == (
            4,
            self._w_mu.shape[0],
            self._w_eta.shape[0],
        )

        # Reshape and apply angular operator
        return ctg.einsum(
            "abcd,b,c->d", x.reshape((*x.shape[:3], -1)), self._w_mu, self._w_eta
        )
