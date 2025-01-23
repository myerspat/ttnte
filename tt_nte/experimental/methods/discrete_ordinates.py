"""
discrete_ordinates.py.

For setting up and building operators H, S, and F.
"""

import math
from typing import Optional

import numpy as np
from scipy.special import lpmv
from quimb.tensor import MatrixProductOperator, tensor_network_apply_op_op

from tt_nte.xs import Server
from tt_nte.geometry import Geometry
from tt_nte import TensorTrain


class DiscreteOrdinates(object):
    _direction_spaces = [
        np.array([[1], [-1]]),
        np.array(np.meshgrid([1, -1], [1, -1])).T.reshape(-1, 2),
        np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3),
    ]

    def __init__(
        self,
        xs_server: Server,
        geometry: Geometry,
        num_ordinates: int,
        octant_ords: Optional[np.ndarray] = None,
        xs_threshold: float = 1e-8,
    ):
        """"""
        self.update_settings(
            xs_server=xs_server,
            geometry=geometry,
            num_ordinates=num_ordinates,
            octant_ords=octant_ords,
            xs_threshold=xs_threshold,
        )

    def update_settings(
        self,
        xs_server: Optional[Server] = None,
        geometry: Optional[Geometry] = None,
        num_ordinates: Optional[int] = None,
        octant_ords: Optional[np.ndarray] = None,
        xs_threshold: Optional[float] = None,
    ):
        """"""
        self._xs_server = self._xs_server if xs_server is None else xs_server
        self._xs_threshold = (
            self._xs_threshold if xs_threshold is None else xs_threshold
        )
        self._geometry = self._geometry if geometry is None else geometry
        self._num_ordinates = (
            self._num_ordinates if num_ordinates is None else num_ordinates
        )

        # Get quadrature set
        # 1D = (N, 2): w, mu
        # 2D = (N, 3): w, mu, eta
        # 3D = (N, 4): w, mu, eta, xi
        self._octant_ords = (
            octant_ords
            if octant_ords
            else self._compute_square_set(self._num_ordinates, self._geometry.num_dim)
        )
        if octant_ords is not None:
            self._num_ordinates = int(
                2**self._geometry.num_dim * self._octant_ords.shape[0]
            )

        # Ensure weights are normalized correctly
        self._octant_ords[:, 0] = (
            1
            / self._direction_spaces[self._geometry.num_dim - 1].shape[0]
            * self._octant_ords[:, 0]
            / np.sum(self._octant_ords[:, 0])
        )

        # Construct operators in TT format
        self._construct_tts()

    # =====================================================================
    # TT Construction

    def _construct_tts(self):
        """"""
        self._octants = self._direction_spaces[self._geometry.num_dim - 1]
        self._num_octants = self._octants.shape[0]
        self._num_dim = self._geometry.num_dim
        self._num_groups = self._xs_server.num_groups

        # Determine if we have a fixed source problem
        self._is_fixed_source = True if np.sum(self._xs_server.chi) == 0 else False

        # Differential and interpolation operators
        self._D = []
        self._Ip = []

        for i in range(self._num_dim):
            # Get differential length along dimension
            diff = self._geometry.diff[i].reshape((-1, 1))

            # Number of nodes along dimension
            dim_num_nodes = diff.size + 1

            d = np.eye(dim_num_nodes, k=0)
            d_p = np.eye(dim_num_nodes, k=-1)
            d_m = np.eye(dim_num_nodes, k=1)

            # Add to differential operator
            self._D.append(
                [
                    (d - d_p)
                    / np.concatenate(
                        [
                            diff[[0],],
                            diff,
                        ]
                    ),
                    (d_m - d) / np.concatenate([diff, diff[[-1],]]),
                ]
            )

            # Add to interpolation operator
            self._Ip.append([(d + d_p) / 2, (d + d_m) / 2])

        # Group identity matrix
        self._Ig = np.identity(self._num_groups)

        # Ordinate identity matrix
        self._IL = np.identity(int(self._num_ordinates / self._num_octants))

        # Create XS operators
        self._total, self._scatter_gtg, self._nu_fission = self.xs_ops()

        # Ensure correct permutation for XS operators
        self._total.permute_arrays("lrud")
        self._nu_fission.permute_arrays("lrud")
        for scatter_l in self._scatter_gtg:
            scatter_l.permute_arrays("lrud")

        print(self._total)
        print(self._scatter_gtg[0])
        print(self._nu_fission)

        # Iterate over half-spaces/quadrants/octants and append
        # TT cores
        self._H = self._num_octants * [None]
        self._F = self._num_octants * [None]
        self._S = self._num_octants * [None]

        # Construct TTs at each BC in parallel
        np.vectorize(self._construct_tts_for_octant)(np.arange(self._num_octants))

        # Check all BCs are filled
        assert np.all([isinstance(H, MatrixProductOperator) for H in self._H])
        assert np.all([isinstance(S, MatrixProductOperator) for S in self._S])
        assert np.all([isinstance(F, MatrixProductOperator) for F in self._F])

        # Sum across all boundaries
        H = self._H[0]
        S = self._S[0]
        F = self._F[0]

        for i in range(1, self._num_octants):
            H += self._H[i]
            S += self._S[i]
            F += self._F[i]

        # Save total operators
        self._H = H
        self._S = S
        self._F = F

        # Delete identity, differential, and interpolation matrices
        del self._D, self._Ip, self._Ig, self._IL

        # Delete XS operators
        del self._total, self._scatter_gtg, self._nu_fission

    def _construct_tts_for_octant(self, i):
        """"""
        # Get current octant
        octant = self._octants[i, :]

        # Get index into D, Ip, and BC based on direction (1 or -1)
        dir_idx = [0 if dir > 0 else 1 for dir in octant]

        C = np.zeros((self._num_octants, self._num_octants), dtype=float)
        C[i, i] = 1.0

        # Get Hx, Hy, Hz
        for j in range(self._num_dim):
            # Get boundary condition
            bc = self._geometry.bcs[int(j + 3 * dir_idx[j])]

            # Angular point matrix
            Q = np.kron(C, octant[j] * np.diag(self._octant_ords[:, j + 1]))

            # Get spatial cores in order of z, y, x
            spatial_cores = []
            for k in reversed(range(self._num_dim)):
                if j != k:
                    spatial_cores.append(self._Ip[k][dir_idx[k]])
                else:
                    spatial_cores.append(self._D[k][dir_idx[k]])

            # Create or append to TT
            H_i = self._create_tt([Q, self._Ig] + spatial_cores)
            self._H[i] = self._append_tt(self._H[i], H_i)

            # Add reflective BC
            if bc == "reflective":
                # Find reflected octant
                ref_octant = np.copy(octant)
                ref_octant[j] *= -1

                # Place values in reflected octant positions
                C_ref = np.zeros((self._num_octants, self._num_octants), dtype=float)
                C_ref[i, np.where((self._octants == ref_octant).all(axis=1))[0]] = 1

                Q_ref = np.kron(
                    C_ref, ref_octant[j] * np.diag(self._octant_ords[:, j + 1])
                )

                # Apply boundary mask ot spatial cores
                sp_idx = self._num_dim - 1 - j
                bc_mask = np.zeros((spatial_cores[sp_idx].shape[0], 1))
                bc_mask[-dir_idx[j], 0] = 1
                spatial_cores[sp_idx] = bc_mask * np.copy(spatial_cores[sp_idx])

                # Append to TT
                self._H[i] += self._create_tt([Q_ref, self._Ig] + spatial_cores)

        # Get spatial cores for total interaction
        spatial_cores = []
        for j in range(self._num_dim - 1, -1, -1):
            spatial_cores.append(self._Ip[j][dir_idx[j]])

        # Apply total XS operator
        H_sig_i = self._apply_xs_op(
            self._create_tt([self._Ig] + spatial_cores), self._total.copy(), dir_idx
        )
        H_sig_i.permute_arrays("ludr")
        self._H[i] += MatrixProductOperator(
            [np.kron(C, self._IL)[..., np.newaxis], H_sig_i[0].data[np.newaxis, ...]]
            + [H_sig_i[i].data for i in range(1, H_sig_i.L)],
            shape="ludr",
        )

        # Fission integral operator
        A = np.zeros((self._num_octants, self._num_octants), dtype=float)
        A[i, :] = 1
        F_Intg = np.kron(
            A,
            np.outer(np.ones(self._octant_ords.shape[0]), self._octant_ords[:, 0]),
        )

        # Apply fission XS operator
        F_sig_i = self._apply_xs_op(
            self._create_tt([self._Ig] + spatial_cores),
            self._nu_fission.copy(),
            dir_idx,
        )
        F_sig_i.permute_arrays("ludr")
        self._F[i] = MatrixProductOperator(
            [F_Intg[..., np.newaxis], F_sig_i[0].data[np.newaxis, ...]]
            + [F_sig_i[i].data for i in range(1, F_sig_i.L)],
            shape="ludr",
        )

        # Number of ordinates in an octant
        n = int(self._num_ordinates / self._num_octants)

        # Iterate through scattering moments
        for l in range(self._xs_server.num_moments):
            # Scattering integral operator
            S_Intg = np.zeros(np.kron(C, self._IL).shape)

            # Outgoing ordinates
            out_ords = np.copy(self._octant_ords)
            out_ords[:, 1:] *= octant[np.newaxis, :]

            for k in range(self._octants.shape[0]):
                # Incoming ordinates
                in_ords = np.copy(self._octant_ords)
                in_ords[:, 1:] *= self._octants[[k], :]

                S_Intg[
                    int(i * n) : int(i * n + n),
                    int(k * n) : int(k * n + n),
                ] = np.outer(
                    self._Y(l, 0, out_ords),
                    self._octant_ords[:, 0] * self._Y(l, 0, in_ords),
                )

            if self._num_dim > 1:
                for m in range(1, l + 1):
                    for k in range(self._octants.shape[0]):
                        # Incoming ordinates
                        in_ords = np.copy(self._octant_ords)
                        in_ords[:, 1:] *= self._octants[[k], :]

                        S_Intg[
                            int(i * n) : int(i * n + n),
                            int(k * n) : int(k * n + n),
                        ] += 2 * np.outer(
                            self._Y(l, m, out_ords, even=True),
                            self._octant_ords[:, 0] * self._Y(l, m, in_ords, even=True),
                        ) + (
                            2
                            * np.outer(
                                self._Y(l, m, out_ords, even=False),
                                self._octant_ords[:, 0]
                                * self._Y(l, m, in_ords, even=False),
                            )
                            if self._num_dim > 2
                            else 0
                        )

            # Append tensor train to scattering
            S_sig_i = self._apply_xs_op(
                self._create_tt([self._Ig] + spatial_cores),
                self._scatter_gtg[l].copy(),
                dir_idx,
            )
            S_sig_i.permute_arrays("ludr")
            self._S[i] = self._append_tt(
                self._S[i],
                MatrixProductOperator(
                    [S_Intg[..., np.newaxis], S_sig_i[0].data[np.newaxis, ...]]
                    + [S_sig_i[i].data for i in range(1, S_sig_i.L)],
                    shape="ludr",
                ),
            )

    def xs_ops(self):
        """"""
        num_dim = self._geometry.num_dim
        num_moments = self._xs_server.num_moments

        # Get total XS tensor
        total = np.transpose(
            self._xs_server.total(), axes=[0] + np.arange(1, num_dim + 1).tolist()[::-1]
        )
        for _ in range(num_dim + 1):
            total = total[..., np.newaxis]

        # Create TT-vector (MPS)
        total = TensorTrain(total, threshold=0)

        # Convert to TT-operator (MPO)
        total = total.diag(np.arange(total.order).tolist()).to_quimb()
        total.compress(cutoff=self._xs_threshold)

        # Get nu_fission tensor
        nu_fission = np.zeros(
            (
                self._num_groups,
                self._num_groups,
                *self._xs_server.nu_fission().shape[1:],
            )
        )
        for g_out in range(self._num_groups):
            for g_in in range(self._num_groups):
                nu_fission[g_out, g_in, ...] = (
                    self._xs_server.chi[g_out] * self._xs_server.nu_fission()[g_in, ...]
                )

        nu_fission = np.transpose(
            nu_fission,
            axes=[0] + np.arange(2, num_dim + 2).tolist()[::-1] + [1],
        )
        for _ in range(num_dim):
            nu_fission = nu_fission[..., np.newaxis]

        # Create TT-vector (MPS)
        nu_fission = TensorTrain(nu_fission, threshold=0)

        # Convert to TT-operator (MPO)
        nu_fission = nu_fission.diag(np.arange(1, nu_fission.order).tolist()).to_quimb()
        nu_fission.compress(cutoff=self._xs_threshold)

        # Iterate through each scattering moment
        scatter_gtg = []
        for l in range(num_moments):
            # Get g-to-g scattering moment tensor
            scatter_l = np.transpose(
                self._xs_server.scatter_gtg()[l,],
                axes=[0] + np.arange(2, num_dim + 2).tolist()[::-1] + [1],
            )
            for _ in range(num_dim):
                scatter_l = scatter_l[..., np.newaxis]

            # Create TT-vector (MPS)
            scatter_l = TensorTrain(scatter_l, threshold=0)

            # Convert to TT-operator (MPO)
            scatter_gtg.append(
                scatter_l.diag(np.arange(1, scatter_l.order).tolist()).to_quimb()
            )
            scatter_gtg[-1].compress(cutoff=self._xs_threshold)

        return total, scatter_gtg, nu_fission

    # =====================================================================
    # Quadrature sets

    @staticmethod
    def _compute_square_set(N, num_dim):
        if num_dim == 1:
            return DiscreteOrdinates._gauss_legendre(N)
        elif num_dim == 2:
            octant_ords = DiscreteOrdinates._chebyshev_legendre(N * 2)[:, :-1]
            octant_ords[:, 0] *= 2
            return octant_ords
        else:
            return DiscreteOrdinates._chebyshev_legendre(N)

    @staticmethod
    def _gauss_legendre(N):
        """
        Gauss Legendre quadrature set.

        Only given for positive half space.
        """
        mu, w = np.polynomial.legendre.leggauss(N)
        w = w[: int(mu.size / 2)] / 2
        mu = np.abs(mu[: int(mu.size / 2)])

        assert np.round(np.sum(w), 1) == 0.5

        return np.concatenate([w[:, np.newaxis], mu[:, np.newaxis]], axis=1)

    @staticmethod
    def _gauss_chebyshev(N):
        """Gauss-Chebyshev quadrature."""
        gamma = (2 * np.arange(1, int(N / 2) + 1) - 1) * np.pi / (2 * N)
        w = np.ones(int(N / 2)) / (2 * N)

        return np.concatenate([w[:, np.newaxis], gamma[:, np.newaxis]], axis=1)

    @staticmethod
    def _chebyshev_legendre(N):
        """
        Chebyshev-Legendre (square) qudrature set given by
        https://www.osti.gov/servlets/purl/5958402.
        """
        assert N % 8 == 0

        n = np.round(np.sqrt(N / 2)).astype(int)

        # Compute quadrature
        q_l = DiscreteOrdinates._gauss_legendre(n)
        q_c = DiscreteOrdinates._gauss_chebyshev(n)

        w_l, mu = q_l[:, 0], q_l[:, 1]
        w_c, gamma = q_c[:, 0], q_c[:, 1]

        # Assert number of ordinates
        assert 8 * gamma.size * mu.size == N

        ordinates = np.zeros((int(N / 8), 4))
        for i in range(mu.size):
            for j in range(gamma.size):
                k = i * gamma.size + j
                ordinates[k, 0] = w_l[i] * w_c[j]
                ordinates[k, 1] = mu[i]
                ordinates[k, 2] = np.sqrt(1 - mu[i] ** 2) * np.cos(gamma[j])
                ordinates[k, 3] = np.sqrt(1 - mu[i] ** 2 - ordinates[k, 2] ** 2)

        return ordinates

    # =====================================================================
    # Static methods

    @staticmethod
    def _append_tt(x, y):
        return x + y if x else y

    @staticmethod
    def _create_tt(cores):
        return MatrixProductOperator(
            [cores[0][..., np.newaxis]]
            + [core[np.newaxis, ..., np.newaxis] for core in cores[1:-1]]
            + [cores[-1][np.newaxis, ...]],
            shape="ludr",
        )

    @staticmethod
    def _apply_xs_op(x, xs_op, dir_idx):
        # Iterate through cores adding an additional row/column of zeros
        # at BC location
        for i in range(1, xs_op.L):
            shape = list(xs_op[i].shape)
            shape[-1] += 1
            shape[-2] += 1

            # Create larger core and place original data
            core = np.zeros(shape)
            if dir_idx[::-1][i - 1] > 0:
                core[..., 1:, 1:] = xs_op[i].data
            else:
                core[..., :-1, :-1] = xs_op[i].data

            # Put core back
            xs_op[i].modify(data=core)

        # Apply operator
        return tensor_network_apply_op_op(xs_op, x, contract=True)

    @staticmethod
    def _Y(l, m, ordinates, even=True):
        y = (
            (-1) ** m
            * np.sqrt(
                (2 * l + 1) * math.factorial(l - abs(m)) / math.factorial(l + abs(m))
            )
            * lpmv(m, l, ordinates[:, 1])
        )

        if m == 0:
            return y
        elif even:
            gamma = np.arccos(ordinates[:, 2] / np.sqrt(1 - ordinates[:, 1] ** 2))
            return y * np.cos(m * gamma)
        else:
            gamma = np.arcsin(ordinates[:, 3] / np.sqrt(1 - ordinates[:, 1] ** 2))
            return y * np.sin(m * gamma)

    # =====================================================================
    # Getters

    @property
    def H(self):
        assert isinstance(self._H, MatrixProductOperator)
        return self._H

    @property
    def S(self):
        assert isinstance(self._S, MatrixProductOperator)
        return self._S

    @property
    def F(self):
        assert isinstance(self._F, MatrixProductOperator)
        return self._F

    @property
    def geometry(self):
        return self._geometry

    @property
    def xs_server(self):
        return self._xs_server

    @property
    def num_ordinates(self):
        return self._num_ordinates
