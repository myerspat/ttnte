import numpy as np
import torchtt as tntt
from tt_nte.tensor_train import TensorTrain
from tt_nte.xs import Server

from geometry import Geometry


class TT(tntt.TT):
    def matvec(self, x):
        x = x.reshape(self.__N)
        return self.__matmul__(x).reshape((-1, 1))


def build_operators(
    domain: Geometry,
    xs_server: Server,
    num_ordinates: int,
    num_points: int,
    fixed_source: bool = False,
):
    """"""
    # Sn quadrature
    mu, w_mu = np.polynomial.legendre.leggauss(num_ordinates)
    w_mu = w_mu[: int(mu.size / 2)] / 2
    mu = abs(mu[: int(mu.size / 2)])

    # Spatial quadrature
    x, w_x = np.polynomial.legendre.leggauss(num_points)

    # Angular integration operator
    Int_N = TensorTrain(
        [
            np.block([2 * [w_mu]]),
            np.identity(xs_server.num_groups),
            np.identity(domain.num_patches),
            np.identity(domain.patches[0].num_dofs),
        ],
        threshold=0,
    ).to_quimb()

    # Calculate operators
    H, S, F = [
        op_p + op_m
        for op_p, op_m in zip(
            _operators_for_boundary(
                domain, xs_server, (mu, w_mu), (x, w_x), 0, fixed_source=fixed_source
            ),
            _operators_for_boundary(
                domain, xs_server, (-mu, w_mu), (x, w_x), 0, fixed_source=fixed_source
            ),
        )
    ]

    for i in range(1, domain.num_patches):
        H_i, S_i, F_i = [
            op_p + op_m
            for op_p, op_m in zip(
                _operators_for_boundary(
                    domain,
                    xs_server,
                    (mu, w_mu),
                    (x, w_x),
                    i,
                    fixed_source=fixed_source,
                ),
                _operators_for_boundary(
                    domain,
                    xs_server,
                    (-mu, w_mu),
                    (x, w_x),
                    i,
                    fixed_source=fixed_source,
                ),
            )
        ]
        H += H_i
        S += S_i
        F += F_i

    return H, S, F, Int_N


def _operators_for_boundary(
    domain,
    xs_server,
    dir_quad,
    space_quad,
    patch_idx,
    fixed_source=False,
):
    # Get patch
    patch = domain.patches[patch_idx]

    # BCs
    bcs = [domain.left_bc, domain.right_bc][::-1]

    # Direction and spatial quadrature
    mu, w_mu = dir_quad
    x, w_x = space_quad

    # Direction index
    dir_idx = 0 if (mu > 0).all() else 1

    # Define identity operators
    Ig = np.identity(xs_server.num_groups)[np.newaxis, ..., np.newaxis]
    IL = np.identity(len(mu))

    # Angular point matrix
    C = np.zeros((2, 2), dtype=float)
    C[dir_idx, dir_idx] = 1.0
    Qu = np.kron(C, np.diag(-mu))[np.newaxis, ..., np.newaxis]

    # Integral operator
    A = np.zeros((2, 2), dtype=float)
    A[dir_idx, :] = 1
    Int_mu = np.kron(A, np.outer(np.ones(w_mu.size), w_mu))[np.newaxis, ..., np.newaxis]

    # Spatial discretization
    D = np.zeros((patch.num_dofs, patch.num_dofs))
    Int_x = np.zeros((patch.num_dofs, patch.num_dofs))
    Out = np.zeros((patch.num_dofs, patch.num_dofs))
    int_x = np.zeros((patch.num_dofs, 1))

    element = 0
    for i in range(patch.num_elements):
        # Bounds of integration
        a = patch.element_spans[i]
        b = patch.element_spans[i + 1]

        # Calculate basis functions and their derivatives
        basis_data = patch.basis_functions_ders((b - a) / 2 * x + (a + b) / 2)

        # Outgoing boundary basis functions
        basis_data_out = patch.basis_functions([b if dir_idx == 0 else a])[0, :]

        # Integral and XS Jacobian determinant
        jacobian = patch.jacobian((b - a) / 2 * x + (a + b) / 2)

        # Outgoing boundary Jacobian determinant
        jacobian_out = patch.jacobian([1 if dir_idx == 0 else 0])[0]

        # Fill local element
        for j in range(patch.curve.degree + 1):
            # Skip first and last rows of matrix for boundary
            # Include 1 for implicit boundary condition
            if (dir_idx == 0 and element == 0 and j == 0) or (
                dir_idx == 1
                and element == patch.num_elements - 1
                and j == patch.curve.degree
            ):
                continue

            # Spatial integral for fixed source
            int_x[element + j, 0] += np.sum(jacobian**2 * w_x * basis_data[:, 0, j])

            for k in range(patch.curve.degree + 1):
                # Streaming operator
                D[element + j, element + k] += (
                    (b - a)
                    / 2
                    * np.sum(jacobian * w_x * basis_data[:, 1, j] * basis_data[:, 0, k])
                )

                # Spatial integral
                Int_x[element + j, element + k] += np.sum(
                    jacobian**2 * w_x * basis_data[:, 0, j] * basis_data[:, 0, k]
                )

                # Outgoing
                if (dir_idx == 0 and element == patch.num_elements - 1) or (
                    dir_idx == 1 and element == 0
                ):
                    Out[element + j, element + k] += (
                        jacobian_out * basis_data_out[j] * basis_data_out[k]
                    )

        # Increment global element
        element += 1

    # Correct shapes for spatial cores
    D = D[np.newaxis, ..., np.newaxis]
    Int_x = Int_x[np.newaxis, ..., np.newaxis]
    Out = Out[np.newaxis, ..., np.newaxis]

    # Patch
    P = np.zeros(2 * [domain.num_patches])[np.newaxis, ..., np.newaxis]
    P[0, patch_idx, patch_idx, 0] = 1.0

    # Boundary
    B = np.zeros(2 * [patch.num_dofs])[np.newaxis, ..., np.newaxis]
    B[0, -dir_idx, -dir_idx, 0] = 1.0

    # Append direction core and add other operators
    H = (
        TensorTrain(
            [
                np.kron(C, IL)[np.newaxis, ..., np.newaxis],
                np.diag(xs_server.total(patch.mat))[np.newaxis, ..., np.newaxis],
                P,
                Int_x,
            ],
            threshold=0,
        )
        + TensorTrain([Qu, Ig, P, D], threshold=0)
        + TensorTrain([np.abs(Qu), Ig, P, Out], threshold=0)
        + TensorTrain(
            [np.kron(C, np.identity(len(mu)))[np.newaxis, ..., np.newaxis], Ig, P, B]
        )
    )
    S = TensorTrain(
        [Int_mu, xs_server.scatter_gtg(patch.mat)[..., np.newaxis], P, Int_x],
        threshold=0,
    )
    if not fixed_source:
        Q = TensorTrain(
            [
                Int_mu,
                np.outer(xs_server.chi, xs_server.nu_fission(patch.mat)),
                P,
                Int_x,
            ],
            threshold=0,
        )

    else:
        Il = np.zeros(2 * len(mu))
        if dir_idx == 0:
            Il[: len(mu)] = 1
        else:
            Il[len(mu) :] = 1

        Q = TensorTrain(
            [
                Il.reshape((1, -1, 1, 1)),
                patch.source * np.ones(xs_server.num_groups).reshape((1, -1, 1, 1)),
                np.sum(P, axis=2).reshape((1, -1, 1, 1)),
                int_x.reshape((1, -1, 1, 1)),
            ],
            threshold=0,
        )

    # Handle interface and boundary conditions
    if patch_idx < domain.num_patches - 1 and dir_idx == 0:
        # Interface for flux from left to right
        P_in = np.zeros(2 * [domain.num_patches])[np.newaxis, ..., np.newaxis]
        P_in[0, patch_idx + 1, patch_idx, 0] = 1.0

        S += TensorTrain(
            [
                np.kron(C, np.identity(len(mu)))[np.newaxis, ..., np.newaxis],
                Ig,
                P_in,
                B[:, :, ::-1, :],
            ]
        )

    elif patch_idx > 0 and dir_idx == 1:
        # Interface for flux from right to left
        P_in = np.zeros(2 * [domain.num_patches])[np.newaxis, ..., np.newaxis]
        P_in[0, patch_idx - 1, patch_idx, 0] = 1.0

        S += TensorTrain(
            [
                np.kron(C, np.identity(len(mu)))[np.newaxis, ..., np.newaxis],
                Ig,
                P_in,
                B[:, :, ::-1, :],
            ]
        )

    elif bcs[dir_idx] == "reflective":
        # Get reflected octant position
        C_ref = np.zeros((2, 2), dtype=float)
        C_ref[np.abs(dir_idx - 1), dir_idx] = 1

        Q_ref = np.kron(C_ref, np.identity(len(mu)))[np.newaxis, ..., np.newaxis]

        # Enforce reflective boundary condition
        S += TensorTrain([Q_ref, Ig, P, B[:, ::-1, ::-1, :]])

    # Convert to quimb MPOs
    H = H.to_quimb()
    S = S.to_quimb()
    Q = Q.to_quimb()

    return H, S, Q


def create_XS_operator(domain, xs_method, num_groups, dir_idx):
    # Initialize zero tensor
    xs = np.zeros((num_groups, num_groups, domain.num_dofs, domain.num_dofs))

    # Boundary offset
    offset = 1 if dir_idx == 0 else 0

    element = 0
    for patch in domain.patches:
        # Get XS for patch
        mat_xs = xs_method(patch.mat)

        # Iterate through degrees of freedom in patch
        for _ in range(patch.num_elements):
            for i in range(patch.curve.degree):
                xs[:, :, element + i + offset, element + i + offset] = mat_xs

            element += 1

    return TensorTrain(np.transpose(xs, axes=(0, 2, 1, 3)))
