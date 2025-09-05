from typing import List, Tuple

import cotengra as ctg

from ttnte.cpp.linalg import ContractionStep
from ttnte.linalg.utils._gen_expr import gen_expr


def torchtt2cpp(shape: List[Tuple[int]]):
    """
    Extract contraction information from ``torchtt.TT``.
    """
    # Generate einsum expression
    expr = gen_expr(len(shape))

    # Add shape for input tensor
    shape += [(shape[0][1], *(s[2] for s in shape[1:]))]

    # Generate tree
    tree = ctg.einsum_tree(
        expr, *shape, canonicalize=True, sort_contraction_indices=True
    )

    # Get contraction information
    contractions = tree.get_contractor().contractions

    # Iterate through each step
    steps = []
    lidxs = []
    ridxs = []
    removed = []
    expected_ndim = [len(s) for s in shape]
    for contraction in contractions:
        assert contraction[3] == True
        assert len(contraction[4][0]) == len(contraction[4][1])

        lidxs.append(
            tuple(contraction[1])[0]
            if len(contraction[1]) == 1
            else min(*tuple(contraction[1]))
        )
        ridxs.append(
            tuple(contraction[2])[0]
            if len(contraction[2]) == 1
            else min(*tuple(contraction[2]))
        )

        # Make any readjustments
        for remove in removed:
            if ridxs[-1] > remove:
                ridxs[-1] -= 1

            if lidxs[-1] > remove:
                lidxs[-1] -= 1

        # Prep for removed indices
        removed.append(max(lidxs[-1], ridxs[-1]))

        # Create contraction step
        steps.append(
            ContractionStep(
                expected_ndim[lidxs[-1]],
                expected_ndim[ridxs[-1]],
                contraction[4][0],
                contraction[4][1],
                contraction[5],
            )
        )

        # Change the number of dimensions
        lndim = expected_ndim[lidxs[-1]]
        rndim = expected_ndim[ridxs[-1]]
        expected_ndim.pop(max(lidxs[-1], ridxs[-1]))

        expected_ndim[min(lidxs[-1], ridxs[-1])] = (
            lndim + rndim - 2 * len(contraction[4][0])
        )

    return steps, lidxs, ridxs
