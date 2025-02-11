import matplotlib.pyplot as plt
import numpy as np
from geomdl import NURBS
from geomdl.helpers import basis_functions, basis_functions_ders, find_spans


class Patch1D(object):
    def __init__(self, mat: str, xl: float, xr: float, source: float = 0.0):
        """Define 1-D NURBS curve."""
        self.mat = mat
        self.L = xr - xl
        self.source = source
        self.curve = NURBS.Curve()
        self.curve.degree = 1
        self.curve.ctrlpts = np.array([(xl, 0, 0), (xr, 0, 0)])
        self.curve.knotvector = np.array(2 * [xl / self.L] + 2 * [xr / self.L])

    def basis_functions(self, knots):
        # Find knot spans from knotvector
        return np.array(
            basis_functions(
                self.curve.degree, self.curve.knotvector, self.find_spans(knots), knots
            )
        )

    def basis_functions_ders(self, knots):
        # Find knot spans from knotvector
        return np.array(
            basis_functions_ders(
                self.curve.degree,
                self.curve.knotvector,
                self.find_spans(knots),
                knots,
                1,
            )
        )

    def find_spans(self, knots):
        return find_spans(
            self.curve.degree, self.curve.knotvector, self.curve.ctrlpts_size, knots
        )

    def jacobian(self, knots):
        spans = self.find_spans(knots)

        def eval_jacobain(knot, span):
            # Find knot span bounds
            a = self.curve.knotvector[span]
            b = self.curve.knotvector[span + 1]

            # Evaluate derivative of the curve
            der = np.sum(
                self.basis_functions_ders([knot])[0, 1, :]
                * np.array(
                    [
                        x[0]
                        for x in self.curve.ctrlpts[span - self.curve.degree : span + 1]
                    ]
                )
            )

            return (b - a) / 2 * der

        return np.array(list(map(eval_jacobain, knots, spans)))

    def plot_basis(self, num_points=1000):
        # Discretize curve
        knots = np.linspace(
            min(self.curve.knotvector), max(self.curve.knotvector), num_points
        )

        # Caluclate basis functions
        plt.clf()
        plt.plot(knots, self.basis_functions(knots))
        plt.legend(
            [
                "$N_{" + f"{a, self.curve.degree}" + "}$"
                for a in range(self.curve.ctrlpts_size)
            ]
        )
        plt.xlabel("$\\xi$")

    @property
    def num_elements(self):
        return np.unique(self.curve.knotvector).size - 1

    @property
    def element_spans(self):
        return np.unique(self.curve.knotvector)

    @property
    def degree(self):
        return self.curve.degree

    @property
    def ctrlpts(self):
        return self.curve.ctrlpts

    @property
    def ctrlpts_size(self):
        return self.curve.ctrlpts_size

    @property
    def knotvector(self):
        return self.curve.knotvector

    @property
    def num_dofs(self):
        return self.ctrlpts_size
