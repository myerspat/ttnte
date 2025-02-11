from geomdl.operations import degree_operations, refine_knotvector

from patch_1d import Patch1D


class Geometry(object):
    def __init__(self, patches, left_bc="vacuum", right_bc="vacuum"):
        # Save BCs
        self.left_bc = left_bc
        self.right_bc = right_bc

        # Create geometry with NURBS curves
        self.patches = []
        self.L = patches[-1]["xr"] - patches[0]["xl"]
        for p in patches:
            self.patches.append(
                Patch1D(
                    mat=p["mat"],
                    xl=p["xl"],
                    xr=p["xr"],
                    source=p["source"] if "source" in p else 0.0,
                )
            )

        self.degree = 1

    def degree_elevate(self, num=1):
        for _ in range(num):
            for j in range(len(self.patches)):
                degree_operations(self.patches[j].curve, [1])

            self.degree += 1

    def knot_insert(self, num=1):
        for i in range(len(self.patches)):
            refine_knotvector(self.patches[i].curve, [num])

    @property
    def num_elements(self):
        return sum([p.num_elements for p in self.patches])

    @property
    def num_patches(self):
        return len(self.patches)

    @property
    def num_dofs(self):
        return sum(
            [self.patches[0].num_dofs] + [p.num_dofs - 1 for p in self.patches[1:]]
        )
