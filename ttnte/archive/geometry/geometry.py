import gmsh
import numpy as np


class Geometry(object):
    def __init__(self, model: gmsh.model, verbose=False):
        self._verbose = verbose

        if not self._verbose:
            gmsh.option.setNumber("General.Terminal", 0)

        self._read_msh(model)

    # ===================================================
    # Methods

    def _read_msh(self, model: gmsh.model):
        # Get element types
        element_types = model.mesh.get_element_types()

        # Get base element type (lines (1D), quadrangles (2D), hexahedra (3D))
        base_type = None

        # Determine dimensionality of the problem
        if np.any(element_types == 5):
            base_type = 5
            self._num_dim = 3
        elif np.any(element_types == 3):
            base_type = 3
            self._num_dim = 2
        elif np.any(element_types == 1):
            base_type = 1
            self._num_dim = 1
        else:
            raise RuntimeError(f"Gmsh element type ({base_type}) is not supported")

        # Load node tags and coordinates
        node_tags, coords, _ = model.mesh.get_nodes(returnParametricCoord=False)

        # Sort tags and coordinates
        node_tags, coords = self._sort_nodes(node_tags, coords)
        self._num_nodes = node_tags.size

        # Number of mesh elements
        element_tags, _ = model.mesh.get_elements_by_type(elementType=base_type)
        self._num_elements = element_tags.size

        # Get bounding box
        xmin, ymin, zmin, xmax, ymax, zmax = model.get_bounding_box(dim=-1, tag=-1)

        # Process differential distances
        self._diff = [None] * 3
        self._find_diff(xmin, xmax, coords, 0)
        self._find_diff(ymin, ymax, coords, 1)
        self._find_diff(zmin, zmax, coords, 2)

        # Get boundary information
        bc_mesh = {}
        for dim, tag in model.get_physical_groups(dim=self._num_dim - 1):
            bc_tags, bc_coords = model.mesh.get_nodes_for_physical_group(
                dim=dim, tag=tag
            )
            bc_coords = bc_coords.reshape((-1, 3))

            bc_mesh[gmsh.model.get_physical_name(dim=dim, tag=tag)] = bc_tags, bc_coords

        # Find face tags and boundary conditions
        self._bcs = [None] * 6
        self._bc_tags = [None] * 6
        self._find_face(xmin, xmax, bc_mesh, 0)
        self._find_face(ymin, ymax, bc_mesh, 1)
        self._find_face(zmin, zmax, bc_mesh, 2)

        # Get material region information
        self._mat_regions = {
            model.get_physical_name(
                dim=dim, tag=tag
            ): model.mesh.get_nodes_for_physical_group(dim=dim, tag=tag)[0]
            for (dim, tag) in model.get_physical_groups(dim=self._num_dim)
        }

        # Determine tags on the boundary
        for i in range(len(self._bc_tags)):
            if self._bc_tags is not None:
                bc_regions = {}
                for mat_name, mat_tags in self._mat_regions.items():
                    idxs = np.isin(self._bc_tags[i], mat_tags, assume_unique=True)

                    if np.any(idxs):
                        bc_regions[mat_name] = self._bc_tags[i][idxs]

                self._bc_tags[i] = bc_regions

        # Convert tags to indices in node_tags
        self._mat_regions = {
            mat_name: self._find_tag_positions(node_tags, mat_tags)
            for (mat_name, mat_tags) in self._mat_regions.items()
        }

        for i in range(len(self._bc_tags)):
            if self._bc_tags[i] is not None:
                self._bc_tags[i] = {
                    mat_name: self._find_tag_positions(node_tags, mat_tags)
                    for (mat_name, mat_tags) in self._bc_tags[i].items()
                }

    def _find_diff(self, xmin, xmax, coords, dim_idx):
        if xmin != xmax:
            x = np.unique(coords[:, dim_idx])
            self._diff[dim_idx] = np.round(x[1:] - x[:-1], 8).reshape((-1, 1))

    def _find_face(self, xmin, xmax, bc_mesh, dim_idx):
        if xmin != xmax:
            for pos, idx in zip([xmin, xmax], [dim_idx, dim_idx + 3]):
                max_nodes = 0

                for bc_name, (bc_tags, bc_coords) in bc_mesh.items():
                    if max_nodes < np.sum(bc_coords[:, dim_idx] == pos):
                        # Save bc name and face tags
                        self._bcs[idx] = bc_name
                        self._bc_tags[idx] = bc_tags[
                            np.argwhere(bc_coords[:, dim_idx] == pos)
                        ]

                        # Increment max nodes
                        max_nodes = np.sum(bc_coords[:, dim_idx] == pos)

    @staticmethod
    def _find_tag_positions(all_tags, sub_tags):
        sort_idx = all_tags.argsort()
        return np.sort(sort_idx[np.searchsorted(all_tags, sub_tags, sorter=sort_idx)])

    @staticmethod
    def _sort_nodes(node_tags, coords):
        # Round and reshape coordinates
        coords = np.round(coords.reshape((-1, 3)), 8)

        # Sorted indices
        idxs = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))

        return node_tags[idxs], coords[idxs, :]

    def region_mask(self, mat_name):
        """Create masks for each spatial dimension."""
        # Create 1d array of node positions
        region = np.zeros(self._num_nodes)
        region[self._mat_regions[mat_name]] = 1

        # Get dimension based masks
        region = np.squeeze(
            region.reshape(
                [diff.size + 1 if diff is not None else 1 for diff in self._diff]
            )
        )

        if self._num_dim == 1:
            return np.where((region[:-1] + region[1:]) == 2, 1, 0)
        elif self._num_dim == 2:
            return np.where(
                (region[:-1, :-1] + region[:-1, 1:] + region[1:, :-1] + region[1:, 1:])
                == 4,
                1,
                0,
            )
        else:
            return np.where(
                (
                    region[:-1, :-1, :-1]
                    + region[:-1, :-1, 1:]
                    + region[:-1, 1:, :-1]
                    + region[1:, :-1, :-1]
                    + region[1:, :1, :-1]
                    + region[1:, :-1, 1:]
                    + region[:-1, 1:, 1:]
                    + region[1:, 1:, 1:]
                )
                == 8,
                1,
                0,
            )

    def bc_region_mask(self, mat_name, idx):
        if self._bc_tags[idx] is not None:
            # Create 1d array of node positions
            region = np.zeros(self._num_nodes)
            region[self._bc_tags[idx][mat_name]] = 1

            # Get dimension based masks
            region = region.reshape(
                [diff.size + 1 if diff is not None else 1 for diff in self._diff]
            )

            # Append masks for each spatial dimension
            masks = []
            for d in range(len(region.shape)):
                if region.shape[d] > 1:
                    forward_region = np.take(
                        region, indices=np.arange(1, region.shape[d]), axis=d
                    )
                    masks.append(
                        (
                            np.apply_over_axes(
                                np.sum,
                                forward_region,
                                np.delete(np.arange(len(region.shape)), d),
                            )
                            > 0
                        )
                        .astype(int)
                        .flatten()
                    )

                    forward_correct = masks[-1][1:] - masks[-1][:-1]
                    masks[-1][np.argwhere(forward_correct == 1).flatten() + 1] = 0

                else:
                    masks.append(None)

            return masks

        else:
            return None

    @staticmethod
    def from_file(model_path, return_model=False):
        # Initialize and open Gmsh file
        gmsh.initialize()
        gmsh.open(model_path)

        # Create geometry object
        geometry = Geometry(gmsh.model)

        if return_model:
            return geometry, gmsh.model

        else:
            # Finalize Gmsh
            gmsh.finalize()

            return geometry

    # ===================================================
    # Getters

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_elements(self):
        return self._num_elements

    @property
    def num_dim(self):
        return self._num_dim

    @property
    def diff(self):
        return self._diff

    @property
    def dx(self):
        return self._diff[0]

    @property
    def dy(self):
        return self._diff[1]

    @property
    def dz(self):
        return self._diff[2]

    @property
    def bcs(self):
        return self._bcs

    @property
    def regions(self):
        return list(self._mat_regions.keys())

    def bc_regions(self, idx):
        return (
            list(self._bc_tags[idx].keys()) if self._bc_tags[idx] is not None else None
        )
