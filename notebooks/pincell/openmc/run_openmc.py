import numpy as np
import openmc

from ttnte.xs.benchmarks import c5g7

# Get XS data from ttnte
xs_server = c5g7()

# Instantiate the energy group data
groups = openmc.mgxs.EnergyGroups(np.logspace(-5, 7, xs_server.num_groups + 1))

# =======================================================
# Creating mg_cross_sections.xml file
mats = ["UO2", "Water"]
xsdata = []

for mat in mats:
    xsdata.append(openmc.XSdata(mat, groups))
    xsdata[-1].order = xs_server.num_moments - 1

    # Fill xs object
    xsdata[-1].set_chi(xs_server.chi)
    xsdata[-1].set_total(xs_server.total(mat))
    xsdata[-1].set_fission(xs_server.fission(mat))
    xsdata[-1].set_nu_fission(xs_server.nu_fission(mat))
    xsdata[-1].set_absorption(xs_server.absorption(mat))
    xsdata[-1].set_scatter_matrix(
        np.transpose(xs_server.scatter_gtg(mat), axes=(2, 1, 0))
    )

# Write XS data
mg_cross_sections_file = openmc.MGXSLibrary(groups)
mg_cross_sections_file.add_xsdatas(xsdata)
mg_cross_sections_file.export_to_hdf5()

# =======================================================
# Create materials.xml
materials = {}
for mat in mats:
    materials[mat] = openmc.Material(name=mat)
    materials[mat].set_density("macro", 1.0)
    materials[mat].add_macroscopic(openmc.Macroscopic(mat))

materials_file = openmc.Materials(materials.values())
materials_file.cross_sections = "./mgxs.h5"
materials_file.export_to_xml()

# =======================================================
# Create geometry.xml
# CSG
left = openmc.XPlane(x0=0)
right = openmc.XPlane(x0=0.63)
bottom = openmc.YPlane(y0=0)
top = openmc.YPlane(y0=0.63)
cylinder = openmc.ZCylinder(x0=0, y0=0, r=0.54)

# Boundary conditions
left.boundary_type = "reflective"
right.boundary_type = "reflective"
bottom.boundary_type = "reflective"
top.boundary_type = "reflective"

# Cells
fuel = openmc.Cell(fill=materials["UO2"], region=(-cylinder & +left & +bottom))
water = openmc.Cell(
    fill=materials["Water"], region=(+cylinder & +left & +bottom & -top & -right)
)

# Universes
pincell = openmc.Universe(universe_id=0, name="Root", cells=[fuel, water])

# Instantiate a Geometry, register the root Universe, and export to XML
geometry = openmc.Geometry()
geometry.root_universe = pincell
geometry.export_to_xml()

# =======================================================
# Create settings.xml
# Instantiate a Settings, set all runtime parameters, and export to XML
settings_file = openmc.Settings()
settings_file.energy_mode = "multi-group"
settings_file.batches = 20000
settings_file.inactive = 500
settings_file.particles = 25000
settings_file.output = {"tallies": True, "summary": True}
settings_file.source = openmc.Source(
    space=openmc.stats.Box([0, 0, -1.0], [0.63, 0.63, 1.0], only_fissionable=True)
)
settings_file.entropy_lower_left = [0, 0, -1.0e50]
settings_file.entropy_upper_right = [0.63, 0.63, 1.0e50]
settings_file.entropy_dimension = [10, 10, 1]
settings_file.export_to_xml()

# =======================================================
# Create plots.xml
plot_1 = openmc.Plot(plot_id=1)
plot_1.filename = "./figs/geometry.png"
plot_1.origin = [0.63 / 2, 0.63 / 2, 0.0]
plot_1.width = [0.63, 0.63]
plot_1.pixels = [500, 500]
plot_1.color_by = "material"
plot_1.basis = "xy"

# Instantiate a Plots collection and export to XML
plot_file = openmc.Plots([plot_1])
plot_file.export_to_xml()

# =======================================================
# Create tallies.xml
tallies_file = openmc.Tallies()

# Define regular mesh
mesh = openmc.RegularMesh(mesh_id=0)
mesh.dimension = [128, 128]
mesh.lower_left = [0, 0]
mesh.upper_right = [0.63, 0.63]

# Define tallies
energy_filter = openmc.EnergyFilter(groups.group_edges)
tally = openmc.Tally(name="Regular Mesh")
tally.filters = [openmc.MeshFilter(mesh), energy_filter]
tally.scores = ["flux"]
tallies_file.append(tally)

tally = openmc.Tally(name="Cell")
tally.filters = [openmc.CellFilter([fuel, water]), energy_filter]
tally.scores = ["flux"]
tallies_file.append(tally)

tallies_file.export_to_xml()
