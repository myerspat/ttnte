import numpy as np
import openmc

from ttnte.xs import Server

# Get XS data
total = 1  # 1/cm
scattering_ratio = 0.9
xs_server = Server(
    {
        "Material": {
            "total": np.array([total]),
            "scatter_gtg": np.array([[[total * scattering_ratio]]]),
            "absorption": np.array([total * (1 - scattering_ratio)]),
            "fission": np.zeros(1),
            "nu_fission": np.zeros(1),
        }
    }
)

# Instantiate the energy group data
groups = openmc.mgxs.EnergyGroups(np.logspace(-5, 7, xs_server.num_groups + 1))

# =======================================================
# Creating mg_cross_sections.xml file
xsdata = openmc.XSdata("Material", groups)
xsdata.order = 0
xsdata.set_total(xs_server.total("Material"))
xsdata.set_absorption(xs_server.absorption("Material"))
xsdata.set_scatter_matrix(
    np.transpose(xs_server.scatter_gtg("Material"), axes=(2, 1, 0))
)

# Write XS data
mg_cross_sections_file = openmc.MGXSLibrary(groups)
mg_cross_sections_file.add_xsdatas([xsdata])
mg_cross_sections_file.export_to_hdf5()

# =======================================================
# Create materials.xml
material = openmc.Material(name="Material")
material.set_density("macro", 1.0)
material.add_macroscopic(openmc.Macroscopic("Material"))

materials_file = openmc.Materials([material])
materials_file.cross_sections = "./mgxs.h5"
materials_file.export_to_xml()

# =======================================================
# Create geometry.xml
# CSG
left = openmc.XPlane(x0=0)
right = openmc.XPlane(x0=10)
bottom = openmc.YPlane(y0=0)
top = openmc.YPlane(y0=10)
ref0 = openmc.ZPlane(-0.5)
ref1 = openmc.ZPlane(0.5)

# Boundary conditions
left.boundary_type = "vacuum"
right.boundary_type = "vacuum"
bottom.boundary_type = "vacuum"
top.boundary_type = "vacuum"
ref0.boundary_type = "reflective"
ref1.boundary_type = "reflective"

# Cells
fuel = openmc.Cell(
    fill=material, region=(+left & +bottom & -right & -top & +ref0 & -ref1)
)

# Universes
universe = openmc.Universe(universe_id=0, name="Root", cells=[fuel])

# Instantiate a Geometry, register the root Universe, and export to XML
geometry = openmc.Geometry()
geometry.root_universe = universe
geometry.export_to_xml()

# =======================================================
# Create settings.xml
# Create source
source = openmc.Source()
source.space = openmc.stats.Box([0, 0, -0.5], [10, 10, 0.5], only_fissionable=False)
source.angle = openmc.stats.Isotropic()
source.energy = openmc.stats.Discrete([1.0], [1.0])

# Instantiate a Settings, set all runtime parameters, and export to XML
settings_file = openmc.Settings()
settings_file.energy_mode = "multi-group"
settings_file.run_mode = "fixed source"
settings_file.batches = 1000
settings_file.particles = 500000
settings_file.output = {"tallies": True, "summary": True}
settings_file.source = source
settings_file.export_to_xml()

# =======================================================
# Create plots.xml
plot_1 = openmc.Plot(plot_id=1)
plot_1.filename = "./figs/geometry.png"
plot_1.origin = [10 / 2, 10 / 2, 0.0]
plot_1.width = [10, 10]
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
mesh.dimension = [128, 128, 1]
mesh.lower_left = [0, 0, -0.5]
mesh.upper_right = [10, 10, 0.5]

# Define tallies
energy_filter = openmc.EnergyFilter(groups.group_edges)
tally = openmc.Tally(name="Regular Mesh")
tally.filters = [openmc.MeshFilter(mesh), energy_filter]
tally.scores = ["flux"]
tallies_file.append(tally)

tally = openmc.Tally(name="Cell")
tally.filters = [openmc.CellFilter([fuel]), energy_filter]
tally.scores = ["flux"]
tallies_file.append(tally)

tallies_file.export_to_xml()
