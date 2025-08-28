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
        },
        "Void": {
            "total": np.array([0]),
            "scattering_gtg": np.array([0]),
            "absorption": np.zeros(1),
            "fission": np.zeros(1),
            "nu_fission": np.zeros(1),
        },
    }
)

# Instantiate the energy group data
groups = openmc.mgxs.EnergyGroups(np.logspace(-5, 7, xs_server.num_groups + 1))

# =======================================================
# Creating mg_cross_sections.xml file
mats = ["Material", "Void"]
xsdata = []

for mat in mats:
    xsdata.append(openmc.XSdata(mat, groups))
    xsdata[-1].order = xs_server.num_moments - 1

    # Fill xs object
    xsdata[-1].set_total(xs_server.total(mat))
    xsdata[-1].set_fission(np.zeros(1))
    xsdata[-1].set_nu_fission(np.zeros(1))
    xsdata[-1].set_absorption(xs_server.absorption(mat))
    xsdata[-1].set_scatter_matrix(
        np.transpose(xs_server.scatter_gtg(mat), axes=(2, 1, 0))
        if mat == "Material"
        else np.zeros(1).reshape((1, 1, 1))
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
right = openmc.XPlane(x0=6)
bottom = openmc.YPlane(y0=0)
top = openmc.YPlane(y0=6)
ref0 = openmc.ZPlane(-0.5)
ref1 = openmc.ZPlane(0.5)
cylinder = openmc.ZCylinder(x0=0, y0=0, r=5)

# Boundary conditions
left.boundary_type = "reflective"
right.boundary_type = "vacuum"
bottom.boundary_type = "reflective"
top.boundary_type = "vacuum"
ref0.boundary_type = "reflective"
ref1.boundary_type = "reflective"

# Cells
fuel = openmc.Cell(
    fill=materials["Material"], region=(-cylinder & +left & +bottom & +ref0 & -ref1)
)
water = openmc.Cell(
    fill=materials["Void"],
    region=(+cylinder & +left & +bottom & -top & -right & +ref0 & -ref1),
)

# Universes
pincell = openmc.Universe(universe_id=0, name="Root", cells=[fuel, water])

# Instantiate a Geometry, register the root Universe, and export to XML
geometry = openmc.Geometry()
geometry.root_universe = pincell
geometry.export_to_xml()

# =======================================================
# Create settings.xml
# Create uniform in volume cylindrical source
source = openmc.Source()
# source.space = openmc.stats.CylindricalIndependent(
#     r=openmc.stats.PowerLaw(a=0.0, b=5, n=1),
#     phi=openmc.stats.Uniform(0.0, np.pi / 2),  # Uniform angle
#     z=openmc.stats.Uniform(-0.5, 0.5),
#     origin=(0, 0, 0),
# )
source.space = openmc.stats.Box((0, 0, -0.5), (5, 5, 0.5))
source.angle = openmc.stats.Isotropic()
source.energy = openmc.stats.Discrete([1.0], [1.0])
source.constraints = {"domains": [fuel]}

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
plot_1.origin = [6 / 2, 6 / 2, 0.0]
plot_1.width = [6, 6]
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
mesh.upper_right = [6, 6]

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
