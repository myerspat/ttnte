import numpy as np
import openmc

from ttnte.xs import Server

xs_server = Server(
    {
        "Source": {
            "total": np.array([0.01]),
            "scatter_gtg": np.array([[[0.008]]]),
        },
        "Void": {
            "total": np.array([0]),
            "scatter_gtg": np.array([[[0]]]),
        },
        "Shield": {
            "total": np.array([3]),
            "scatter_gtg": np.array([[[0.5]]]),
        },
    }
)

# Instantiate the energy group data
groups = openmc.mgxs.EnergyGroups(np.logspace(-5, 7, xs_server.num_groups + 1))

# =======================================================
# Creating mg_cross_sections.xml file
mats = ["Source", "Void", "Shield"]
xsdata = []

for mat in mats:
    xsdata.append(openmc.XSdata(mat, groups))
    xsdata[-1].order = xs_server.num_moments - 1

    scatter = np.zeros((1, 1, 1))
    if mat != "Void":
        scatter = xs_server.scatter_gtg(mat)

    # Fill xs object
    xsdata[-1].set_total(xs_server.total(mat))
    xsdata[-1].set_fission(np.zeros(1))
    xsdata[-1].set_nu_fission(np.zeros(1))
    xsdata[-1].set_absorption(xs_server.total(mat) - scatter.flatten())
    xsdata[-1].set_scatter_matrix((scatter))

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
## Initialize dimensional variables
X = 10  # Channel pitch

# Cruciform
R = 2  # Radius defining valleys of fixed source
delta = 1  # Width of lobes
d2 = delta * 0.5  # Half width of lobes
x = 0.25  # Portrusion of lobes

# Shielding
I = 3.75  # Inner radius
O = 4.5  # Outer radius

cylinderIa = openmc.ZCylinder(d2 + R, d2 + R, R)
elipseIa = openmc.Quadric(
    a=d2**-2,
    b=x**-2,
    h=-2 * (d2 + R) / (x**2),
    k=-1 + (d2 + R) ** 2 / (x**2),
)
elipseIIa = openmc.Quadric(
    a=x**-2,
    b=d2**-2,
    g=-2 * (d2 + R) / (x**2),
    k=-1 + (d2 + R) ** 2 / (x**2),
)
square = (
    +openmc.XPlane(0, boundary_type="reflective")
    & -openmc.XPlane(d2 + R)
    & +openmc.YPlane(0, boundary_type="reflective")
    & -openmc.YPlane(d2 + R)
)
source = (
    ((square & +cylinderIa) | (-elipseIa | -elipseIIa))
    & +openmc.XPlane(0, boundary_type="reflective")
    & +openmc.YPlane(0, boundary_type="reflective")
)

# =======================================================
# Make shield
shield = (
    +openmc.ZCylinder(r=I)
    & -openmc.ZCylinder(r=O)
    & +openmc.XPlane(0, boundary_type="reflective")
    & +openmc.YPlane(0, boundary_type="reflective")
)

# =======================================================
# Void
left = openmc.XPlane(0, boundary_type="reflective")
bottom = openmc.YPlane(0, boundary_type="reflective")
right = openmc.XPlane(X / 2, boundary_type="vacuum")
top = openmc.YPlane(X / 2, boundary_type="vacuum")
ref0 = openmc.ZPlane(-0.5, boundary_type="reflective")
ref1 = openmc.ZPlane(0.5, boundary_type="reflective")
coolant = (-right & -top & +left & +bottom) & (~source & ~shield)

# ======================================================
# Define cells and universe
cell0 = openmc.Cell()
cell0.region = source & +ref0 & -ref1
cell0.fill = materials["Source"]
cell1 = openmc.Cell()
cell1.region = shield & +ref0 & -ref1
cell1.fill = materials["Shield"]
cell2 = openmc.Cell()
cell2.region = coolant & +ref0 & -ref1
cell2.fill = materials["Void"]

# Universe
universe = openmc.Universe(cells=[cell0, cell1, cell2])
geometry = openmc.Geometry(universe)
geometry.export_to_xml()

# ======================================================
# Plots
plot = openmc.Plot()
plot.filename = "./figs/geometry.png"
plot.origin = (X / 4, X / 4, 0)
plot.width = (X / 2, X / 2)
plot.pixels = (500, 500)
plot.color_by = "material"
plot.basis = "xy"

plots = openmc.Plots()
plots.append(plot)
plots.export_to_xml()

# =======================================================
# Create settings.xml
# Create uniform in volume cylindrical source
source = openmc.IndependentSource()
source.space = openmc.stats.Box((0, 0, -0.5), (d2 + R + x, d2 + R + x, 0.5))
source.angle = openmc.stats.Isotropic()
source.energy = openmc.stats.Discrete([1.0], [1.0])
source.constraints = {"domains": [cell0]}

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
# Create tallies.xml
tallies_file = openmc.Tallies()

# Define regular mesh
mesh = openmc.RegularMesh(mesh_id=0)
mesh.dimension = [128, 128]
mesh.lower_left = [0, 0]
mesh.upper_right = [5, 5]

# Define tallies
energy_filter = openmc.EnergyFilter(groups.group_edges)
tally = openmc.Tally(name="Regular Mesh")
tally.filters = [openmc.MeshFilter(mesh), energy_filter]
tally.scores = ["flux"]
tallies_file.append(tally)

tallies_file.export_to_xml()
