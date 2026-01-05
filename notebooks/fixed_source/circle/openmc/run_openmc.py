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
cylinder = openmc.ZCylinder(0, 0, 5)

# Boundary conditions
cylinder.boundary_type = "vacuum"

# Cells
fuel = openmc.Cell(fill=material, region=(-cylinder))

# Universes
universe = openmc.Universe(universe_id=0, name="Root", cells=[fuel])

# Instantiate a Geometry, register the root Universe, and export to XML
geometry = openmc.Geometry()
geometry.root_universe = universe
geometry.export_to_xml()

# =======================================================
# Create settings.xml
# Create uniform in volume cylindrical source
source = openmc.Source()
source.space = openmc.stats.CylindricalIndependent(
    r=openmc.stats.PowerLaw(a=0.0, b=5, n=1.0),
    phi=openmc.stats.Uniform(0.0, 2.0 * np.pi),  # Uniform angle
    z=openmc.stats.Uniform(0, 0),  # Z extent of source
    origin=(0, 0, 0),
)
source.angle = openmc.stats.Isotropic()
source.energy = openmc.stats.Discrete([1.0], [1.0])

# Instantiate a Settings, set all runtime parameters, and export to XML
settings_file = openmc.Settings()
settings_file.energy_mode = "multi-group"
settings_file.run_mode = "fixed source"
settings_file.batches = 1000
settings_file.particles = 500000
settings_file.output = {"tallies": True, "summary": True}
settings_file.entropy_lower_left = [0, 0, -1.0e50]
settings_file.entropy_upper_right = [10, 10, 1.0e50]
settings_file.entropy_dimension = [128, 128, 1]
settings_file.source = source
settings_file.export_to_xml()

# =======================================================
# Create plots.xml
plot_1 = openmc.Plot(plot_id=1)
plot_1.filename = "./figs/geometry.png"
plot_1.origin = [0, 0, 0.0]
plot_1.width = [10, 10]
plot_1.pixels = [500, 500]
plot_1.color_by = "material"
plot_1.basis = "xy"

# Instantiate a Plots collection and export to XML
plot_file = openmc.Plots([plot_1])
plot_file.export_to_xml()
