from setuptools import find_packages, setup

# Get version from tt_nte/__init__.py (always last line)
with open("tt_nte/__init__.py") as f:
    version = f.readlines()[-1].split()[-1][1:-1]

setup(
    name="tt_nte",
    version=version,
    packages=find_packages(include=["tt_nte", "tt_nte.*"]),
    install_requires=["numpy", "scikit_tt", "scipy", "gmsh"],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "pre-commit",
            "flake8",
            "black",
            "docformatter",
            "sphinx",
            "sphinx_rtd_theme",
            "jupyter",
            "sphinxcontrib.bibtex",
            "chardet",
            "nbsphinx",
        ]
    },
    discription="Tensor Trains applied to the Neutron Transport Equation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
