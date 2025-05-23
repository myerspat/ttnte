from setuptools import find_packages, setup

# Get version from tt_nte/__init__.py (always last line)
with open("ttnte/__init__.py") as f:
    version = f.readlines()[-1].split()[-1][1:-1]

setup(
    name="ttnte",
    version=version,
    packages=find_packages(include=["ttnte", "ttnte.*"]),
    install_requires=[
        "torch",
        "torchtt",
        "numpy",
        "igakit",
        "geomdl",
        "cotengra",
        "pandas",
        "matplotlib",
        "plotly",
    ],
    extras_require={
        "archive": [
            "quimb",
            "cupy",
            "scikit_tt",
        ],
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
        ],
    },
    discription="Tensor Trains applied to the Neutron Transport Equation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
