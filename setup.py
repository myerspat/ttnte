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
        "numpy<2.0",
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
    package_data={"ttnte.xs.data": ["*.json"]},
    discription="Tensor Trains (TTs) applied to the Neutron Transport Equation (NTE).",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Patrick Myers",
    author_email="myerspat@umich.edu",
    project_urls={
        "Source Code": "https://github.com/myerspat/ttnte",
    },
    license="Apache 2.0",
    classifiers=[
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.9",
    ],
)
