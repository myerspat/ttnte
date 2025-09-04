import os
import sys
import sysconfig
import warnings
import subprocess
import shutil
from pathlib import Path
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext as BuildExtension
from distutils.command.clean import clean as Clean
import torch
import pybind11

# Get version from tt_nte/__init__.py (always last line)
with open("ttnte/__init__.py") as f:
    version = f.readlines()[-1].split()[-1][1:-1]

# Get path to setup file
dir = os.path.dirname(os.path.abspath(__file__))


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(BuildExtension):
    def run(self):
        # Ensure CMake is available
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            warnings.warn("CMake is not installed, falling back to Python")
            return

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        ext_fullpath = self.get_ext_fullpath(ext.name)
        extdir = os.path.abspath(os.path.dirname(ext_fullpath))

        # Check if C++ backend should be compiled
        if bool(os.environ.get("COMPILE_CPP", True)):
            # Get configuration
            cfg = "Debug" if self.debug else "Release"

            # Get CMake arguments
            cmake_args = [
                f"-D{arg}={os.environ.get(arg, default)}"
                for arg, default in zip(
                    [
                        "Python3_ROOT_DIR",
                        "Python3_INCLUDE_DIR",
                        "Python3_EXECUTABLE",
                        "CMAKE_EXPORT_COMPILE_COMMANDS",
                        "pybind11_DIR",
                        "TORCH_INSTALL_PREFIX",
                    ],
                    [
                        sys.prefix,
                        sysconfig.get_path("include"),
                        sys.executable,
                        "ON",
                        pybind11.get_cmake_dir(),
                        os.path.abspath(os.path.dirname(torch.__file__)),
                    ],
                )
            ] + [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
                f"-DCMAKE_BUILD_TYPE={cfg}",
            ]

            # Temporary build directory
            build_temp = os.path.abspath(self.build_temp)
            os.makedirs(build_temp, exist_ok=True)

            # Configure
            try:
                subprocess.run(
                    ["cmake", ext.sourcedir] + cmake_args,
                    cwd=build_temp,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(e.stderr)
                warnings.warn("C++ backend failed to configure, falling back to Python")
                return

            # Build
            try:
                subprocess.run(
                    ["cmake", "--build", ".", "--config", cfg],
                    cwd=build_temp,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(e.stderr)
                warnings.warn("C++ backend failed to build, falling back to Python")
                return


class CleanBuild(Clean):
    def run(self):
        # Run setuptools clean
        super().run()

        # Get root directory
        root = Path(os.path.abspath(os.path.dirname(__file__)))

        # Clean folders
        for folder in ["build", "ttnte.egg-info"]:
            # Get absolute path relative to this setup.py script
            folder = root / folder
            if os.path.exists(folder):
                print(f"Removing folder: {folder}")
                shutil.rmtree(folder, ignore_errors=True)

        # Iterate through and remove pycache directories
        for dir in ["ttnte", "tests", "notebooks", "scripts"]:
            # Add root abs path
            dir = root / dir

            # Delete pycache
            for folder in dir.rglob("__pycache__"):
                if folder.is_dir():
                    print(f"Removing folder: {folder}")
                    shutil.rmtree(folder, ignore_errors=True)

        # Delete all ttnte/cpp/*.os files
        for file in (root / "ttnte/cpp").rglob("*.so"):
            if file.is_file():
                print(f"Removing file: {file}")
                file.unlink()


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
        "h5py",
        "tqdm",
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
            "papermill",
        ],
    },
    package_data={"ttnte.xs.data": ["*.json"]},
    description="Tensor Trains (TTs) applied to the Neutron Transport Equation (NTE).",
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
    ext_modules=[CMakeExtension("ttnte.cpp")],
    cmdclass={"build_ext": CMakeBuild, "clean": CleanBuild},
)
