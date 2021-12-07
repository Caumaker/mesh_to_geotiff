import setuptools
import os
import re
import sys
import platform
import subprocess
import warnings

from distutils.version import LooseVersion
from setuptools.command.build_ext import build_ext

def main():
    with open("README.md", "r") as fh:
        long_description = fh.read()

    setuptools.setup(
        name="mesh_to_geotiff",
        version="0.1.0",
        author="Jeremy Butler",
        author_email="jeremy.butler@maptek.com.au",
        description="A class for converting a 3D mesh into a geotiff",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/jeremybutlermaptek/mesh_to_geotiff",
        packages=setuptools.find_packages(exclude=["tests","examples"]),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: MIT",
        ],
        zip_safe=False,
        install_requires=[
            'numpy',
            'numba',
            'gdal',
            'rasterio'
        ],
        test_suite="tests"
    )


if __name__ == "__main__":
    main()

