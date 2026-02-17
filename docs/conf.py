#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Xstack"
author = "Shi-Jiang Chen, Johannes Buchner, Teng Liu"
release = "1.1.2"

extensions = [
    "myst_parser",
    "sphinxarg.ext",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "Xstack Documentation"
numfig = True

autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "astropy",
    "fitsio",
    "joblib",
    "matplotlib",
    "numba",
    "numpy",
    "pandas",
    "psutil",
    "scipy",
    "sfdmap",
    "tqdm",
]

source_suffix = {
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]
