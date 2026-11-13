import os
import warnings

# Suppress TensorFlow and zfit warnings during documentation build
os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress all Python warnings during documentation build
warnings.filterwarnings("ignore")

import phasespace

# -- General configuration ---------------------------------------------

default_role = "py:obj"

# use for classes the class and the __init__ docs combined
autoclass_content = "both"

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

napoleon_include_special_with_doc = True
napoleon_numpy_docstring = False  # docstrings use Google style

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "https://raw.githubusercontent.com/GPflow/tensorflow-intersphinx/master/tf2_py_objects.inv",
    ),
    "vector": ("https://vector.readthedocs.io/en/latest", None),
}

# MyST and MyST-NB settings. Notebooks are stored without output (see the
# nbstripout pre-commit hook) and executed during the build.
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
]
myst_heading_anchors = 3
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_execution_timeout = -1
nb_merge_streams = True
# TensorFlow writes CUDA and absl messages to stderr in the kernel process
nb_output_stderr = "remove"

project = "phasespace"
copyright = "2019-2026, Albert Puig Navarro, Jonas Eschle"
author = "Albert Puig Navarro"
version = phasespace.__version__
release = phasespace.__version__

exclude_patterns = [
    "_build",
    ".DS_Store",
    ".ipynb_checkpoints",
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "Thumbs.db",
]

# -- Options for HTML output -------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {"text": "phasespace"},
    # Sidebar configuration
    "primary_sidebar_end": [],
    "secondary_sidebar_items": [
        "page-toc",
        "edit-this-page",
    ],
    "show_toc_level": 2,
    # Footer configuration
    "footer_start": [],
    "footer_center": ["copyright"],
    "footer_end": [],
    # Navigation options
    "navigation_with_keys": True,
    # External links
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/zfit/phasespace",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}

# The pages have no subpages, so the section navigation sidebar would be empty
html_sidebars = {"**": []}

# Configure navigation links in the navbar
html_context = {
    "default_mode": "light",
    "github_user": "zfit",
    "github_repo": "phasespace",
    "github_version": "main",
    "doc_path": "docs",
}

html_title = "phasespace"
