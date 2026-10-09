"""Build local BladesALL documentation without importing the training runtime."""

project = "BladesALL"
author = "Shenghui Li and contributors"
copyright = "2023, Shenghui Li"
release = "0.4.0"

extensions = ["sphinx.ext.mathjax"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "sg_execution_times.rst"]
html_theme = "alabaster"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
