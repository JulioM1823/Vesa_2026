'''
Purpose
-------
Build the dispersion explorer for the document created by
``bokeh serve Code/diagnostic_diagram/bokeh_app``.

Inputs
------
None

Outputs
-------
None
    Add the explorer layout and callbacks to bokeh.io.curdoc() when this
    entry point is executed by the Bokeh server.

Notes
-----
The diagnostic_diagram directory is added to sys.path if necessary so
the directory application can import its sibling interactive modules.
'''

# Import filesystem and module-path helpers for the directory application
from pathlib import Path
import sys

# Obtain the Bokeh document assigned to this server session
from bokeh.io import curdoc


# Make the sibling diagnostic-diagram modules importable
MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

# Import the document builder after resolving the sibling module path
from interactive_bokeh import build_document  # noqa: E402


# Populate the current session document with the interactive explorer
build_document(curdoc())
