Installation
============

Use Python 3.10, 3.11, or 3.12. Create a clean virtual environment to avoid a
previous ``blades`` installation sharing the same import namespace. The
package distribution is named ``BladesALL`` and the Python import is ``blades``.

From the BladesALL project root, on Windows PowerShell:

.. code-block:: powershell

   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"

On Linux or macOS, create the environment with ``python3.11 -m venv .venv`` and
activate it with ``source .venv/bin/activate``. The ``dev`` extra adds pytest,
pytest-xdist, and the build frontend; omit it for runtime-only installation.
The ``examples`` extra adds matplotlib for plotting scripts.

Editable installation uses this project's source files. Restart running Python
processes after editing code. Confirm which package is loaded with:

.. code-block:: console

   python -c "import blades; print(blades.__file__)"

The result should point to this project's ``blades/__init__.py``. Dependency
versions are declared in ``pyproject.toml``, including Ray 2.39.0 and the source
project's PyTorch/torchvision constraints.

To build these local documentation pages:

.. code-block:: console

   python -m pip install -r docs/requirements.txt
   python -m sphinx -b html docs/source docs/_build/html

The documentation build does not execute experiments or import the training
runtime.
