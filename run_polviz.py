"""Launcher script for PyInstaller (and for running without `-m`).

PyInstaller needs a plain script as its entry point; the package itself uses
relative imports, so it cannot be bundled from polviz/app.py directly.
"""
import sys

from polviz.app import main

if __name__ == "__main__":
    sys.exit(main())
