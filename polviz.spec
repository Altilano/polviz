# PyInstaller build recipe for PolViz.  Build with:  pyinstaller polviz.spec
from PyInstaller.utils.hooks import collect_submodules

# PyOpenGL and pyqtgraph.opengl load some modules dynamically, so list them explicitly.
hiddenimports = (
    collect_submodules("OpenGL.platform")
    + collect_submodules("OpenGL.arrays")
    + collect_submodules("pyqtgraph.opengl")
)

a = Analysis(
    ["run_polviz.py"],
    pathex=["."],
    hiddenimports=hiddenimports,
    # Other Qt bindings and unused heavy packages: keeps the bundle smaller
    # and avoids pyqtgraph picking up the wrong binding.
    excludes=[
        "PyQt5", "PyQt6", "PySide2", "tkinter", "matplotlib", "scipy", "pandas",
        "h5py", "numba", "cupy", "pytest", "IPython", "jupyter_rfb",
    ],
    datas=[("polviz/assets", "polviz/assets")],
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PolViz",
    console=False,   # set to True while debugging to see error messages
    icon="polviz/assets/polviz.ico",
)

coll = COLLECT(exe, a.binaries, a.datas, name="PolViz")
