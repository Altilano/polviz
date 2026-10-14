"""Application entry point: `python -m polviz` or the `polviz` command."""
import sys
from pathlib import Path

ICON = Path(__file__).resolve().parent / "assets" / "polviz.png"


def main():
    from . import gui  # noqa: F401  (selects the Qt binding)
    import pyqtgraph as pg
    from PySide6 import QtGui, QtWidgets

    from .gui.main_window import MainWindow

    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("PolViz")
    if ICON.exists():
        app.setWindowIcon(QtGui.QIcon(str(ICON)))
    if sys.platform == "win32":  # show our icon (not Python's) in the Windows taskbar
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("polviz.polviz")
    window = MainWindow()
    window.resize(1400, 880)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())