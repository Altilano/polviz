"""Qt user interface.  PySide6 is imported first so pyqtgraph binds to it."""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
import PySide6  # noqa: E402,F401
