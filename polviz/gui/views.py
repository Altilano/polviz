"""The three visualizations: Poincare sphere, field projection (ellipse) and Stokes bar chart."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui

from .. import physics as ph

# Colour for each element's trajectory (cycled); the source is white.
PALETTE = [
    (0.36, 0.72, 1.00), (1.00, 0.62, 0.25), (0.45, 0.90, 0.45), (0.95, 0.40, 0.55),
    (0.75, 0.55, 1.00), (1.00, 0.88, 0.30), (0.30, 0.90, 0.85), (0.90, 0.90, 0.90),
]
SOURCE_COLOR = (1.0, 1.0, 1.0)
AXIS_COLORS = {"S1": (240, 90, 90), "S2": (90, 210, 110), "S3": (100, 150, 255)}


def element_color(index: int):
    return PALETTE[index % len(PALETTE)]


def _normalized(S, eps=1e-9):
    """(s1, s2, s3) = (S1, S2, S3)/S0, i.e. a point inside the sphere at radius DOP; None if no light."""
    return None if S[0] <= eps else np.asarray(S[1:4]) / S[0]


# ---------------------------------------------------------------------------
# Poincare sphere
# ---------------------------------------------------------------------------
class PoincareView(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor((18, 20, 26))
        self.reset_camera()
        self._build_static()
        self._lines: list[gl.GLLinePlotItem] = []
        self._vector = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1, 1, 1, 0.9), width=2,
                                         antialias=True)
        self.addItem(self._vector)
        self._points = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=10, pxMode=True)
        self.addItem(self._points)

    def reset_camera(self):
        self.setCameraPosition(distance=4.3, elevation=22, azimuth=35)

    def _build_static(self):
        sphere = gl.GLMeshItem(meshdata=gl.MeshData.sphere(rows=40, cols=80, radius=1.0),
                               smooth=True, shader="shaded", color=(0.35, 0.5, 0.85, 0.13),
                               glOptions="translucent")
        sphere.setDepthValue(10)  # draw last so trajectories behind it show through
        self.addItem(sphere)

        t = np.linspace(0, 2 * np.pi, 181)
        for lat in (-60, -30, 0, 30, 60):  # circles of constant ellipticity
            z, r = np.sin(np.radians(lat)), np.cos(np.radians(lat))
            pos = np.column_stack([r * np.cos(t), r * np.sin(t), np.full_like(t, z)])
            alpha = 0.45 if lat == 0 else 0.18
            self.addItem(gl.GLLinePlotItem(pos=pos, color=(0.8, 0.85, 1, alpha), width=1,
                                           antialias=True))
        for lon in range(0, 180, 30):  # meridians: constant azimuth
            phi = np.radians(lon)
            pos = np.column_stack([np.cos(t) * np.cos(phi), np.cos(t) * np.sin(phi), np.sin(t)])
            self.addItem(gl.GLLinePlotItem(pos=pos, color=(0.8, 0.85, 1, 0.18), width=1,
                                           antialias=True))

        labels = {
            "S1": [((1.32, 0, 0), "S₁"), ((1.12, 0, 0), "H"), ((-1.18, 0, 0), "V")],
            "S2": [((0, 1.32, 0), "S₂"), ((0, 1.12, 0), "+45°"), ((0, -1.22, 0), "−45°")],
            "S3": [((0, 0, 1.32), "S₃"), ((0, 0, 1.12), "CCW"), ((0, 0, -1.18), "CW")],
        }
        font = QtGui.QFont("Sans", 11)
        for k, (axis, rgb) in enumerate(AXIS_COLORS.items()):
            pos = np.zeros((2, 3))
            pos[:, k] = (-1.25, 1.25)
            self.addItem(gl.GLLinePlotItem(pos=pos, color=tuple(c / 255 for c in rgb) + (0.9,),
                                           width=2, antialias=True))
            for p, text in labels[axis]:
                self.addItem(gl.GLTextItem(pos=p, text=text, color=QtGui.QColor(*rgb), font=font))

    def set_states(self, states, paths, highlight: int):
        # Trajectories: reuse line items, hide the unused ones.
        for i, path in enumerate(paths):
            if i >= len(self._lines):
                item = gl.GLLinePlotItem(width=3, antialias=True, mode="line_strip")
                self.addItem(item)
                self._lines.append(item)
            pts = np.array([p for p in (_normalized(S) for S in path) if p is not None])
            item = self._lines[i]
            if len(pts) >= 2:
                item.setData(pos=pts, color=element_color(i) + (1.0,))
                item.setVisible(True)
            else:
                item.setVisible(False)
        for item in self._lines[len(paths):]:
            item.setVisible(False)

        # State markers.
        pos, colors, sizes = [], [], []
        for i, S in enumerate(states):
            p = _normalized(S)
            if p is None:
                continue
            pos.append(p)
            rgb = SOURCE_COLOR if i == 0 else element_color(i - 1)
            colors.append(rgb + (1.0,))
            sizes.append(17 if i == highlight else 9)
        if pos:
            self._points.setData(pos=np.array(pos), color=np.array(colors), size=np.array(sizes))
            self._points.setVisible(True)
        else:
            self._points.setVisible(False)

        p = _normalized(states[highlight])
        if p is not None:
            self._vector.setData(pos=np.array([[0, 0, 0], p]))
        self._vector.setVisible(p is not None)


# ---------------------------------------------------------------------------
# Field projection in the transverse plane
# ---------------------------------------------------------------------------
class EllipseView(pg.PlotWidget):
    """Polarization ellipse traced by the E-field, with an animated field vector."""

    PERIOD_S = 3.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Field projection (facing the source)")
        self.setAspectLocked(True)
        self.setRange(xRange=(-1.15, 1.15), yRange=(-1.15, 1.15), padding=0)
        self.setMouseEnabled(False, False)
        self.hideButtons()
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("bottom", "Eₓ")
        self.setLabel("left", "E_y")

        t = np.linspace(0, 2 * np.pi, 200)
        self.plot(np.cos(t), np.sin(t), pen=pg.mkPen((110, 110, 110), style=QtCore.Qt.DashLine))
        self._axis_line = self.plot(pen=pg.mkPen((255, 200, 80), width=2, style=QtCore.Qt.DashLine))
        self._axis_text = pg.TextItem(color=(255, 200, 80), anchor=(0.5, 1.2))
        self.addItem(self._axis_text)
        self._curve = self.plot(pen=pg.mkPen((90, 180, 255), width=3))
        self._trail = self.plot(pen=pg.mkPen((255, 255, 255), width=4))
        self._vector = self.plot(pen=pg.mkPen((255, 255, 255, 160), width=2))
        self._dot = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(255, 255, 255))
        self.addItem(self._dot)
        self._note = pg.TextItem(color=(200, 200, 200), anchor=(0, 0))
        self._note.setPos(-1.12, 1.12)
        self.addItem(self._note)

        self._amp = (0.0, 0.0, 0.0)  # Ax, Ay, phase difference
        self._phase = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    def set_state(self, S, ref_intensity: float, axis=None):
        st = ph.describe(S)
        if not st.has_light:
            self._amp = (0.0, 0.0, 0.0)
            self._note.setText("No light")
        else:
            E = ph.jones_from_stokes(S) / np.sqrt(ref_intensity)
            self._amp = (abs(E[0]), abs(E[1]), float(np.angle(E[1])))
            unpol = 1 - st.dop
            self._note.setText(f"Unpolarized: {100 * unpol:.0f} %" if unpol > 0.005 else "")
        ax, ay, phi = self._amp
        t = np.linspace(0, 2 * np.pi, 200)
        self._curve.setData(ax * np.cos(t), ay * np.cos(t - phi))

        if axis is None:
            self._axis_line.setData([], [])
            self._axis_text.setText("")
        else:
            angle, label = axis
            d = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            self._axis_line.setData([-1.1 * d[0], 1.1 * d[0]], [-1.1 * d[1], 1.1 * d[1]])
            self._axis_text.setText(label)
            self._axis_text.setPos(1.1 * d[0], 1.1 * d[1])
        self._draw_vector()

    def _tick(self):
        self._phase = (self._phase + 2 * np.pi * 0.03 / self.PERIOD_S) % (2 * np.pi)
        self._draw_vector()

    def _draw_vector(self):
        ax, ay, phi = self._amp
        t = np.linspace(self._phase - 0.7, self._phase, 25)
        x, y = ax * np.cos(t), ay * np.cos(t - phi)
        self._trail.setData(x, y)
        self._vector.setData([0, x[-1]], [0, y[-1]])
        self._dot.setData([x[-1]], [y[-1]])


# ---------------------------------------------------------------------------
# Stokes parameters
# ---------------------------------------------------------------------------
class StokesView(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Stokes parameters (normalized to source S₀)")
        self.setMouseEnabled(False, False)
        self.hideButtons()
        self.setXRange(-0.6, 3.6, padding=0)
        self.setYRange(-1.1, 1.25, padding=0)
        self.showGrid(y=True, alpha=0.15)
        self.getAxis("bottom").setTicks([[(0, "S₀"), (1, "S₁"), (2, "S₂"), (3, "S₃")]])
        self.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen((150, 150, 150))))
        brushes = [pg.mkBrush(220, 220, 220)] + [pg.mkBrush(*c) for c in AXIS_COLORS.values()]
        self._bars = pg.BarGraphItem(x=[0, 1, 2, 3], height=[0, 0, 0, 0], width=0.6, brushes=brushes)
        self.addItem(self._bars)
        self._labels = []
        for i in range(4):
            label = pg.TextItem(color=(230, 230, 230), anchor=(0.5, 1))
            self.addItem(label)
            self._labels.append(label)

    def set_state(self, S, ref_intensity: float):
        h = np.asarray(S, float) / ref_intensity
        self._bars.setOpts(height=h)
        for i, (label, v) in enumerate(zip(self._labels, h)):
            label.setText(f"{v:+.3f}")
            label.setPos(i, max(v, 0) + 0.02)
