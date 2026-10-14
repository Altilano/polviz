"""Small reusable widgets."""
from __future__ import annotations

from functools import partial

from PySide6 import QtCore, QtWidgets


class ParamEditor(QtWidgets.QWidget):
    """Slider + spin box for each Param of a Parametrized object; edits the object in place."""

    valueChanged = QtCore.Signal(str, float)
    RESOLUTION = 1000

    def __init__(self, obj, parent=None):
        super().__init__(parent)
        self.obj = obj
        self._rows = {}
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setColumnStretch(1, 1)
        for r, p in enumerate(obj.params):
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, self.RESOLUTION)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(p.minimum, p.maximum)
            spin.setDecimals(p.decimals)
            spin.setSingleStep(p.step)
            spin.setKeyboardTracking(False)
            spin.setMinimumWidth(90)
            if p.unit:
                spin.setSuffix(f" {p.unit}")
            layout.addWidget(QtWidgets.QLabel(p.label), r, 0)
            layout.addWidget(slider, r, 1)
            layout.addWidget(spin, r, 2)
            self._rows[p.key] = (p, slider, spin)
            slider.valueChanged.connect(partial(self._on_slider, p.key))
            spin.valueChanged.connect(partial(self._on_spin, p.key))
        self.refresh()

    def refresh(self):
        """Re-read values from the object without emitting signals."""
        for key, (p, slider, spin) in self._rows.items():
            for w in (slider, spin):
                w.blockSignals(True)
            spin.setValue(self.obj[key])
            slider.setValue(self._to_slider(p, self.obj[key]))
            for w in (slider, spin):
                w.blockSignals(False)

    def _to_slider(self, p, v):
        return int(round((v - p.minimum) / (p.maximum - p.minimum) * self.RESOLUTION))

    def _on_slider(self, key, i):
        p, _, spin = self._rows[key]
        spin.blockSignals(True)
        spin.setValue(p.minimum + (p.maximum - p.minimum) * i / self.RESOLUTION)
        spin.blockSignals(False)
        self._commit(key, spin.value())

    def _on_spin(self, key, v):
        p, slider, _ = self._rows[key]
        slider.blockSignals(True)
        slider.setValue(self._to_slider(p, v))
        slider.blockSignals(False)
        self._commit(key, v)

    def _commit(self, key, v):
        self.obj.set(key, v)
        self.valueChanged.emit(key, v)
