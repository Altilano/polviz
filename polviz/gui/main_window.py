"""Main application window."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import physics as ph
from ..elements import ELEMENT_FACTORIES, default_system
from ..serialization import load_setup, save_setup
from .views import EllipseView, PoincareView, StokesView, element_color, export_plot
from .widgets import ParamEditor

SOURCE_PRESETS = {
    "H": (0, 0), "V": (90, 0), "+45°": (45, 0),
    "−45°": (-45, 0), "CCW": (0, 45), "CW": (0, -45),
}

CONVENTIONS = """<h3>Conventions</h3>
<p>The beam travels along +z; the field projection is drawn as seen by an observer
<b>facing the source</b> (looking into the beam).</p>
<p>S₀ = |Eₓ|² + |E_y|², S₁ = |Eₓ|² − |E_y|², S₂ = 2 Re(Eₓ* E_y), S₃ = 2 Im(Eₓ* E_y),
with fields written as Re{E exp[i(kz − ωt)]}.</p>
<p><b>S₃ &gt; 0 ⇔ the field rotates counter-clockwise (CCW)</b> as seen facing the source.
Textbooks disagree on whether this is called left- or right-handed, so the program
only uses CCW/CW.</p>
<p>On the Poincaré sphere a point is drawn at (S₁, S₂, S₃)/S₀, so partially polarized
light sits inside the sphere at radius equal to its degree of polarization.
A waveplate rotates the state by its retardance Γ about its fast-axis point,
counter-clockwise when viewed from outside the sphere along that axis.</p>
<p>Angles are measured from +x towards +y.</p>
<p>Waveplate trajectories: the <b>solid</b> curve keeps the fast axis θ fixed and grows the
retardance from 0 to Γ (the state inside the plate as light propagates). The <b>dashed</b>
curve keeps Γ fixed and turns the fast axis from 0° to θ; it starts at the output the plate
would give with θ = 0° and ends at the actual output.</p>"""

SETUP_FILTER = "PolViz setup (*.json);;All files (*)"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.system = default_system()
        self.current_path: Path | None = None
        self.dirty = False
        self.obs_index = len(self.system.elements)  # which state the 2D views show
        self._editor = None

        self.poincare = PoincareView()
        self.ellipse = EllipseView()
        self.stokes = StokesView()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_views())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 1000])
        self.setCentralWidget(splitter)
        self._build_menu()

        self._rebuild_chain(select=0)
        self.refresh()
        self._update_title()

    # ------------------------------------------------------------------ layout
    def _build_menu(self):
        def action(menu, text, slot, shortcut=None):
            act = menu.addAction(text)
            if shortcut is not None:
                act.setShortcut(shortcut)
            act.triggered.connect(slot)
            return act

        file_menu = self.menuBar().addMenu("&File")
        action(file_menu, "&New", self.new_setup, QtGui.QKeySequence.New)
        action(file_menu, "&Open…", self.open_setup, QtGui.QKeySequence.Open)
        action(file_menu, "&Save", self.save, QtGui.QKeySequence.Save)
        action(file_menu, "Save &As…", self.save_as, QtGui.QKeySequence.SaveAs)
        file_menu.addSeparator()
        export_menu = file_menu.addMenu("&Export figure")
        action(export_menu, "Poincaré sphere…", lambda: self.export_figure("sphere"))
        action(export_menu, "Field projection…", lambda: self.export_figure("ellipse"))
        action(export_menu, "Stokes parameters…", lambda: self.export_figure("stokes"))
        action(export_menu, "Whole window…", lambda: self.export_figure("window"))
        file_menu.addSeparator()
        action(file_menu, "&Quit", self.close, QtGui.QKeySequence.Quit)

        view_menu = self.menuBar().addMenu("&View")
        action(view_menu, "Reset sphere camera", self.poincare.reset_camera)
        help_menu = self.menuBar().addMenu("&Help")
        action(help_menu, "Conventions…",
               lambda: QtWidgets.QMessageBox.information(self, "Conventions", CONVENTIONS))

    def _build_controls(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # Source
        src_box = QtWidgets.QGroupBox("Light source")
        src_layout = QtWidgets.QVBoxLayout(src_box)
        self.source_editor = ParamEditor(self.system.source)
        self.source_editor.valueChanged.connect(lambda *_: self._modified())
        src_layout.addWidget(self.source_editor)
        presets = QtWidgets.QHBoxLayout()
        for name, (psi, chi) in SOURCE_PRESETS.items():
            btn = QtWidgets.QPushButton(name)
            btn.setToolTip(f"ψ = {psi}°, χ = {chi}°")
            btn.clicked.connect(lambda _=False, p=psi, c=chi: self._set_source(p, c))
            presets.addWidget(btn)
        src_layout.addLayout(presets)
        layout.addWidget(src_box)

        # Chain
        chain_box = QtWidgets.QGroupBox("Optical elements (in beam order)")
        chain_layout = QtWidgets.QVBoxLayout(chain_box)
        self.chain_list = QtWidgets.QListWidget()
        self.chain_list.currentRowChanged.connect(self._on_select)
        self.chain_list.itemChanged.connect(self._on_item_changed)
        chain_layout.addWidget(self.chain_list)
        buttons = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QToolButton()
        add_btn.setText("Add ▾")
        add_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(add_btn)
        for name, factory in ELEMENT_FACTORIES.items():
            menu.addAction(name).triggered.connect(lambda _=False, f=factory: self._add_element(f()))
        add_btn.setMenu(menu)
        buttons.addWidget(add_btn)
        for text, tip, slot in (("Remove", "Remove selected element", self._remove_element),
                                ("▲", "Move up", lambda: self._move_element(-1)),
                                ("▼", "Move down", lambda: self._move_element(+1))):
            btn = QtWidgets.QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            buttons.addWidget(btn)
        chain_layout.addLayout(buttons)
        hint = QtWidgets.QLabel("Tick/untick to switch an element on or off.")
        hint.setStyleSheet("color: gray;")
        chain_layout.addWidget(hint)
        layout.addWidget(chain_box, 1)

        # Selected element parameters
        self.editor_box = QtWidgets.QGroupBox("Selected element")
        self.editor_layout = QtWidgets.QVBoxLayout(self.editor_box)
        layout.addWidget(self.editor_box)
        return panel

    def _build_views(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Show state at:"))
        self.obs_combo = QtWidgets.QComboBox()
        self.obs_combo.setMinimumWidth(320)
        self.obs_combo.currentIndexChanged.connect(self._on_obs_changed)
        top.addWidget(self.obs_combo)
        top.addStretch(1)
        layout.addLayout(top)

        vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vsplit.addWidget(self.poincare)
        bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bottom.addWidget(self.ellipse)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.stokes, 1)
        self.info = QtWidgets.QLabel()
        self.info.setTextFormat(QtCore.Qt.RichText)
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        right_layout.addWidget(self.info)
        bottom.addWidget(right)
        vsplit.addWidget(bottom)
        vsplit.setSizes([520, 360])
        layout.addWidget(vsplit, 1)
        return panel

    # ------------------------------------------------------------------ chain editing
    def _rebuild_chain(self, select=None):
        self.chain_list.blockSignals(True)
        self.chain_list.clear()
        for i, el in enumerate(self.system.elements):
            item = QtWidgets.QListWidgetItem()
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if el.enabled else QtCore.Qt.Unchecked)
            item.setForeground(QtGui.QColor.fromRgbF(*element_color(i)))
            self.chain_list.addItem(item)
        self._update_labels()
        self.chain_list.blockSignals(False)

        self.obs_combo.blockSignals(True)
        self.obs_combo.clear()
        self.obs_combo.addItem("Source")
        n = len(self.system.elements)
        for i, el in enumerate(self.system.elements):
            suffix = "  (output)" if i == n - 1 else ""
            self.obs_combo.addItem(f"After {i + 1}: {el.name}{suffix}")
        self.obs_combo.blockSignals(False)

        if select is not None and 0 <= select < n:
            self.chain_list.setCurrentRow(select)
            self._on_select(select)
        else:
            self._on_select(-1)

    def _update_labels(self):
        for i, el in enumerate(self.system.elements):
            self.chain_list.item(i).setText(f"{i + 1}. {el.label()}")

    def _add_element(self, element):
        row = self.chain_list.currentRow()
        pos = row + 1 if row >= 0 else len(self.system.elements)
        self.system.elements.insert(pos, element)
        self._rebuild_chain(select=pos)
        self._modified()

    def _remove_element(self):
        row = self.chain_list.currentRow()
        if row < 0:
            return
        del self.system.elements[row]
        self._rebuild_chain(select=min(row, len(self.system.elements) - 1))
        self._modified()

    def _move_element(self, step):
        row = self.chain_list.currentRow()
        new = row + step
        els = self.system.elements
        if row < 0 or not 0 <= new < len(els):
            return
        els[row], els[new] = els[new], els[row]
        self._rebuild_chain(select=new)
        self._modified()

    def _on_item_changed(self, item):
        row = self.chain_list.row(item)
        self.system.elements[row].enabled = item.checkState() == QtCore.Qt.Checked
        self.chain_list.blockSignals(True)
        self._update_labels()
        self.chain_list.blockSignals(False)
        self._modified()

    def _on_select(self, row):
        if self._editor is not None:
            self._editor.setParent(None)
            self._editor.deleteLater()
            self._editor = None
        if 0 <= row < len(self.system.elements):
            el = self.system.elements[row]
            self.editor_box.setTitle(f"Selected element: {row + 1}. {el.name}")
            self._editor = self._make_element_editor(el)
            self.editor_layout.addWidget(self._editor)
            self.obs_index = row + 1
        else:
            self.editor_box.setTitle("Selected element: none")
            self.obs_index = len(self.system.elements)
        self.obs_combo.blockSignals(True)
        self.obs_combo.setCurrentIndex(self.obs_index)
        self.obs_combo.blockSignals(False)
        self.refresh()

    def _make_element_editor(self, el):
        """Parameter sliders plus, for elements that have them, trajectory toggles."""
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        params = ParamEditor(el)
        params.valueChanged.connect(self._on_element_param)
        layout.addWidget(params)
        if el.path_options:
            layout.addWidget(QtWidgets.QLabel("Show on the Poincaré sphere:"))
            for key, label, _ in el.path_options:
                check = QtWidgets.QCheckBox(label)
                check.setChecked(el.show_paths[key])
                check.toggled.connect(lambda on, k=key, e=el: self._on_path_toggle(e, k, on))
                layout.addWidget(check)
        return box

    def _on_path_toggle(self, element, key, on):
        element.show_paths[key] = on
        self._modified()

    def _on_element_param(self, *_):
        self.chain_list.blockSignals(True)
        self._update_labels()
        self.chain_list.blockSignals(False)
        self._modified()

    def _on_obs_changed(self, index):
        self.obs_index = max(index, 0)
        self.refresh()

    def _set_source(self, psi, chi):
        self.system.source.set("azimuth", psi)
        self.system.source.set("ellipticity", chi)
        self.source_editor.refresh()
        self._modified()

    # ------------------------------------------------------------------ files
    def _modified(self):
        if not self.dirty:
            self.dirty = True
            self._update_title()
        self.refresh()

    def _update_title(self):
        name = self.current_path.name if self.current_path else "Untitled"
        self.setWindowTitle(f"{name}{' *' if self.dirty else ''} — PolViz")

    def _maybe_save(self) -> bool:
        """Ask to save unsaved changes; False means the user cancelled."""
        if not self.dirty:
            return True
        answer = QtWidgets.QMessageBox.question(
            self, "Unsaved changes", "Save the changes to the current setup?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel)
        if answer == QtWidgets.QMessageBox.Save:
            return self.save()
        return answer == QtWidgets.QMessageBox.Discard

    def _set_system(self, system, path):
        # Keep the same source object (the source editor is bound to it).
        self.system.source.values.update(system.source.values)
        self.system.elements = system.elements
        self.source_editor.refresh()
        self.current_path = path
        self.dirty = False
        self._rebuild_chain(select=0 if system.elements else None)
        self.refresh()
        self._update_title()

    def new_setup(self):
        if self._maybe_save():
            self._set_system(default_system(), None)

    def open_setup(self):
        if not self._maybe_save():
            return
        start = str(self.current_path.parent) if self.current_path else ""
        name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open setup", start, SETUP_FILTER)
        if not name:
            return
        try:
            system = load_setup(name)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "Could not open setup", str(exc))
            return
        self._set_system(system, Path(name))

    def save(self) -> bool:
        if self.current_path is None:
            return self.save_as()
        try:
            save_setup(self.system, self.current_path)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Could not save setup", str(exc))
            return False
        self.dirty = False
        self._update_title()
        return True

    def save_as(self) -> bool:
        start = str(self.current_path) if self.current_path else "setup.json"
        name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save setup", start, SETUP_FILTER)
        if not name:
            return False
        path = Path(name)
        if not path.suffix:
            path = path.with_suffix(".json")
        self.current_path = path
        return self.save()

    def export_figure(self, which: str):
        titles = {"sphere": "Poincaré sphere", "ellipse": "Field projection",
                  "stokes": "Stokes parameters", "window": "Whole window"}
        vector_ok = which in ("ellipse", "stokes")
        filters = "PNG image (*.png);;JPEG image (*.jpg)"
        if vector_ok:
            filters = "SVG vector image (*.svg);;" + filters
        base = self.current_path.stem if self.current_path else "polviz"
        default = f"{base}_{which}.{'svg' if vector_ok else 'png'}"
        name, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, f"Export {titles[which]}", default, filters)
        if not name:
            return
        path = Path(name)
        if not path.suffix:
            path = path.with_suffix("." + chosen.split("*.")[1].rstrip(")"))
        try:
            if which == "sphere":
                self.poincare.export_image(path)
            elif which == "ellipse":
                export_plot(self.ellipse, path)
            elif which == "stokes":
                export_plot(self.stokes, path)
            else:
                self.grab_window().save(str(path))
        except Exception as exc:  # exporters can fail in many ways; report, don't crash
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))

    def grab_window(self) -> QtGui.QPixmap:
        """Screenshot of the window; the 3D view is pasted from its framebuffer, which
        (unlike QWidget.grab) includes the axis labels."""
        pixmap = self.grab()
        sphere = self.poincare.grabFramebuffer()
        sphere.setDevicePixelRatio(self.poincare.devicePixelRatioF())
        painter = QtGui.QPainter(pixmap)
        painter.drawImage(self.poincare.mapTo(self, QtCore.QPoint(0, 0)), sphere)
        painter.end()
        return pixmap

    def closeEvent(self, event):
        if self._maybe_save():
            event.accept()
        else:
            event.ignore()

    # ------------------------------------------------------------------ update views
    def refresh(self):
        if not hasattr(self, "info"):
            return
        states = self.system.states()
        trajectories = self.system.trajectories()
        obs = int(np.clip(self.obs_index, 0, len(states) - 1))
        ref = states[0][0]
        axis = None
        if obs > 0 and self.system.elements[obs - 1].enabled:
            axis = self.system.elements[obs - 1].axis()
        self.poincare.set_states(states, trajectories, obs)
        self.ellipse.set_state(states[obs], ref, axis)
        self.stokes.set_state(states[obs], ref)
        self.info.setText(self._info_text(states[obs], ref))

    @staticmethod
    def _info_text(S, ref):
        st = ph.describe(S)
        if not st.has_light:
            return "<b>No light</b> reaches this point."
        rot = {"CCW": "counter-clockwise (S₃ &gt; 0)", "CW": "clockwise (S₃ &lt; 0)",
               "linear": "linear", "none": "— (unpolarized)"}[st.rotation]
        s = S[1:] / S[0]
        return (
            f"<table cellspacing='4'>"
            f"<tr><td>Intensity I/I₀</td><td><b>{S[0] / ref:.4f}</b></td>"
            f"<td>&nbsp;&nbsp;DOP</td><td><b>{st.dop:.4f}</b></td></tr>"
            f"<tr><td>Azimuth ψ</td><td><b>{np.degrees(st.azimuth):+.2f}°</b></td>"
            f"<td>&nbsp;&nbsp;Ellipticity χ</td><td><b>{np.degrees(st.ellipticity):+.2f}°</b></td></tr>"
            f"<tr><td>Sphere point</td><td colspan='3'><b>({s[0]:+.3f}, {s[1]:+.3f}, {s[2]:+.3f})</b></td></tr>"
            f"<tr><td>Rotation</td><td colspan='3'><b>{rot}</b></td></tr></table>"
        )
