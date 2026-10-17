# PolViz

An interactive visualizer for the polarization of light, aimed at physics students and researchers.

Build a beam path from a light source and a chain of optical elements, drag the sliders, and watch the
polarization state evolve simultaneously on the **Poincaré sphere**, in the **field projection**
(the polarization ellipse with an animated field vector) and as **Stokes parameters**.

## Features (v0.2)

- **Source**: azimuth ψ, ellipticity χ, degree of polarization, with H / V / ±45° / CCW / CW presets.
- **Elements**: linear polarizer, λ/4 and λ/2 plates, variable waveplate, optical rotator,
  depolarizer, and a mirror with Fresnel reflection from a complex index *n + ik*
  (default ≈ silver at 633 nm), adjustable angle of incidence and plane of incidence.
- Elements can be added, removed, reordered and switched on/off.
- On the sphere, each element's action is drawn as a continuous trajectory
  (e.g. a waveplate traces an arc as its retardance grows from 0 to Γ). Partially polarized
  states lie *inside* the sphere at radius equal to the degree of polarization.
- For waveplates, two trajectories can be toggled independently in the element's editor:
  the **retardance path** (solid: fast axis θ fixed, Γ grows from 0 to its value) and the
  **angle path** (dashed: Γ fixed, the fast axis turns from 0° to θ).
- "Show state at" lets you inspect the light after any element; the element's
  transmission / fast axis is drawn on the field projection.
- **Save / load setups** as JSON files (File → Open / Save / Save As); see `examples/`.
- **Export figures** (File → Export figure): the Poincaré sphere (PNG/JPG), the field
  projection and Stokes chart (SVG vector or PNG/JPG), or the whole window (PNG/JPG).

## Installation

```bash
git clone https://github.com/Altilano/polviz.git 
pip install numpy PySide6 pyqtgraph PyOpenGL pytest   
```

Run the tests with `pytest`.

## Project layout

```
polviz/
  physics.py      Stokes / Jones / Mueller calculus — no GUI dependencies
  elements.py     Source, optical elements (with editable parameters) and OpticalSystem
  serialization.py  Save / load setups as JSON
  app.py          Entry point
  gui/
    main_window.py  Controls and wiring
    views.py        Poincaré sphere (OpenGL), field projection, Stokes bar chart
    widgets.py      Slider + spin-box parameter editor
tests/
  test_physics.py   Checks Mueller matrices against Jones calculus, Malus' law, etc.
  test_features.py  Waveplate trajectories and JSON save/load
examples/           Example setups (open with File → Open)
tools/make_icon.py  Regenerates the application icon
```

The physics core is independent of the GUI, so it can also be used in scripts or notebooks:

```python
import numpy as np
from polviz.elements import OpticalSystem, Waveplate, Polarizer

system = OpticalSystem()
system.source.set("azimuth", 45)
system.elements = [Waveplate(angle=0, retardance=0.25), Polarizer(angle=90)]
print(system.states()[-1])   # Stokes vector at the output
```

## Conventions

- The beam travels along +z, fields are written as Re{**E** exp[i(kz − ωt)]}.
- S₀ = |Eₓ|² + |E_y|², S₁ = |Eₓ|² − |E_y|², S₂ = 2 Re(Eₓ* E_y), S₃ = 2 Im(Eₓ* E_y).
- **S₃ > 0 ⇔ the field rotates counter-clockwise as seen facing the source.** Because textbooks
  disagree about calling this left- or right-handed, the program only says CCW / CW.
- Angles are measured from +x towards +y. A waveplate with fast axis θ and retardance Γ rotates the
  state on the sphere by Γ about the point (cos 2θ, sin 2θ, 0), following the right-hand rule.
- Mueller calculus is used throughout so that depolarization is handled correctly.

## Setup file format

Setups are plain JSON, so they are easy to read, edit by hand, and keep under version control:

```json
{
  "format": "polviz-setup",
  "version": 1,
  "source": {"azimuth": 30.0, "ellipticity": 0.0, "dop": 1.0},
  "elements": [
    {"type": "Waveplate", "name": "λ/2 plate", "enabled": true,
     "values": {"angle": 60.0, "retardance": 0.5},
     "show_paths": {"retardance": true, "angle": true}}
  ]
}
```

Angles are in degrees and retardance in waves. Element types: `Polarizer`, `Waveplate`,
`Rotator`, `Depolarizer`, `Mirror`. Missing values fall back to their defaults.

## Adding a new element

Subclass `OpticalElement` in `elements.py`, declare its `params`, implement `mueller()`
(and optionally `mueller_partial(t)` or `trajectories()` for the curves on the sphere, and
`axis()`), then add it to `ELEMENT_FACTORIES` and `ELEMENT_TYPES`. The GUI builds the controls automatically.

## Roadmap ideas

- Elliptical retarders, partial polarizers / diattenuators, Babinet–Soleil compensator
- Export the Stokes vectors along the beam path as CSV
- Wavelength dependence (dispersive waveplates) and polychromatic light
- Animate the full beam path in 3D along z

## Yes, it was "vibecoded" with Claude.
