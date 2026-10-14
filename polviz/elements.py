"""Optical elements and the light source, each with a list of editable parameters.

Parameters are stored in user-friendly units (degrees, waves) and converted to
radians when the Mueller matrix is built.  Adding a new element type means
subclassing OpticalElement, declaring its `params`, implementing `mueller()` and
registering it in ELEMENT_FACTORIES; the GUI builds its controls automatically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import physics as ph


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    minimum: float
    maximum: float
    default: float
    unit: str = ""
    decimals: int = 1
    step: float = 1.0


class Parametrized:
    """Base class: holds a dict of parameter values described by `params`."""
    kind = "Object"
    params: tuple[Param, ...] = ()

    def __init__(self, name: str | None = None, **values):
        self.values = {p.key: p.default for p in self.params}
        unknown = set(values) - set(self.values)
        if unknown:
            raise ValueError(f"Unknown parameter(s) for {self.kind}: {sorted(unknown)}")
        for key, value in values.items():
            self.set(key, value)
        self.name = name or self.kind

    def set(self, key: str, value: float) -> None:
        spec = next(p for p in self.params if p.key == key)
        self.values[key] = float(np.clip(value, spec.minimum, spec.maximum))

    def __getitem__(self, key: str) -> float:
        return self.values[key]


# ---------------------------------------------------------------------------
# Light source
# ---------------------------------------------------------------------------
class LightSource(Parametrized):
    kind = "Source"
    params = (
        Param("azimuth", "Azimuth ψ", -90, 90, 0, "°"),
        Param("ellipticity", "Ellipticity χ", -45, 45, 0, "°"),
        Param("dop", "Degree of pol.", 0, 1, 1, "", 2, 0.05),
    )

    def stokes(self) -> np.ndarray:
        return ph.stokes_from_ellipse(np.radians(self["azimuth"]), np.radians(self["ellipticity"]),
                                      self["dop"])


# ---------------------------------------------------------------------------
# Optical elements
# ---------------------------------------------------------------------------
class OpticalElement(Parametrized):
    kind = "Element"
    # Optional trajectories that can be toggled: (key, label, shown by default)
    path_options: tuple[tuple[str, str, bool], ...] = ()

    def __init__(self, name: str | None = None, **values):
        super().__init__(name, **values)
        self.enabled = True
        self.show_paths = {key: default for key, _, default in self.path_options}

    def trajectories(self, S_in, samples: int = 60) -> list[tuple[str, np.ndarray]]:
        """Curves to draw on the Poincare sphere for this element: [(style, stokes_array)].

        style is "solid" or "dashed".  The default is one solid curve following
        mueller_partial(t) for t from 0 to 1.
        """
        ts = np.linspace(0.0, 1.0, samples + 1)
        return [("solid", np.array([self.mueller_partial(t) @ S_in for t in ts]))]

    def mueller(self) -> np.ndarray:
        raise NotImplementedError

    def mueller_partial(self, t: float) -> np.ndarray:
        """Mueller matrix for a fraction t in [0, 1] of the element's action.

        Used to draw a continuous trajectory on the Poincare sphere.  The default
        is a linear blend (an incoherent mixture), elements with a natural
        continuous parameter (retardance, rotation, ...) override it.
        """
        return (1 - t) * np.eye(4) + t * self.mueller()

    def axis(self):
        """(angle in degrees, label) of a characteristic axis to draw, or None."""
        return None

    def describe(self) -> str:
        return ""

    def label(self) -> str:
        text = f"{self.name}  —  {self.describe()}"
        return text if self.enabled else text + "  (off)"


class Polarizer(OpticalElement):
    kind = "Linear polarizer"
    params = (Param("angle", "Transmission axis θ", -90, 90, 0, "°"),)

    def mueller(self):
        return ph.linear_polarizer(np.radians(self["angle"]))

    def axis(self):
        return self["angle"], "transmission axis"

    def describe(self):
        return f"θ = {self['angle']:.1f}°"


class Waveplate(OpticalElement):
    kind = "Waveplate"
    params = (
        Param("angle", "Fast axis θ", -90, 90, 0, "°"),
        Param("retardance", "Retardance Γ", 0, 1, 0.25, "λ", 3, 0.01),
    )

    def mueller(self):
        return self.mueller_partial(1.0)

    def mueller_partial(self, t):
        return ph.linear_retarder(np.radians(self["angle"]), 2 * np.pi * self["retardance"] * t)

    path_options = (
        ("retardance", "Retardance path: θ fixed, Γ from 0 to set value (solid)", True),
        ("angle", "Angle path: Γ fixed, θ from 0° to set value (dashed)", False),
    )

    def trajectories(self, S_in, samples=60):
        ts = np.linspace(0.0, 1.0, samples + 1)
        theta = np.radians(self["angle"])
        delta = 2 * np.pi * self["retardance"]
        out = []
        if self.show_paths["retardance"]:
            out.append(("solid", np.array([ph.linear_retarder(theta, delta * t) @ S_in for t in ts])))
        if self.show_paths["angle"]:
            out.append(("dashed", np.array([ph.linear_retarder(theta * t, delta) @ S_in for t in ts])))
        return out

    def axis(self):
        return self["angle"], "fast axis"

    def describe(self):
        return f"θ = {self['angle']:.1f}°, Γ = {self['retardance']:.3f} λ"


class Rotator(OpticalElement):
    kind = "Optical rotator"
    params = (Param("angle", "Rotation ρ", -180, 180, 45, "°"),)

    def mueller(self):
        return self.mueller_partial(1.0)

    def mueller_partial(self, t):
        return ph.rotator(np.radians(self["angle"]) * t)

    def describe(self):
        return f"ρ = {self['angle']:.1f}°"


class Depolarizer(OpticalElement):
    kind = "Depolarizer"
    params = (Param("amount", "Depolarization", 0, 1, 0.5, "", 2, 0.05),)

    def mueller(self):
        return self.mueller_partial(1.0)

    def mueller_partial(self, t):
        return ph.depolarizer(1 - self["amount"] * t)

    def describe(self):
        return f"{100 * self['amount']:.0f} % depolarizing"


class Mirror(OpticalElement):
    kind = "Mirror"
    params = (
        Param("aoi", "Angle of incidence", 0, 89, 45, "°"),
        Param("n", "Index n", 0.01, 5, 0.13, "", 2, 0.05),
        Param("k", "Extinction k", 0, 10, 3.99, "", 2, 0.05),
        Param("orientation", "Plane of incidence", -90, 90, 0, "°"),
    )

    def mueller(self):
        return ph.mirror(np.radians(self["aoi"]), complex(self["n"], self["k"]),
                         np.radians(self["orientation"]))

    def axis(self):
        return self["orientation"], "plane of incidence"

    def describe(self):
        return f"AOI = {self['aoi']:.1f}°, N = {self['n']:.2f} + {self['k']:.2f}i"


# Class name -> class, used when loading saved setups.
ELEMENT_TYPES = {cls.__name__: cls for cls in (Polarizer, Waveplate, Rotator, Depolarizer, Mirror)}

# Menu entries -> factory.  Defaults for the mirror approximate silver at 633 nm.
ELEMENT_FACTORIES = {
    "Linear polarizer": lambda: Polarizer(),
    "Quarter-wave plate (λ/4)": lambda: Waveplate("λ/4 plate", retardance=0.25),
    "Half-wave plate (λ/2)": lambda: Waveplate("λ/2 plate", retardance=0.5),
    "Waveplate (variable retarder)": lambda: Waveplate("Waveplate", retardance=0.1),
    "Optical rotator": lambda: Rotator(),
    "Depolarizer": lambda: Depolarizer(),
    "Mirror (silver)": lambda: Mirror("Mirror"),
}


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
class OpticalSystem:
    """A light source followed by an ordered chain of optical elements."""

    def __init__(self):
        self.source = LightSource()
        self.elements: list[OpticalElement] = []

    def states(self) -> list[np.ndarray]:
        """Stokes vectors: [source, after element 1, ..., after element n]."""
        S = self.source.stokes()
        out = [S]
        for el in self.elements:
            if el.enabled:
                S = el.mueller() @ S
            out.append(S)
        return out

    def trajectories(self, samples: int = 60) -> list[list[tuple[str, np.ndarray]]]:
        """For each element, the list of (style, Stokes array) curves to draw on the sphere."""
        states = self.states()
        return [el.trajectories(S_in, samples) if el.enabled else []
                for S_in, el in zip(states[:-1], self.elements)]


def default_system() -> OpticalSystem:
    """The setup shown at start-up and after File > New."""
    system = OpticalSystem()
    system.elements = [Waveplate("λ/4 plate", angle=45, retardance=0.25)]
    return system
