"""Save and load optical setups as JSON.

File layout (version 1)::

    {
      "format": "polviz-setup",
      "version": 1,
      "source": {"azimuth": 0.0, "ellipticity": 0.0, "dop": 1.0},
      "elements": [
        {"type": "Waveplate", "name": "λ/4 plate", "enabled": true,
         "values": {"angle": 45.0, "retardance": 0.25},
         "show_paths": {"retardance": true, "angle": false}}
      ]
    }

Parameters use the same units as the GUI (degrees, waves).  Missing values
fall back to defaults, so files stay loadable when new parameters are added.
"""
from __future__ import annotations

import json
from pathlib import Path

from .elements import ELEMENT_TYPES, OpticalSystem

FORMAT = "polviz-setup"
VERSION = 1


def system_to_dict(system: OpticalSystem) -> dict:
    return {
        "format": FORMAT,
        "version": VERSION,
        "source": dict(system.source.values),
        "elements": [
            {
                "type": type(el).__name__,
                "name": el.name,
                "enabled": el.enabled,
                "values": dict(el.values),
                **({"show_paths": dict(el.show_paths)} if el.show_paths else {}),
            }
            for el in system.elements
        ],
    }


def system_from_dict(data: dict) -> OpticalSystem:
    """Build an OpticalSystem from a dict; raises ValueError with a readable message if invalid."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("This is not a PolViz setup file.")
    if data.get("version", 0) > VERSION:
        raise ValueError(f"The file was saved by a newer PolViz (format version {data['version']}).")

    system = OpticalSystem()
    for key, value in data.get("source", {}).items():
        if key in system.source.values:
            system.source.set(key, value)

    for i, item in enumerate(data.get("elements", []), start=1):
        cls = ELEMENT_TYPES.get(item.get("type"))
        if cls is None:
            raise ValueError(f"Element {i}: unknown type {item.get('type')!r}.")
        known = {k: v for k, v in item.get("values", {}).items() if k in {p.key for p in cls.params}}
        try:
            el = cls(item.get("name"), **known)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Element {i} ({item.get('type')}): {exc}") from exc
        el.enabled = bool(item.get("enabled", True))
        for key, shown in item.get("show_paths", {}).items():
            if key in el.show_paths:
                el.show_paths[key] = bool(shown)
        system.elements.append(el)
    return system


def save_setup(system: OpticalSystem, path) -> None:
    text = json.dumps(system_to_dict(system), indent=2, ensure_ascii=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def load_setup(path) -> OpticalSystem:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    return system_from_dict(data)
