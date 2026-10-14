import json

import numpy as np
import pytest

from polviz.elements import Depolarizer, Mirror, OpticalSystem, Polarizer, Waveplate, default_system
from polviz.serialization import load_setup, save_setup, system_from_dict, system_to_dict

H = np.array([1.0, 1, 0, 0])


def styles(curves):
    return [style for style, _ in curves]


def test_waveplate_path_toggles():
    wp = Waveplate(angle=30, retardance=0.25)
    assert styles(wp.trajectories(H)) == ["solid"]          # default: retardance path only
    wp.show_paths["angle"] = True
    assert styles(wp.trajectories(H)) == ["solid", "dashed"]
    wp.show_paths["retardance"] = False
    assert styles(wp.trajectories(H)) == ["dashed"]
    wp.show_paths["angle"] = False
    assert wp.trajectories(H) == []


def test_both_paths_end_at_output():
    wp = Waveplate(angle=-37, retardance=0.31)
    wp.show_paths["angle"] = True
    out = wp.mueller() @ H
    for _, path in wp.trajectories(H):
        assert np.allclose(path[-1], out)


def test_angle_path_starts_at_theta_zero_output():
    wp = Waveplate(angle=50, retardance=0.2)
    wp.show_paths["angle"] = True
    dashed = wp.trajectories(H)[1][1]
    assert np.allclose(dashed[0], Waveplate(angle=0, retardance=0.2).mueller() @ H)


def test_half_wave_plate_angle_path_stays_on_equator():
    wp = Waveplate(angle=60, retardance=0.5)
    wp.show_paths["angle"] = True
    for S in wp.trajectories(np.array([1.0, 0.6, 0.8, 0]))[1][1]:
        assert abs(S[3]) < 1e-12


def test_other_elements_keep_single_solid_path():
    for el in (Polarizer(), Depolarizer(), Mirror()):
        assert styles(el.trajectories(H)) == ["solid"]


def test_serialization_roundtrip(tmp_path):
    system = OpticalSystem()
    system.source.set("azimuth", 30)
    system.source.set("dop", 0.8)
    wp = Waveplate("λ/2 plate", angle=12.5, retardance=0.5)
    wp.show_paths.update(retardance=False, angle=True)
    mirror = Mirror(aoi=60)
    mirror.enabled = False
    system.elements = [wp, Polarizer(angle=-20), mirror]

    path = tmp_path / "setup.json"
    save_setup(system, path)
    loaded = load_setup(path)
    assert system_to_dict(loaded) == system_to_dict(system)
    assert loaded.elements[0].show_paths == {"retardance": False, "angle": True}
    assert not loaded.elements[2].enabled
    for a, b in zip(loaded.states(), system.states()):
        assert np.allclose(a, b)


def test_missing_values_use_defaults():
    data = system_to_dict(default_system())
    del data["elements"][0]["values"]["retardance"]
    del data["elements"][0]["show_paths"]
    loaded = system_from_dict(data)
    assert loaded.elements[0]["retardance"] == Waveplate.params[1].default
    assert loaded.elements[0].show_paths == {"retardance": True, "angle": False}


@pytest.mark.parametrize("data, message", [
    ({"hello": 1}, "not a PolViz"),
    ({"format": "polviz-setup", "version": 99}, "newer"),
    ({"format": "polviz-setup", "version": 1, "elements": [{"type": "Laser"}]}, "unknown type"),
])
def test_invalid_files(data, message):
    with pytest.raises(ValueError, match=message):
        system_from_dict(data)


def test_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_setup(bad)
    good = tmp_path / "good.json"
    good.write_text(json.dumps(system_to_dict(default_system())))
    assert len(load_setup(good).elements) == 1
