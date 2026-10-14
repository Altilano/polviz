import numpy as np
import pytest

from polviz import physics as ph
from polviz.elements import ELEMENT_FACTORIES, OpticalSystem, Polarizer, Waveplate

rng = np.random.default_rng(0)
H = np.array([1.0, 1, 0, 0])
D = np.array([1.0, 0, 1, 0])


def random_stokes(n=20):
    for _ in range(n):
        v = rng.normal(size=3)
        yield np.concatenate([[1.0], rng.uniform(0, 1) * v / np.linalg.norm(v)])


def test_stokes_jones_roundtrip():
    for S in random_stokes():
        S_pol = np.concatenate([[np.linalg.norm(S[1:])], S[1:]])
        assert np.allclose(ph.stokes_from_jones(ph.jones_from_stokes(S_pol)), S_pol)


def test_ccw_convention():
    # Ex = 1, Ey = i  ->  field goes x -> y (CCW facing the source) -> S3 = +1
    assert np.allclose(ph.stokes_from_jones([1 / np.sqrt(2), 1j / np.sqrt(2)]), [1, 0, 0, 1])


@pytest.mark.parametrize("theta", np.linspace(-np.pi / 2, np.pi / 2, 7))
def test_polarizer_matches_jones(theta):
    R = ph.jones_rotation(theta)
    J = R.conj().T @ np.diag([1, 0]) @ R
    assert np.allclose(ph.linear_polarizer(theta), ph.jones_to_mueller(J))
    assert np.allclose(ph.rotate_element(ph.linear_polarizer(0), theta), ph.linear_polarizer(theta))


@pytest.mark.parametrize("theta,delta", [(t, d) for t in (0, 0.3, -1.1) for d in (0.5, np.pi / 2, 2.7)])
def test_retarder_matches_jones(theta, delta):
    R = ph.jones_rotation(theta)
    J = R.conj().T @ np.diag([1, np.exp(1j * delta)]) @ R  # slow axis (y') delayed
    assert np.allclose(ph.linear_retarder(theta, delta), ph.jones_to_mueller(J))


def test_quarter_wave_plate_makes_circular():
    out = ph.linear_retarder(0, np.pi / 2) @ D
    assert np.allclose(out, [1, 0, 0, 1])


def test_half_wave_plate_rotates_linear():
    out = ph.linear_retarder(np.radians(22.5), np.pi) @ H
    assert np.allclose(out, D)


def test_malus_law():
    for deg in (0, 30, 60, 90):
        out = ph.linear_polarizer(np.radians(deg)) @ H
        assert np.isclose(out[0], np.cos(np.radians(deg)) ** 2)


def test_rotator():
    assert np.allclose(ph.rotator(np.pi / 4) @ H, D)


def test_mirror_normal_incidence_flips_handedness():
    M = ph.mirror(0.0, 1e6j)  # nearly perfect conductor
    assert np.allclose(M, np.diag([1, 1, -1, -1]), atol=1e-5)


def test_depolarizer():
    out = ph.depolarizer(0.25) @ H
    assert np.isclose(ph.describe(out).dop, 0.25)


def test_ellipse_roundtrip():
    S = ph.stokes_from_ellipse(0.4, -0.3, 0.7, 2.0)
    st = ph.describe(S)
    assert np.isclose(st.azimuth, 0.4) and np.isclose(st.ellipticity, -0.3)
    assert np.isclose(st.dop, 0.7) and np.isclose(st.intensity, 2.0)


@pytest.mark.parametrize("factory", list(ELEMENT_FACTORIES.values()))
def test_elements_are_physical(factory):
    el = factory()
    for S in random_stokes():
        assert ph.is_physical(el.mueller() @ S)
        assert np.allclose(el.mueller_partial(1.0), el.mueller())
        assert np.allclose(el.mueller_partial(0.0), np.eye(4))


def test_system_chain_and_disable():
    sys_ = OpticalSystem()
    sys_.source.set("azimuth", 45)
    sys_.elements = [Waveplate(angle=0, retardance=0.25), Polarizer(angle=90)]
    states = sys_.states()
    assert np.allclose(states[1], [1, 0, 0, 1])
    assert np.isclose(states[2][0], 0.5)
    sys_.elements[1].enabled = False
    assert np.allclose(sys_.states()[2], states[1])
    assert len(sys_.paths()) == 2
