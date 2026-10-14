"""Core polarization physics: Stokes vectors, Jones and Mueller calculus.

This module has no GUI dependencies so it can be tested and reused on its own.

Conventions
-----------
* The beam travels along +z and the field is written as
  E(z, t) = Re{ (Ex, Ey) * exp[i(kz - wt)] }.
* Stokes parameters:
      S0 = |Ex|^2 + |Ey|^2        S1 = |Ex|^2 - |Ey|^2
      S2 = 2 Re(Ex* Ey)           S3 = 2 Im(Ex* Ey)
* With these choices S3 > 0 means the field vector rotates counter-clockwise
  (from +x towards +y) for an observer looking into the beam, i.e. facing the
  source.  The GUI uses the words "CCW"/"CW" to avoid the left/right-handed
  ambiguity that differs between textbooks.
* All angles are in radians here (the GUI layer works in degrees).
* Azimuth psi is measured from +x towards +y; ellipticity angle chi is in
  [-pi/4, pi/4] with chi > 0 for counter-clockwise rotation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Pauli-like basis matching the Stokes definition above: S_k = E^dagger sigma_k E
PAULI = np.array(
    [
        [[1, 0], [0, 1]],
        [[1, 0], [0, -1]],
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
    ],
    dtype=complex,
)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Stokes vectors
# ---------------------------------------------------------------------------
def stokes_from_jones(E) -> np.ndarray:
    """Stokes vector of a (fully polarized) Jones vector."""
    E = np.asarray(E, dtype=complex)
    return np.real(np.einsum("i,kij,j->k", E.conj(), PAULI, E))


def jones_from_stokes(S) -> np.ndarray:
    """Jones vector of the polarized part of S (global phase chosen so Ex is real)."""
    S = np.asarray(S, dtype=float)
    ip = float(np.linalg.norm(S[1:]))
    ax = np.sqrt(max(ip + S[1], 0.0) / 2.0)
    ay = np.sqrt(max(ip - S[1], 0.0) / 2.0)
    phi = np.arctan2(S[3], S[2])
    return np.array([ax, ay * np.exp(1j * phi)], dtype=complex)


def stokes_from_ellipse(azimuth: float, ellipticity: float, dop: float = 1.0,
                        intensity: float = 1.0) -> np.ndarray:
    """Stokes vector from ellipse azimuth psi, ellipticity angle chi and degree of polarization."""
    c = np.cos(2 * ellipticity)
    return intensity * np.array([
        1.0,
        dop * c * np.cos(2 * azimuth),
        dop * c * np.sin(2 * azimuth),
        dop * np.sin(2 * ellipticity),
    ])


@dataclass(frozen=True)
class PolarizationState:
    """Human-friendly description of a Stokes vector."""
    intensity: float            # S0
    polarized_intensity: float  # sqrt(S1^2 + S2^2 + S3^2)
    dop: float                  # degree of polarization, 0..1
    azimuth: float              # psi, radians, (-pi/2, pi/2]
    ellipticity: float          # chi, radians, [-pi/4, pi/4]
    direction: np.ndarray       # unit vector (s1, s2, s3) on the sphere (zeros if unpolarized)

    @property
    def has_light(self) -> bool:
        return self.intensity > _EPS

    @property
    def rotation(self) -> str:
        if self.polarized_intensity <= 1e-9 or abs(np.sin(2 * self.ellipticity)) < 1e-6:
            return "linear" if self.polarized_intensity > 1e-9 else "none"
        return "CCW" if self.ellipticity > 0 else "CW"


def describe(S) -> PolarizationState:
    S = np.asarray(S, dtype=float)
    s0 = float(S[0])
    ip = float(np.linalg.norm(S[1:]))
    dop = min(ip / s0, 1.0) if s0 > _EPS else 0.0
    if ip > _EPS:
        direction = S[1:] / ip
        azimuth = 0.5 * np.arctan2(S[2], S[1])
        ellipticity = 0.5 * np.arcsin(np.clip(S[3] / ip, -1.0, 1.0))
    else:
        direction = np.zeros(3)
        azimuth = ellipticity = 0.0
    return PolarizationState(s0, ip, dop, float(azimuth), float(ellipticity), direction)


# ---------------------------------------------------------------------------
# Jones <-> Mueller
# ---------------------------------------------------------------------------
def jones_to_mueller(J) -> np.ndarray:
    """Mueller matrix of a (non-depolarizing) Jones matrix: M_ij = 1/2 tr(sigma_i J sigma_j J^dagger)."""
    J = np.asarray(J, dtype=complex)
    Jh = J.conj().T
    M = np.empty((4, 4))
    for i in range(4):
        for j in range(4):
            M[i, j] = 0.5 * np.real(np.trace(PAULI[i] @ J @ PAULI[j] @ Jh))
    return M


def jones_rotation(theta: float) -> np.ndarray:
    """Projects lab-frame field components onto a frame whose x-axis is at angle theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]], dtype=complex)


# ---------------------------------------------------------------------------
# Mueller matrices of optical elements
# ---------------------------------------------------------------------------
def rotation_mueller(theta: float) -> np.ndarray:
    """Mueller matrix of a rotation of the reference frame by theta."""
    c, s = np.cos(2 * theta), np.sin(2 * theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, s, 0],
        [0, -s, c, 0],
        [0, 0, 0, 1],
    ], dtype=float)


def rotate_element(M, theta: float) -> np.ndarray:
    """Mueller matrix of element M physically rotated by theta about the beam axis."""
    return rotation_mueller(-theta) @ np.asarray(M, float) @ rotation_mueller(theta)


def linear_polarizer(theta: float) -> np.ndarray:
    """Ideal linear polarizer with transmission axis at angle theta."""
    c, s = np.cos(2 * theta), np.sin(2 * theta)
    return 0.5 * np.array([
        [1, c, s, 0],
        [c, c * c, c * s, 0],
        [s, c * s, s * s, 0],
        [0, 0, 0, 0],
    ], dtype=float)


def linear_retarder(theta: float, delta: float) -> np.ndarray:
    """Linear retarder (waveplate) with fast axis at angle theta and retardance delta (radians).

    On the Poincare sphere this is a rotation by +delta (right-hand rule) about the
    axis (cos 2theta, sin 2theta, 0) that represents the fast-axis polarization.
    """
    c, s = np.cos(2 * theta), np.sin(2 * theta)
    cd, sd = np.cos(delta), np.sin(delta)
    return np.array([
        [1, 0, 0, 0],
        [0, c * c + s * s * cd, c * s * (1 - cd), s * sd],
        [0, c * s * (1 - cd), s * s + c * c * cd, -c * sd],
        [0, -s * sd, c * sd, cd],
    ], dtype=float)


def rotator(rho: float) -> np.ndarray:
    """Optical rotator: rotates the polarization ellipse by rho (from +x towards +y)."""
    return rotation_mueller(-rho)


def depolarizer(retention: float) -> np.ndarray:
    """Isotropic partial depolarizer; retention = 1 keeps the state, 0 fully depolarizes."""
    a = float(retention)
    return np.diag([1.0, a, a, a])


def fresnel_coefficients(aoi: float, n2: complex, n1: float = 1.0):
    """Fresnel amplitude reflection coefficients (rs, rp) for incidence from medium n1 onto n2.

    Sign convention: rp = -rs at normal incidence, so an ideal mirror flips the
    handedness (S2, S3 -> -S2, -S3), as is standard for reflection.
    """
    n2 = complex(n2)
    cos_i = np.cos(aoi)
    sin_t = n1 * np.sin(aoi) / n2
    cos_t = np.sqrt(1 - sin_t ** 2 + 0j)
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    return rs, rp


def mirror(aoi: float, n2: complex, orientation: float = 0.0) -> np.ndarray:
    """Reflection from a (possibly absorbing) surface with complex index n2 = n + ik.

    orientation is the angle of the plane of incidence measured from +x.
    """
    rs, rp = fresnel_coefficients(aoi, n2)
    M0 = jones_to_mueller(np.diag([rp, rs]))
    return rotate_element(M0, orientation)


def is_physical(S, tol: float = 1e-9) -> bool:
    """True if S0 >= sqrt(S1^2 + S2^2 + S3^2) (within tolerance)."""
    S = np.asarray(S, float)
    return S[0] + tol >= np.linalg.norm(S[1:])
