"""PyrSim: a small, didactic pyramid wavefront sensor simulator.

This module is intentionally self-contained to support student lab exercises.
It includes:

- Zernike-based phase-screen generation
- Telescope aperture generation with central obstruction and spiders
- A simple pyramid wavefront sensor model using a single prism phase screen
- A detector helper for sampling, displaying and saving images
- A forward simulator that chains all components together
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, sqrt
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DEFAULT_PUPIL_RADIUS = 0.25


def _coordinate_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized x/y coordinate grids in the range [-1, 1]."""
    axis = np.linspace(-1.0, 1.0, size, endpoint=True)
    return np.meshgrid(axis, axis, indexing="xy")


def _noll_to_nm(index: int) -> tuple[int, int]:
    """Convert a 1-based Noll index to (n, m) Zernike indices."""
    if index < 1:
        raise ValueError("Noll indices start at 1")

    # Hardcoded mapping for Noll indices 1 through 11
    noll_mapping = {
        1: (0, 0),    # Piston
        2: (1, 1),    # Tip (X-tilt)
        3: (1, -1),   # Tilt (Y-tilt)
        4: (2, 0),    # Defocus
        5: (2, -2),   # Primary Astigmatism (oblique)
        6: (2, 2),    # Primary Astigmatism (vertical)
        7: (3, -1),   # Primary Coma (vertical)
        8: (3, 1),    # Primary Coma (horizontal)
        9: (3, -3),   # Trefoil (oblique)
        10: (3, 3),   # Trefoil (horizontal)
        11: (4, 0),   # Spherical aberration
    }
    if index in noll_mapping:
        return noll_mapping[index]

    # Fallback to standard Noll mapping formula
    n = 0
    while index > (n + 1) * (n + 2) // 2:
        n += 1
    j0 = index - n * (n + 1) // 2 - 1
    m_values = list(range(-n, n + 1, 2))
    return n, m_values[j0]


def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Compute the radial part R_n^m(rho) for the Zernike polynomial."""
    m = abs(m)
    if (n - m) % 2:
        return np.zeros_like(rho)

    radial = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coeff = (
            ((-1) ** k)
            * factorial(n - k)
            / (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))
        )
        radial = radial + coeff * rho ** (n - 2 * k)
    return radial


def zernike_mode(index: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Return the Zernike mode (Noll convention, 1-based index)."""
    n, m = _noll_to_nm(index)
    radial = _zernike_radial(n, m, rho)

    if m == 0:
        mode = radial
    elif m > 0:
        mode = radial * np.cos(m * theta)
    else:
        mode = radial * np.sin(abs(m) * theta)

    # Unit RMS normalization on the unit disk in the common AO convention.
    norm = sqrt(n + 1) if m == 0 else sqrt(2 * (n + 1))
    return norm * mode


def generate_phase_screen_zernike(
    size: int,
    coefficients: Sequence[float] | dict[int, float],
    pupil_radius: float = DEFAULT_PUPIL_RADIUS,
) -> np.ndarray:
    """Generate a phase screen from Zernike coefficients (in radians)."""
    if not (0.0 < pupil_radius <= 1.0):
        raise ValueError("pupil_radius must be in the interval (0, 1]")

    x, y = _coordinate_grid(size)
    rho = np.sqrt(x * x + y * y) / pupil_radius
    theta = np.arctan2(y, x)
    pupil = rho <= 1.0

    phase = np.zeros((size, size), dtype=float)

    if isinstance(coefficients, dict):
        items = coefficients.items()
    else:
        items = ((i + 1, amp) for i, amp in enumerate(coefficients))

    for index, amplitude in items:
        phase += float(amplitude) * zernike_mode(int(index), rho, theta)

    phase[~pupil] = 0.0
    return phase


def generate_telescope_aperture(
    size: int,
    secondary_obstruction_ratio: float = 0.0,
    spider_width_ratio: float = 0.0,
    spider_angles_deg: Iterable[float] = (0.0, 90.0),
    pupil_radius: float = DEFAULT_PUPIL_RADIUS,
) -> np.ndarray:
    """Generate a binary aperture with optional central obscuration and spiders."""
    if not (0.0 <= secondary_obstruction_ratio < 1.0):
        raise ValueError("secondary_obstruction_ratio must be in [0, 1)")
    if spider_width_ratio < 0.0:
        raise ValueError("spider_width_ratio must be >= 0")
    if not (0.0 < pupil_radius <= 1.0):
        raise ValueError("pupil_radius must be in the interval (0, 1]")

    x, y = _coordinate_grid(size)
    rho = np.sqrt(x * x + y * y) / pupil_radius

    aperture = rho <= 1.0

    if secondary_obstruction_ratio > 0.0:
        aperture &= rho >= secondary_obstruction_ratio

    if spider_width_ratio > 0.0:
        half_width = spider_width_ratio * pupil_radius
        for angle_deg in spider_angles_deg:
            angle = np.deg2rad(float(angle_deg))
            distance = np.abs(-np.sin(angle) * x + np.cos(angle) * y)
            aperture &= distance >= half_width

    return aperture.astype(float)


def generate_pyramid_phase_screen(size: int, slope: float = 8.0 * np.pi) -> np.ndarray:
    """Represent a 4-face pyramid prism as one phase screen in the focal plane."""
    x, y = _coordinate_grid(size)
    return slope * (np.abs(x) + np.abs(y))


def simulate_pyramid_sensor(complex_pupil: np.ndarray, pyramid_phase: np.ndarray) -> np.ndarray:
    """Propagate through a pyramid WFS model and return pupil-plane intensity."""
    focal_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(complex_pupil), norm="ortho"))
    prism = np.exp(1j * pyramid_phase)
    modulated_field = focal_field * prism
    detector_plane = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(modulated_field), norm="ortho"))
    return np.abs(detector_plane) ** 2


@dataclass
class Detector:
    """Simple detector model with optional integer binning."""

    binning: int = 1
    shape: tuple[int, int] | None = None

    def sample(self, image: np.ndarray) -> np.ndarray:
        """Sample an image with optional integer binning and custom shape cropping."""
        if self.binning > 1:
            h, w = image.shape
            bh = h // self.binning
            bw = w // self.binning
            trimmed = image[: bh * self.binning, : bw * self.binning]
            image = trimmed.reshape(bh, self.binning, bw, self.binning).mean(axis=(1, 3))

        if self.shape is not None:
            sh, sw = self.shape
            h, w = image.shape
            # If the simulated image is smaller than the detector, center-pad it with zeros
            pad_y = max(0, sh - h)
            pad_x = max(0, sw - w)
            if pad_y > 0 or pad_x > 0:
                image = np.pad(
                    image,
                    ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)),
                    mode="constant",
                )
                h, w = image.shape

            # Center crop
            y_start = h // 2 - sh // 2
            x_start = w // 2 - sw // 2
            image = image[y_start : y_start + sh, x_start : x_start + sw]

        return image

    def display(self, image: np.ndarray, cmap: str = "viridis") -> None:
        """Display an image using matplotlib (if installed)."""
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(image, origin="lower", cmap=cmap)
        plt.colorbar(label="Intensity")
        plt.title("Detector image")
        plt.tight_layout()
        plt.show()

    def save(self, image: np.ndarray, path: str | Path) -> None:
        """Save image data to disk as NumPy .npy format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, image)


def forward_simulate(
    size: int,
    zernike_coefficients: Sequence[float] | dict[int, float],
    secondary_obstruction_ratio: float = 0.0,
    spider_width_ratio: float = 0.0,
    spider_angles_deg: Iterable[float] = (0.0, 90.0),
    pyramid_slope: float = 8.0 * np.pi,
    detector: Detector | None = None,
    pupil_radius: float = DEFAULT_PUPIL_RADIUS,
    detector_shape: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Run a full forward simulation from phase screen to detector image."""
    phase = generate_phase_screen_zernike(
        size=size,
        coefficients=zernike_coefficients,
        pupil_radius=pupil_radius,
    )
    aperture = generate_telescope_aperture(
        size=size,
        secondary_obstruction_ratio=secondary_obstruction_ratio,
        spider_width_ratio=spider_width_ratio,
        spider_angles_deg=spider_angles_deg,
        pupil_radius=pupil_radius,
    )
    complex_pupil = aperture * np.exp(1j * phase)
    # Scale the pyramid slope to keep the pupil separation scale-invariant.
    # The default slope (8 * pi) is calibrated for size=64 and pupil_radius=0.25.
    scaled_slope = pyramid_slope * (size / 64.0) * (pupil_radius / 0.25)
    pyramid_phase = generate_pyramid_phase_screen(size=size, slope=scaled_slope)
    sensor_intensity = simulate_pyramid_sensor(complex_pupil=complex_pupil, pyramid_phase=pyramid_phase)

    if detector is None:
        detector = Detector(shape=detector_shape)
    detector_image = detector.sample(sensor_intensity)

    return {
        "phase_screen": phase,
        "aperture": aperture,
        "pyramid_phase": pyramid_phase,
        "sensor_intensity": sensor_intensity,
        "detector_image": detector_image,
    }
