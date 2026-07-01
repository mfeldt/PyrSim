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

    n = 0
    while index > (n + 1) * (n + 2) // 2:
        n += 1

    start_j = n * (n + 1) // 2 + 1
    offset = index - start_j

    if n == 0:
        return 0, 0

    if n % 2 == 0:
        if offset == 0:
            return n, 0
        abs_m = 2 * ((offset - 1) // 2 + 1)
    else:
        abs_m = 2 * (offset // 2) + 1

    m = abs_m if index % 2 == 0 else -abs_m
    return n, m


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
    """Generate a phase screen from Zernike coefficients (in radians).

    Parameters
    ----------
    size:
        Number of pixels on each axis of the square phase screen.
    coefficients:
        Either a sequence where element i corresponds to Noll index i+1,
        or a dictionary mapping 1-based Noll indices to amplitudes.
    pupil_radius:
        Radius of the simulated pupil in normalized grid coordinates. The
        default leaves enough margin in the array to separate the four pupils
        formed by the pyramid sensor.
    """
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
    """Generate a binary aperture with optional central obscuration and spiders.

    Ratios are relative to the telescope pupil diameter. The pupil radius is
    expressed in normalized grid coordinates, where 1.0 would touch the array
    edges and the default keeps the pupil compact enough to form four
    separated pupils after the pyramid sensor.
    """
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
        half_width_coord = spider_width_ratio * pupil_radius
        for angle_deg in spider_angles_deg:
            angle = np.deg2rad(float(angle_deg))
            distance_coord = np.abs(-np.sin(angle) * x + np.cos(angle) * y)
            aperture &= distance_coord >= half_width_coord

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
    """Simple detector model with optional integer binning and custom shapes."""

    binning: int = 1
    shape: tuple[int, int] | None = None  # (height, width)

    def sample(self, image: np.ndarray) -> np.ndarray:
        """Sample an image with optional integer binning and custom shape cropping."""
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

        if self.binning <= 1:
            return image

        h, w = image.shape
        bh = h // self.binning
        bw = w // self.binning
        trimmed = image[: bh * self.binning, : bw * self.binning]
        return trimmed.reshape(bh, self.binning, bw, self.binning).mean(axis=(1, 3))

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


class TelescopeSimulator:
    """Simulator of a telescope with a pyramid wavefront sensor viewing a sky map.

    This class supports loading a 2D map of the night sky (with an associated
    angular field of view in arcseconds) and computing the resulting detector
    image for any telescope pointing coordinates (X, Y).
    """

    def __init__(
        self,
        size: int = 512,
        pupil_radius: float | None = None,
        pyramid_slope: float = (32.0 / 3.0) * np.pi,
        detector_shape: tuple[int, int] = (256, 320),
        detector_fov_arcsec: float = 2.2,
        secondary_obstruction_ratio: float = 0.0,
        spider_width_ratio: float = 0.0,
        spider_angles_deg: Iterable[float] = (0.0, 90.0),
        modulation_amplitude_arcsec: float = 0.0,
        modulation_steps: int = 16,
    ) -> None:
        self.size = size
        # Default pupil_radius is calibrated for the 96px pupil diameter
        self.pupil_radius = pupil_radius if pupil_radius is not None else 96.0 / size
        self.pyramid_slope = pyramid_slope
        self.detector_shape = detector_shape
        self.detector_fov_arcsec = detector_fov_arcsec
        self.secondary_obstruction_ratio = secondary_obstruction_ratio
        self.spider_width_ratio = spider_width_ratio
        self.spider_angles_deg = spider_angles_deg
        self.modulation_amplitude_arcsec = modulation_amplitude_arcsec
        self.modulation_steps = modulation_steps

        # State variable: list of point sources as (x_arcsec, y_arcsec, brightness)
        self.sources: list[tuple[float, float, float]] = []

    def set_sky_map(self, sky_map: np.ndarray, sky_fov_arcsec: float, brightness_threshold: float = 1e-4) -> None:
        """Convert a 2D sky map image to a list of continuous point sources."""
        self.sources = []
        sky_map = np.array(sky_map, dtype=float)
        sh, sw = sky_map.shape
        x_axis = np.linspace(-sky_fov_arcsec / 2, sky_fov_arcsec / 2, sw)
        y_axis = np.linspace(-sky_fov_arcsec / 2, sky_fov_arcsec / 2, sh)

        # Convert non-zero image pixels to continuous coordinates
        y_indices, x_indices = np.where(sky_map > brightness_threshold)
        for y_idx, x_idx in zip(y_indices, x_indices):
            x_arcsec = float(x_axis[x_idx])
            y_arcsec = float(y_axis[y_idx])
            brightness = float(sky_map[y_idx, x_idx])
            self.sources.append((x_arcsec, y_arcsec, brightness))

    def set_stars(self, stars: Iterable[tuple[float, float, float]]) -> None:
        """Set the sky sources as a list of continuous point sources.

        Each star is a tuple: (x_offset_arcsec, y_offset_arcsec, brightness)
        """
        self.sources = [(float(x), float(y), float(b)) for x, y, b in stars]

    def get_detector_image(
        self,
        x_pointing_arcsec: float,
        y_pointing_arcsec: float,
        common_aberrations: Sequence[float] | dict[int, float] | None = None,
    ) -> np.ndarray:
        """Calculate the detector image based on telescope pointing and the sky map.

        This sums the incoherent contributions of all active point sources that fall within
        the telescope's field of view, incorporating modulation if configured.
        """
        if not self.sources:
            return np.zeros(self.detector_shape)

        # Convert sources to a numpy array for fast vectorized filtering
        src_arr = np.array(self.sources)
        xs = src_arr[:, 0]
        ys = src_arr[:, 1]
        brightnesses = src_arr[:, 2]

        # Pre-filter candidates that can potentially enter the detector FOV during any modulation step
        max_mod = self.modulation_amplitude_arcsec
        half_fov_limit = self.detector_fov_arcsec / 2.0 + max_mod
        in_range = (np.abs(xs - x_pointing_arcsec) <= half_fov_limit) & (np.abs(ys - y_pointing_arcsec) <= half_fov_limit)

        active_xs = xs[in_range]
        active_ys = ys[in_range]
        active_brightnesses = brightnesses[in_range]

        if len(active_brightnesses) == 0:
            return np.zeros(self.detector_shape)

        # Determine modulation offsets. If modulation amplitude is 0, we just do one step at (0, 0)
        if self.modulation_amplitude_arcsec > 0.0 and self.modulation_steps > 0:
            angles = np.linspace(0, 2.0 * np.pi, self.modulation_steps, endpoint=False)
            offsets = [
                (self.modulation_amplitude_arcsec * np.cos(theta), self.modulation_amplitude_arcsec * np.sin(theta))
                for theta in angles
            ]
        else:
            offsets = [(0.0, 0.0)]

        total_detector_image = np.zeros(self.detector_shape)

        # Determine base aberrations (Zernike dict format)
        if common_aberrations is None:
            base_coeffs = {}
        elif isinstance(common_aberrations, dict):
            base_coeffs = dict(common_aberrations)
        else:
            base_coeffs = {i + 1: val for i, val in enumerate(common_aberrations)}

        # Loop over the modulation path offsets
        for mod_dx, mod_dy in offsets:
            curr_x_pointing = x_pointing_arcsec + mod_dx
            curr_y_pointing = y_pointing_arcsec + mod_dy

            # Filter candidates for the current pointing
            half_fov = self.detector_fov_arcsec / 2.0
            in_fov = (np.abs(active_xs - curr_x_pointing) <= half_fov) & (np.abs(active_ys - curr_y_pointing) <= half_fov)

            step_xs = active_xs[in_fov]
            step_ys = active_ys[in_fov]
            step_brightnesses = active_brightnesses[in_fov]

            # Loop through active point sources and sum their incoherent intensities
            for x_s, y_s, brightness in zip(step_xs, step_ys, step_brightnesses):
                dx = x_s - curr_x_pointing
                dy = y_s - curr_y_pointing

                # Calculate tip/tilt for this specific star
                pixel_scale = self.detector_fov_arcsec / self.size
                delta_x_pixels = -dx / pixel_scale
                delta_y_pixels = -dy / pixel_scale

                a2 = 0.5 * np.pi * self.pupil_radius * delta_x_pixels
                a3 = 0.5 * np.pi * self.pupil_radius * delta_y_pixels

                # Add to base aberrations
                coeffs = dict(base_coeffs)
                coeffs[2] = coeffs.get(2, 0.0) + a2
                coeffs[3] = coeffs.get(3, 0.0) + a3

                # Run simulation for this point source
                res = forward_simulate(
                    size=self.size,
                    zernike_coefficients=coeffs,
                    secondary_obstruction_ratio=self.secondary_obstruction_ratio,
                    spider_width_ratio=self.spider_width_ratio,
                    spider_angles_deg=self.spider_angles_deg,
                    pyramid_slope=self.pyramid_slope,
                    pupil_radius=self.pupil_radius,
                    detector_shape=self.detector_shape,
                )

                total_detector_image += brightness * res["detector_image"]

        # Average the integrated image over the number of modulation steps
        return total_detector_image / len(offsets)

