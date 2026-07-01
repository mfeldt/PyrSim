import pyrsim
import numpy as np
import matplotlib.pyplot as plt

# 1. Initialize the Telescope Simulator with your hardware layout
sim = pyrsim.TelescopeSimulator(
    size=256,                             # Square grid size for FFT propagation
    pupil_radius=96.0 / 256,              # Calibrated for 96px pupil diameter (0.375)
    pyramid_slope=(32.0 / 3.0) * np.pi,   # Calibrated for 32px gap (S = 64px center shift)
    detector_shape=(224, 224),            # (height, width) of your camera sensor (shows full 224x224 pupil pattern)
    detector_fov_arcsec=2.2,              # Field of view of the detector grid
    modulation_amplitude_arcsec=0.2,      # Circular modulation amplitude in arcseconds
    modulation_steps=8,                   # Sampling steps per revolution (1 frame integration, 8 is fast and accurate)
)

# 2. Generate a 2D night sky map image separately
sky_fov = 10.0
sky_pixels = 100

# Define a Gaussian-profile star centered at continuous coordinates:
# Star coordinates on sky: (x = -0.4, y = 0.2) arcseconds
star_x_arcsec = -0.4
star_y_arcsec = 0.2
fwhm_arcsec = 0.3
sigma_arcsec = fwhm_arcsec / 2.355

# Generate physical coordinate grids matching pyrsim's mapping
x_axis = np.linspace(-sky_fov / 2, sky_fov / 2, sky_pixels)
y_axis = np.linspace(-sky_fov / 2, sky_fov / 2, sky_pixels)
xx_sky, yy_sky = np.meshgrid(x_axis, y_axis, indexing="xy")

# Calculate distance and generate Gaussian sky map directly in arcseconds
dist_sq = (xx_sky - star_x_arcsec)**2 + (yy_sky - star_y_arcsec)**2
sky_map = np.exp(-dist_sq / (2 * sigma_arcsec**2))

# Load this sky map into the simulator
# Using threshold=0.05 ignores low-intensity tails, speeding up runs by only simulating core active pixels.
sim.set_sky_map(sky_map, sky_fov_arcsec=sky_fov, brightness_threshold=0.05)

# Show the sky map we generated
plt.figure()
plt.imshow(sky_map, extent=[-sky_fov/2, sky_fov/2, -sky_fov/2, sky_fov/2])
plt.title(f"Sky Map: Gaussian Star (FWHM = {fwhm_arcsec} arcsec)")
plt.xlabel("X (arcsec)")
plt.ylabel("Y (arcsec)")
plt.colorbar(label="Intensity")
plt.show()

# 3. Simulate pointing the telescope at different coordinates on the sky (in arcseconds)
# Point directly at the star:
img_pointing_star = sim.get_detector_image(x_pointing_arcsec=star_x_arcsec, y_pointing_arcsec=star_y_arcsec)

# Point slightly off-center:
img_pointing_off = sim.get_detector_image(x_pointing_arcsec=star_x_arcsec + 0.1, y_pointing_arcsec=star_y_arcsec + 0.1)

# 4. Plot the detector results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img_pointing_star, origin="lower")
axes[0].set_title(f"Pointed at Star Center ({star_x_arcsec:+.2f}, {star_y_arcsec:+.2f})")
axes[0].set_xlabel("Pixels")
axes[0].set_ylabel("Pixels")

axes[1].imshow(img_pointing_off, origin="lower")
axes[1].set_title("Pointed 0.1 arcsec Off-Center")
axes[1].set_xlabel("Pixels")

plt.tight_layout()
plt.show()
