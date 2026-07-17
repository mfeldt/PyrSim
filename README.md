# PyrSim
Simple, student-friendly pyramid WFS simulation.

`pyrsim` is an educational package that provides:

- Zernike-based phase-screen generation
- Telescope aperture generation with central obstruction and spiders
- Pyramid wavefront sensor simulation using one prism phase screen
- A detector module for sampling, display, and saving images
- A forward simulator chaining all elements together

## Installation

You can install `pyrsim` directly from the directory:

### For Users
To install the package and its dependencies:
```bash
pip install .
```

### For Developers / Classroom Labs
To install the package in editable mode (so changes to the source code are reflected immediately without reinstalling):
```bash
pip install -e .
```

## Basic Usage

Here is a simple example of how to use the library to set up a simulation with point sources:

```python
import pyrsim

# 1. Initialize the simulator
sim = pyrsim.TelescopeSimulator(
    size=256,
    pupil_radius=0.375,
    detector_shape=(224, 224),
    detector_fov_arcsec=2.2,
)

# 2. Define sky sources as a list of continuous point sources:
# Each source is a tuple: (x_offset_arcsec, y_offset_arcsec, brightness)
sim.set_stars([
    (0.0, 0.0, 1.0),     # Bright central star
    (0.3, -0.2, 0.5),    # Fainter companion star
])

# 3. Simulate pointing the telescope at (x, y) coordinates on the sky (in arcseconds)
img = sim.get_detector_image(x_pointing_arcsec=0.0, y_pointing_arcsec=0.0)

# img is a 2D numpy array of shape (224, 224) representing the camera sensor intensity
```

## Examples

Run the included example script to test the installation and visualize the results:
```bash
python3 examples/example.py
```

## Running Tests

To run the unit tests, use:
```bash
python3 -m unittest discover -s tests
```
