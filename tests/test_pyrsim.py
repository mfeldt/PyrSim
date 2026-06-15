import unittest

import numpy as np

import pyrsim


class PyrSimTests(unittest.TestCase):
    def test_phase_screen_shape_and_mask(self):
        phase = pyrsim.generate_phase_screen_zernike(64, {2: 0.1, 3: -0.2})
        self.assertEqual(phase.shape, (64, 64))
        self.assertTrue(np.isfinite(phase).all())

    def test_aperture_has_obscuration_and_spiders(self):
        aperture = pyrsim.generate_telescope_aperture(
            128,
            secondary_obstruction_ratio=0.2,
            spider_width_ratio=0.03,
            spider_angles_deg=(0, 90),
        )
        self.assertEqual(aperture.shape, (128, 128))
        self.assertLess(aperture.sum(), 128 * 128)
        self.assertGreater(aperture.sum(), 0)

    def test_forward_simulate_outputs(self):
        result = pyrsim.forward_simulate(
            size=64,
            zernike_coefficients={4: 0.2, 5: 0.1},
            secondary_obstruction_ratio=0.15,
            spider_width_ratio=0.02,
        )

        for key in ("phase_screen", "aperture", "pyramid_phase", "sensor_intensity", "detector_image"):
            self.assertIn(key, result)
            self.assertTrue(np.isfinite(result[key]).all())


if __name__ == "__main__":
    unittest.main()
