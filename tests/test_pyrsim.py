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

    def test_noll_to_nm_conversion(self):
        expected = {
            1: (0, 0),
            2: (1, 1),
            3: (1, -1),
            4: (2, 0),
            5: (2, -2),
            6: (2, 2),
            7: (3, -1),
            8: (3, 1),
            9: (3, -3),
            10: (3, 3),
            11: (4, 0),
        }
        for j, nm in expected.items():
            self.assertEqual(pyrsim._noll_to_nm(j), nm, f"Noll index {j} conversion failed")

    def test_spider_width_scaling(self):
        size = 100
        spider_width_ratio = 0.1
        y_coords = np.linspace(-1.0, 1.0, size, endpoint=True)
        for pupil_radius in [0.2, 0.4]:
            aperture = pyrsim.generate_telescope_aperture(
                size=size,
                spider_width_ratio=spider_width_ratio,
                spider_angles_deg=(0,),
                pupil_radius=pupil_radius,
            )
            column = aperture[:, size // 2]
            in_pupil_mask = np.abs(y_coords) <= pupil_radius
            spider_masked = (column == 0) & in_pupil_mask
            if spider_masked.any():
                masked_y = y_coords[spider_masked]
                span = masked_y.max() - masked_y.min()
                expected_span = 2.0 * spider_width_ratio * pupil_radius
                pixel_spacing = 2.0 / (size - 1)
                self.assertAlmostEqual(span, expected_span, delta=1.5 * pixel_spacing)

    def test_scale_invariant_pupil_separation(self):
        pupil_radius = 0.25
        result_64 = pyrsim.forward_simulate(
            size=64,
            zernike_coefficients={},
            pupil_radius=pupil_radius,
        )
        result_128 = pyrsim.forward_simulate(
            size=128,
            zernike_coefficients={},
            pupil_radius=pupil_radius,
        )
        phase_64 = result_64["pyramid_phase"]
        phase_128 = result_128["pyramid_phase"]
        self.assertAlmostEqual(phase_128.max(), 2.0 * phase_64.max(), places=5)


if __name__ == "__main__":
    unittest.main()
