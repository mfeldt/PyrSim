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

    def test_custom_system_configuration(self):
        # Grid size N = 256.
        # Pupil diameter = 96px => pupil_radius = 48px => 48 / 128 = 0.375 in coordinates
        # Separation = 32px gap => shift = 48 + 16 = 64px => pyramid_slope = 32/3 * pi
        # Detector shape = 224x224
        size = 256
        pupil_radius = 96.0 / size
        pyramid_slope = (32.0 / 3.0) * np.pi
        detector_shape = (224, 224)
        
        result = pyrsim.forward_simulate(
            size=size,
            zernike_coefficients={},
            pupil_radius=pupil_radius,
            pyramid_slope=pyramid_slope,
            detector_shape=detector_shape,
        )
        img = result["detector_image"]
        
        # Verify shape
        self.assertEqual(img.shape, (224, 224))
        
        # Find peak positions in the four quadrants of the cropped detector image
        # Center of 224x224 image is at x = 112, y = 112.
        # Expected shift is 64 pixels.
        # Expected centers: x in [48, 176], y in [48, 176].
        half_y, half_x = 112, 112
        
        quadrants = {
            "Top-Right": (img[half_y:, half_x:], half_x, half_y),
            "Top-Left": (img[half_y:, :half_x], 0, half_y),
            "Bottom-Left": (img[:half_y, :half_x], 0, 0),
            "Bottom-Right": (img[:half_y, half_x:], half_x, 0),
        }
        
        for name, (quad_img, offset_x, offset_y) in quadrants.items():
            y_idx, x_idx = np.unravel_index(np.argmax(quad_img), quad_img.shape)
            x_glob = x_idx + offset_x
            y_glob = y_idx + offset_y
            
            # Distance of peaks from the center should be close to 64 pixels on each axis
            # (allowing for knife-edge peak shifting effects, but definitely in the correct quadrant)
            x_rel = abs(x_glob - half_x)
            y_rel = abs(y_glob - half_y)
            
            self.assertGreater(x_rel, 20)  # Verify they are shifted away from center
            self.assertGreater(y_rel, 20)
            self.assertLess(x_rel, 96)     # Verify they don't exceed the boundaries
            self.assertLess(y_rel, 96)


if __name__ == "__main__":
    unittest.main()
