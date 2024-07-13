import unittest
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from segment_multiwell_plate.segment_multiwell_plate import _average_d_min, correct_rotations, find_well_centres

from chlamy_impi.database_creation.error_correction import remove_failed_photos, remove_repeated_initial_frame_tif
from chlamy_impi.paths import find_all_tif_images
from chlamy_impi.well_segmentation_preprocessing.main import load_image


class TestRotationCorrection(unittest.TestCase):
    def test_rotation_correction_real_data(self):
        filenames = find_all_tif_images()

        print(f"Found {len(filenames)} tif files")

        for im_path in filenames:
            tif = load_image(im_path)
            tif = remove_failed_photos(tif)
            tif = remove_repeated_initial_frame_tif(tif)
            im = tif[0]

            assert im.shape == (480, 640)
            assert np.std(im) > 1e-6

            blob_log_kwargs = {"threshold": 0.12, "min_sigma": 1, "max_sigma": 3}
            well_coords = find_well_centres(tif, **blob_log_kwargs)

            rotated_im, rotation_angle = correct_rotations(im, well_coords, return_theta=True)

            if abs(rotation_angle) > 0.01:
                fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=200)
                fig.suptitle(f'Rotation: {rotation_angle:.2g} rad')
                axs[0].imshow(im)
                axs[0].scatter(well_coords[:, 1], well_coords[:, 0], s=1, c='red')
                axs[1].imshow(rotated_im)
                plt.show()


    def test_average_d_min(self):
        # Test that the average_d_min function works as expected
        # This is just a copy of the test from segment_multiwell_plate, to verify that the import works correctly
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 5, 6)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.flatten(), Y.flatten()], axis=-1)

        # Test that the average distance between points is correct
        d_min = _average_d_min(points)
        self.assertAlmostEqual(d_min, 1, places=5)

        # Now rotate this grid a tiny bit
        for theta in np.random.uniform(-0.05, 0.05, 100):
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            points_rotated = points @ rotation_matrix.T

            # Test that the average distance between points is larger than before
            d_min_rotated = _average_d_min(points_rotated)
            self.assertGreater(d_min_rotated, d_min)

