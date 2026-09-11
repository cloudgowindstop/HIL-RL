import unittest

import cv2
import numpy as np

from data_convert_refactored.conditioning.camera import prepare_primary_image, prepare_wrist_image


class WristCropTests(unittest.TestCase):
    def test_center_width_75_percent_with_right_offsets(self):
        row = np.arange(640, dtype=np.uint16)[None, :, None]
        left = np.broadcast_to(row, (480, 640, 3)).astype(np.uint8)
        right = (left + 17).astype(np.uint8)

        actual = prepare_wrist_image(
            left,
            right,
            True,
            crop_mode="center_width",
            crop_fraction=0.75,
            left_center_offset_x=64,
            right_center_offset_x=64,
        )
        expected_left = cv2.resize(left[:, 144:624], (224, 112))
        expected_right = cv2.resize(right[:, 144:624], (224, 112))
        expected = np.concatenate((expected_left, expected_right), axis=0)

        self.assertEqual(actual.shape, (224, 224, 3))
        np.testing.assert_array_equal(actual, expected)

    def test_single_arm_does_not_apply_dual_crop(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        actual = prepare_wrist_image(image, None, False)
        self.assertEqual(actual.shape, (224, 224, 3))

    def test_head_camera_removes_only_top_100_pixels(self):
        image = np.broadcast_to(
            np.arange(480, dtype=np.uint16)[:, None, None],
            (480, 640, 3),
        ).astype(np.uint8)
        actual = prepare_primary_image(image, "camera_head")
        expected = cv2.resize(image[100:480, :], (224, 224))
        np.testing.assert_array_equal(actual, expected)

    def test_non_head_primary_camera_is_not_cropped(self):
        image = np.broadcast_to(
            np.arange(480, dtype=np.uint16)[:, None, None],
            (480, 640, 3),
        ).astype(np.uint8)
        actual = prepare_primary_image(image, "camera_right")
        expected = cv2.resize(image, (224, 224))
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
