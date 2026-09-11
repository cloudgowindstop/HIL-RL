import unittest

from data_convert_refactored.conditioning.camera import CameraState, resolve_camera_processing


class CameraConfigTest(unittest.TestCase):
    def test_normal_profile_preserves_verified_crop_parameters(self):
        profile = resolve_camera_processing(CameraState.NORMAL)
        self.assertEqual(profile.wrist_crop_mode, "center_width")
        self.assertEqual(profile.wrist_crop_fraction, 0.75)
        self.assertEqual(profile.wrist_left_center_offset_x, 64)
        self.assertEqual(profile.wrist_right_center_offset_x, 64)
        self.assertEqual(profile.head_crop_top_pixels, 100)


if __name__ == "__main__":
    unittest.main()
