import unittest
from pathlib import Path

from data_convert_refactored.preparation.episode import (
    action_proprio_source_arrays,
    load_episode,
)


ROOT = Path(__file__).resolve().parents[2]


class EpisodeReaderTests(unittest.TestCase):
    def test_pick_spoon_single_arm(self):
        episode = load_episode(ROOT / "dataset_raw/pick_spoon/0731_103412/trajectory.hdf5")
        self.assertFalse(episode.is_dual_arm)
        self.assertEqual(episode.length, 274)
        self.assertEqual(episode.left.puppet_pose_xyzw.shape, (274, 7))

    def test_tienyi_dual_arm(self):
        episode = load_episode(
            ROOT
            / "dataset_raw/tienyi_prod2_dualArm-gripper-3cameras_193_insert_hose_into_motor_20260626_pm/0626_192345/data/trajectory.hdf5"
        )
        self.assertTrue(episode.is_dual_arm)
        self.assertEqual(episode.length, 1206)
        self.assertEqual(episode.left.master_joints.shape, (1206, 7))
        self.assertEqual(episode.right.master_joints.shape, (1206, 7))

    def test_tienyi_second_dual_arm_dataset(self):
        episode = load_episode(
            ROOT
            / "dataset_raw/tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260803_pm/0803_173020/data/trajectory.hdf5"
        )
        self.assertTrue(episode.is_dual_arm)
        self.assertEqual(episode.length, 1432)
        self.assertEqual(episode.camera_names, ("camera_head", "camera_left", "camera_right"))
        source = action_proprio_source_arrays(episode, "puppet_next_frame")
        self.assertEqual(source["source.puppet.pose_xyzw"].shape, (1432, 14))
        self.assertEqual(source["source.puppet.gripper"].shape, (1432, 2))

    def test_master_joint_source_fields_follow_dual_arm_order(self):
        episode = load_episode(
            ROOT
            / "dataset_raw/tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260803_pm/0803_173020/data/trajectory.hdf5"
        )
        source = action_proprio_source_arrays(episode, "master_joint_fk_same_frame")
        self.assertEqual(source["source.puppet.joints"].shape, (1432, 14))
        self.assertEqual(source["source.master.joints"].shape, (1432, 14))
        self.assertEqual(source["source.master.gripper"].shape, (1432, 2))


if __name__ == "__main__":
    unittest.main()
