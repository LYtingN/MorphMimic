"""Regression checks for dataset decisions that affect an expensive training run."""
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from morphmimic.data.difficulty import DifficultyCfg, assess_motion


class DifficultyTests(unittest.TestCase):
  def setUp(self):
    policy = Path(__file__).resolve().parents[1] / "configs/data/nyx_g1_basic_filter.yaml"
    self.cfg = DifficultyCfg(**yaml.safe_load(policy.read_text()))
    self.names = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"]
    self.p = np.zeros((50, 4, 3))
    self.p[:, :, 2] = [.75, .8, .03, .03]
    self.q = np.zeros((50, 4, 4))
    self.q[..., 0] = 1

  def assess(self, name="walk_forward", cfg=None):
    return assess_motion(name, self.p, self.q, self.names, 50, cfg or self.cfg)

  def test_upright_walking_is_retained(self):
    self.assertEqual(self.assess()[0], [])

  def test_handstand_name_is_excluded_even_without_inversion_in_sample(self):
    self.assertIn("acrobatics", self.assess("230324__handstand_ff_loop_180_R_001")[0])

  def test_camelcase_lafan_floor_and_combat_names(self):
    self.assertIn("floor_motion", self.assess("fallAndGetUp1_subject1")[0])
    self.assertIn("combat_or_sprint", self.assess("fightAndSports1_subject4")[0])

  def test_unnamed_sustained_inversion_is_excluded(self):
    self.q[:10, :2, :] = [0, 1, 0, 0]
    self.assertIn("excessive_tilt", self.assess()[0])

  def test_one_frame_tilt_spike_is_not_sustained(self):
    self.q[0, :2, :] = [0, 1, 0, 0]
    self.assertNotIn("excessive_tilt", self.assess()[0])

  def test_quaternion_scale_does_not_change_tilt(self):
    self.q[:10, :2, :] = [0, 1, 0, 0]
    before = self.assess()
    self.q *= 3
    after = self.assess()
    self.assertEqual(before, after)

  def test_floor_pose_is_detected_independently_of_filename(self):
    self.p[:10, 0, 2] = .2
    self.assertIn("low_pelvis", self.assess()[0])

  def test_raised_ankles_duration_boundary(self):
    self.p[:14, 2:, 2] = .2
    self.assertNotIn("raised_ankles", self.assess()[0])
    self.p[14, 2:, 2] = .2
    self.assertIn("raised_ankles", self.assess()[0])

  def test_jump_option_changes_only_name_rule(self):
    self.assertIn("jumps_and_kicks", self.assess("jumps1_subject5")[0])
    cfg = replace(self.cfg, exclude_jumps_and_kicks=False)
    self.assertEqual(self.assess("jumps1_subject5", cfg)[0], [])

  def test_nonfinite_pose_is_rejected(self):
    self.p[0, 0, 0] = np.nan
    with self.assertRaises(ValueError):
      self.assess()

  def test_zero_quaternion_is_rejected(self):
    self.q[0, 0] = 0
    with self.assertRaises(ValueError):
      self.assess()


if __name__ == "__main__":
  unittest.main()
