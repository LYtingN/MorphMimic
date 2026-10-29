"""CPU checks that the upper interface observes pelvis and both hands only."""
import unittest

import torch

from morphmimic.constants import GOAL_K, KP_FEAT_DIM, KP_GOAL_DIM, KP_POINT_DIM, NUM_KP, PROPRIO_DIM
from morphmimic.model.keypoints import fixed_kp_masks, mask_keypoint_goal, sample_kp_masks
from morphmimic.model.policy import Policy, PolicyCfg
from morphmimic.train.imagination import act_kp
from morphmimic.train.normalizers import RunningNormalizer


class ThreePointTests(unittest.TestCase):
  def test_training_and_evaluation_masks_select_exactly_pelvis_and_wrists(self):
    points, frames = sample_kp_masks(32, "upper", torch.device("cpu"))
    expected = torch.tensor([True, False, False, True, True]).expand(32, NUM_KP)
    self.assertTrue(torch.equal(points, expected))
    self.assertTrue(torch.equal(points.sum(-1), torch.full((32,), 3)))
    self.assertTrue(frames.all())
    evaluation = fixed_kp_masks(32, "upper", GOAL_K, torch.device("cpu"))
    self.assertTrue(torch.equal(points, evaluation[0]))
    self.assertTrue(torch.equal(frames, evaluation[1]))

  def test_ankle_targets_are_zeroed_after_normalization(self):
    points, frames = fixed_kp_masks(2, "upper", GOAL_K, torch.device("cpu"))
    features = torch.randn(2, KP_FEAT_DIM)
    masked = mask_keypoint_goal(features, points, frames)
    self.assertEqual(masked.shape, (2, KP_GOAL_DIM))
    blocks = masked[:, :KP_FEAT_DIM].reshape(2, GOAL_K, NUM_KP, KP_POINT_DIM)
    self.assertEqual(torch.count_nonzero(blocks[:, :, 1:3]).item(), 0)
    original = features.reshape_as(blocks)
    self.assertTrue(torch.equal(blocks[:, :, [0, 3, 4]], original[:, :, [0, 3, 4]]))

  def test_actor_output_is_invariant_to_ankle_targets(self):
    torch.manual_seed(3)
    policy = Policy(PolicyCfg(enc_hidden=(16,), dec_hidden=(16,), latent=8,
                              hist_len=2, keypoint=True, bf16=False)).eval()
    obs_norm = RunningNormalizer(PROPRIO_DIM).eval()
    kp_norm = RunningNormalizer(KP_FEAT_DIM).eval()
    kp_norm.mean.copy_(torch.randn(KP_FEAT_DIM))
    kp_norm.var.copy_(torch.rand(KP_FEAT_DIM) + .1)
    history = torch.randn(2, 2*PROPRIO_DIM)
    goal = torch.randn(2, KP_FEAT_DIM)
    changed_feet = goal.clone()
    changed_feet.reshape(2, GOAL_K, NUM_KP, KP_POINT_DIM)[:, :, 1:3] = 1e6
    masks = fixed_kp_masks(2, "upper", GOAL_K, torch.device("cpu"))
    with torch.inference_mode():
      original = act_kp(policy, obs_norm, kp_norm, history, goal, *masks)
      changed = act_kp(policy, obs_norm, kp_norm, history, changed_feet, *masks)
      changed_hand = goal.clone()
      changed_hand.reshape(2, GOAL_K, NUM_KP, KP_POINT_DIM)[:, :, 3] += 4
      hand_action = act_kp(policy, obs_norm, kp_norm, history, changed_hand, *masks)
    self.assertTrue(torch.equal(original, changed))
    self.assertFalse(torch.equal(original, hand_action))


if __name__ == "__main__":
  unittest.main()
