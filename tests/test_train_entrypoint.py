"""Ensure torchrun workers train directly rather than launching recursively."""
import os
import sys
import unittest
from unittest.mock import patch

from morphmimic.scripts import train


class EntrypointTests(unittest.TestCase):
  def route(self, environment, gpus):
    args = train.TrainArgs(gpu_ids=gpus)
    with patch.dict(os.environ, environment, clear=True), \
         patch.object(sys, "argv", ["train"]), \
         patch.object(train.tyro, "cli", return_value=args), \
         patch.object(train, "select_gpus") as select, \
         patch.object(train, "launch") as launch, \
         patch.object(train, "run") as run:
      train.main()
    self.last_select = select
    return launch, run

  def test_torchrun_worker_does_not_launch_recursively(self):
    launch, run = self.route({"LOCAL_RANK": "3", "RANK": "3", "WORLD_SIZE": "8"}, list(range(8)))
    launch.assert_not_called()
    run.assert_called_once()
    self.assertTrue(run.call_args.kwargs["distributed"])
    self.last_select.assert_not_called()
    self.assertIsNone(run.call_args.args[0].gpu_ids)

  def test_multinode_worker_preserves_scheduler_allocation(self):
    launch, run = self.route({"LOCAL_RANK": "3", "RANK": "11", "WORLD_SIZE": "16",
                              "CUDA_VISIBLE_DEVICES": "2,3,5,7"}, None)
    launch.assert_not_called()
    self.last_select.assert_not_called()
    self.assertTrue(run.call_args.kwargs["distributed"])
    self.assertIsNone(run.call_args.args[0].gpu_ids)

  def test_single_visible_gpu_uses_logical_zero(self):
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}, clear=True):
      self.assertEqual(train.local_cuda_index(3), 0)

  def test_shared_node_allocation_uses_local_rank(self):
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3,5,7"}, clear=True):
      self.assertEqual(train.local_cuda_index(3), 3)

  def test_uuid_gpu_allocation_is_supported(self):
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-allocated-device"}, clear=True):
      self.assertEqual(train.local_cuda_index(7), 0)

  def test_invalid_local_rank_fails_before_cuda(self):
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear=True):
      with self.assertRaises(ValueError):
        train.local_cuda_index(3)

  def test_existing_multi_gpu_entrypoint_still_uses_torchrunx(self):
    launch, run = self.route({}, list(range(8)))
    launch.assert_called_once()
    run.assert_not_called()

  def test_existing_single_gpu_entrypoint_still_trains_directly(self):
    launch, run = self.route({}, None)
    launch.assert_not_called()
    run.assert_called_once()
    self.assertNotIn("distributed", run.call_args.kwargs)


if __name__ == "__main__":
  unittest.main()
