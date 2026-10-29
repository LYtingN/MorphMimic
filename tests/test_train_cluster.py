"""CPU-only checks for the MD cluster entrypoint; never starts a Trainer."""
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tyro
import yaml

from morphmimic.scripts import train_cluster as cluster


class ClusterTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.data = self.root / "data"
    self.data.mkdir()
    (self.data / "walk.npz").touch()
    self.manifest = self.root / "subset.yaml"
    self.manifest.write_text(yaml.safe_dump({"root": "/old/data/path", "hours": .01, "clips": ["walk.npz"]}))
    self.run_dir = self.root / "platform_logs" / "unique_run"
    self.env = {"RANK": "11", "WORLD_SIZE": "16", "LOCAL_RANK": "3",
                "LOCAL_WORLD_SIZE": "4", "MASTER_ADDR": "10.0.0.1", "MASTER_PORT": "29500",
                "CUDA_VISIBLE_DEVICES": "2,3,5,7", "MD_RUN_LOG_DIR": str(self.run_dir),
                "MORPHMIMIC_DATA_ROOT": str(self.data)}
    self.args = cluster.ClusterArgs(subset_yaml=str(self.manifest), num_envs=1024,
                                    wm_batch=1024, imag_batch=2048, kp_imag_batch=2048)

  def test_multinode_plan_uses_global_world_size_and_platform_logs(self):
    cfg, subset, plan = cluster.prepare_run(self.args, self.env)
    self.assertEqual(plan["total_envs"], 16384)
    self.assertEqual(plan["global_wm_batch"], 16384)
    self.assertEqual(plan["global_imag_batch"], 32768)
    self.assertEqual(plan["global_kp_imag_batch"], 32768)
    self.assertEqual(subset["root"], str(self.data))
    self.assertEqual(Path(cfg.log_dir) / cfg.run_name, self.run_dir)
    self.assertIsNone(cfg.gpu_ids)
    self.assertFalse(self.run_dir.exists())

  def test_expected_world_size_is_checked(self):
    with self.assertRaises(ValueError):
      cluster.prepare_run(self.args, {**self.env, "WORLD_SIZE": "8"})

  def test_missing_platform_context_is_rejected(self):
    with self.assertRaises(ValueError):
      cluster.prepare_run(self.args, {})

  def test_missing_data_is_detected_before_training(self):
    (self.data / "walk.npz").unlink()
    with self.assertRaises(FileNotFoundError):
      cluster.prepare_run(self.args, self.env)

  def test_relative_paths_cannot_escape_dataset_root(self):
    self.manifest.write_text(yaml.safe_dump({"root": str(self.data), "clips": ["../escape.npz"]}))
    with self.assertRaises(ValueError):
      cluster.prepare_run(self.args, self.env)

  def test_explicit_data_root_overrides_environment(self):
    self.args.data_root = str(self.data)
    _, subset, _ = cluster.prepare_run(self.args, {**self.env, "MORPHMIMIC_DATA_ROOT": "/missing"})
    self.assertEqual(subset["root"], str(self.data))

  def test_dry_run_never_calls_training_or_writes_run_directory(self):
    self.args.dry_run = True
    with patch.dict(os.environ, self.env, clear=True), \
         patch.object(cluster.tyro, "cli", return_value=self.args), \
         patch.object(cluster, "run") as run, \
         contextlib.redirect_stdout(io.StringIO()):
      cluster.main()
    run.assert_not_called()
    self.assertFalse(self.run_dir.exists())

  def test_manifest_materialization_keeps_input_unchanged(self):
    original = self.manifest.read_bytes()
    _, subset, plan = cluster.prepare_run(self.args, self.env)
    path = cluster.materialize_manifest(subset, self.run_dir, plan["manifest_sha256"])
    self.assertEqual(yaml.safe_load(path.read_text()), subset)
    self.assertEqual(self.manifest.read_bytes(), original)
    self.assertEqual(cluster.materialize_manifest(subset, self.run_dir, plan["manifest_sha256"]), path)

  def test_published_job_arguments_match_the_entrypoint(self):
    repo = Path(__file__).resolve().parents[1]
    job = yaml.safe_load((repo / "configs/jobs/morphmimic_baidu_bj_4090_16gpu.yaml").read_text())
    task = job["tasks"][0]
    args = tyro.cli(cluster.ClusterArgs, args=task["args"])
    self.assertEqual(job["cluster"], "baidu_bj_4090")
    self.assertEqual(job["num_gpus"], args.expected_world_size)
    self.assertEqual(task["type"], "robot_trainer")
    self.assertEqual(task["entry"], "morphmimic.scripts.train_cluster")
    self.assertIsNone(args.gpu_ids)
    self.assertTrue(args.kp_interface)
    self.assertEqual(args.iterations, 100000)
    self.assertEqual(args.ckpt_every, 1000)
    self.assertAlmostEqual(args.policy_lr * 4, 2e-4 * 8**.5)


if __name__ == "__main__":
  unittest.main()
