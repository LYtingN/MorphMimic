"""MorphMimic entry module for MD robot_trainer's per-GPU subprocesses.

--dry-run validates data paths and prints the resolved plan without creating a
Trainer, a CUDA context, a process group, or a training run directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Mapping

import tyro
import yaml

from morphmimic.scripts.train import TrainArgs, explicit_fields, run
from morphmimic.train.trainer import TrainCfg


@dataclass
class ClusterArgs(TrainArgs):
  subset_yaml: str = "configs/data/nyx_g1_basic.yaml"
  data_root: str = ""
  """Override the manifest's dataset root; falls back to MORPHMIMIC_DATA_ROOT."""
  expected_world_size: int = 16
  """Fail before training if the platform allocated a different number of ranks."""
  dry_run: bool = False
  """Check the configuration and files without initializing training or CUDA."""


def prepare_run(args: ClusterArgs, env: Mapping[str, str]):
  required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
  missing = [key for key in required if not env.get(key)]
  if missing:
    raise ValueError("robot_trainer environment is missing: " + ", ".join(missing))
  rank, world, local = (int(env[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
  if world != args.expected_world_size or not 0 <= rank < world or local < 0:
    raise ValueError(f"Expected {args.expected_world_size} workers; got rank={rank}, world_size={world}, local_rank={local}")
  source_path = Path(args.subset_yaml).expanduser().resolve()
  subset = yaml.safe_load(source_path.read_text())
  clips = subset.get("clips", [])
  if not clips or len(clips) != len(set(clips)):
    raise ValueError("The training manifest must contain unique, non-empty clip paths")
  root = Path(args.data_root or env.get("MORPHMIMIC_DATA_ROOT") or subset["root"]).expanduser().resolve()
  absent = []
  for clip in clips:
    relative = Path(clip)
    if relative.is_absolute() or ".." in relative.parts:
      raise ValueError(f"Clip paths must be relative to the dataset root: {clip}")
    if not (root / relative).is_file():
      absent.append(clip)
  if absent:
    raise FileNotFoundError(f"{len(absent)} motion files missing under {root}; examples: {absent[:3]}. Set MORPHMIMIC_DATA_ROOT to the dataset mount on every node.")
  subset = {**subset, "root": str(root)}
  cfg = TrainCfg(**{field.name: getattr(args, field.name) for field in fields(TrainCfg)})
  cfg = replace(cfg, gpu_ids=None)
  platform_dir = env.get("MD_RUN_LOG_DIR")
  if platform_dir:
    run_dir = Path(platform_dir).expanduser().resolve()
    cfg = replace(cfg, log_dir=str(run_dir.parent), run_name=run_dir.name)
  else:
    cfg = replace(cfg, run_name=cfg.run_name or "morphmimic-g1-basic-16gpu")
    run_dir = Path(cfg.log_dir).expanduser().resolve() / cfg.run_name
  lr_scale = math.sqrt(world) if cfg.lr_scale_sqrt_world_size else 1.
  plan = {"dry_run": args.dry_run, "rank": rank, "world_size": world,
          "local_rank": local, "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
          "master_addr": env["MASTER_ADDR"], "master_port": int(env["MASTER_PORT"]),
          "source_manifest": str(source_path), "dataset_root": str(root),
          "dataset_clips": len(clips), "dataset_hours": subset.get("hours"),
          "iterations": cfg.iterations, "envs_per_gpu": cfg.num_envs,
          "total_envs": cfg.num_envs * world,
          "global_wm_batch": cfg.wm_batch * world,
          "global_imag_batch": cfg.imag_batch * world,
          "global_kp_imag_batch": cfg.kp_imag_batch * world,
          "effective_learning_rates": {name: getattr(cfg, name) * lr_scale for name in ("wm_lr", "policy_lr", "critic_lr")},
          "run_dir": str(run_dir), "resume": args.resume,
          "manifest_sha256": hashlib.sha256(yaml.safe_dump(subset, sort_keys=True).encode()).hexdigest()}
  return cfg, subset, plan


def materialize_manifest(subset: dict, run_dir: Path, digest: str) -> Path:
  """Identical atomic writes from all ranks are safe on the shared run directory."""
  directory = run_dir / "inputs"
  directory.mkdir(parents=True, exist_ok=True)
  target = directory / f"motions_{digest[:16]}.yaml"
  with tempfile.NamedTemporaryFile("w", dir=directory, prefix=".motions_", suffix=".yaml", delete=False) as stream:
    temporary = Path(stream.name)
    yaml.safe_dump(subset, stream, sort_keys=False)
  try:
    os.replace(temporary, target)
  finally:
    temporary.unlink(missing_ok=True)
  return target


def main() -> None:
  args = tyro.cli(ClusterArgs)
  cfg, subset, plan = prepare_run(args, os.environ)
  print("[morphmimic cluster] " + json.dumps(plan, ensure_ascii=False), flush=True)
  if args.dry_run:
    return
  run_dir = Path(plan["run_dir"])
  manifest = materialize_manifest(subset, run_dir, plan["manifest_sha256"])
  cfg = replace(cfg, subset_yaml=str(manifest))
  if plan["rank"] == 0:
    (run_dir / "cluster_plan.json").write_text(json.dumps({**plan, "train_cfg": asdict(cfg)}, indent=2) + "\n")
  run(cfg, args.resume, distributed=True, explicit=explicit_fields(sys.argv[1:]))


if __name__ == "__main__":
  main()
