"""Auditable whole-clip difficulty filtering for a first-stage G1 dataset.

Kinematic limits are heuristics for selecting upright motions, not a guarantee
that a retained reference is dynamically feasible or easy for a learned policy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from morphmimic.data.filters import name_filter_reject


@dataclass(frozen=True)
class DifficultyCfg:
  version: int
  exclude_jumps_and_kicks: bool
  minimum_clip_seconds: float
  maximum_tilt_degrees: float
  minimum_excessive_tilt_seconds: float
  minimum_pelvis_height_m: float
  minimum_low_pelvis_seconds: float
  maximum_both_ankles_height_m: float
  minimum_raised_ankles_seconds: float
  name_rules: Mapping[str, str]

  def __post_init__(self) -> None:
    if self.version != 1:
      raise ValueError(f"Unsupported difficulty policy version: {self.version}")
    if not 0 < self.maximum_tilt_degrees < 180:
      raise ValueError("maximum_tilt_degrees must be between 0 and 180")
    for value in (self.minimum_clip_seconds, self.minimum_excessive_tilt_seconds,
                  self.minimum_low_pelvis_seconds, self.minimum_raised_ankles_seconds):
      if value <= 0:
        raise ValueError("Duration limits must be positive")
    for pattern in self.name_rules.values():
      re.compile(pattern)


def normalized_clip_name(name: str) -> str:
  # Preserve LAFAN categories such as fallAndGetUp1 and fightAndSports1.
  name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
  return re.sub(r"[_\-]+", " ", name).lower()


def longest_run(mask: np.ndarray) -> int:
  """Consecutive frames, with isolated excursions kept separate."""
  edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
  return int(np.max(edges[1::2] - edges[::2], initial=0))


def assess_motion(
  name: str, positions: np.ndarray, quaternions: np.ndarray,
  body_names: list[str], fps: float, cfg: DifficultyCfg,
) -> tuple[list[str], dict[str, float | int]]:
  """Return every exclusion reason and measured kinematics (wxyz quaternions)."""
  if not np.isfinite(fps) or fps != 50:
    raise ValueError(f"Expected finite 50 Hz motion, got {fps}")
  if positions.ndim != 3 or positions.shape[-1] != 3:
    raise ValueError("body_pos_w must have shape [frames, bodies, 3]")
  frames, bodies, _ = positions.shape
  if frames < 2 or bodies != len(body_names) or quaternions.shape != (frames, bodies, 4):
    raise ValueError("Inconsistent frame/body dimensions")
  if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
    raise ValueError("Non-finite reference pose")
  indices = {name: body_names.index(name) for name in
             ("pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link")}
  selected = quaternions[:, [indices["pelvis"], indices["torso_link"]], :].astype(np.float64)
  norm_squared = np.sum(selected**2, axis=-1)
  if np.any(norm_squared < 1e-12):
    raise ValueError("Zero-length body quaternion")
  up_z = 1 - 2 * (selected[..., 1]**2 + selected[..., 2]**2) / norm_squared
  tilt = np.degrees(np.arccos(np.clip(up_z, -1, 1))).max(axis=1)
  pelvis_z = positions[:, indices["pelvis"], 2]
  ankles_z = positions[:, [indices["left_ankle_roll_link"], indices["right_ankle_roll_link"]], 2]
  tilt_frames = longest_run(tilt > cfg.maximum_tilt_degrees)
  low_frames = longest_run(pelvis_z < cfg.minimum_pelvis_height_m)
  raised_frames = longest_run(np.all(ankles_z > cfg.maximum_both_ankles_height_m, axis=1))
  stats = {"frames": frames, "fps": fps, "seconds": frames / fps,
           "maximum_tilt_degrees": float(tilt.max()),
           "minimum_pelvis_height_m": float(pelvis_z.min()),
           "excessive_tilt_seconds": tilt_frames / fps,
           "low_pelvis_seconds": low_frames / fps,
           "raised_ankles_seconds": raised_frames / fps}
  text = normalized_clip_name(name)
  reasons = [category for category, pattern in cfg.name_rules.items()
             if (category != "jumps_and_kicks" or cfg.exclude_jumps_and_kicks)
             and re.search(pattern, text)]
  unsupported = name_filter_reject(name)
  if unsupported:
    reasons.append(f"unsupported_prop:{unsupported}")
  for rejected, reason in (
    (frames / fps < cfg.minimum_clip_seconds, "too_short"),
    (tilt_frames >= int(np.ceil(cfg.minimum_excessive_tilt_seconds * fps - 1e-6)), "excessive_tilt"),
    (low_frames >= int(np.ceil(cfg.minimum_low_pelvis_seconds * fps - 1e-6)), "low_pelvis"),
    (raised_frames >= int(np.ceil(cfg.minimum_raised_ankles_seconds * fps - 1e-6)), "raised_ankles"),
  ):
    if rejected:
      reasons.append(reason)
  return reasons, stats
