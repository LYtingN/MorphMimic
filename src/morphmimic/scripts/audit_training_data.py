"""CPU-only audit of G1 references and a filtered training manifest.

Reads every source array, checks the old inclusion decisions, compares saved body
poses to sampled MuJoCo forward kinematics, and measures collision-geometry foot
clearance over every frame. No simulator stepping or training is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
import yaml
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg, get_spec

from morphmimic.constants import JOINT_NAMES, TRACKED_BODY_NAMES
from morphmimic.data.difficulty import DifficultyCfg, assess_motion, longest_run


def rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
  """Rotate vectors by unit wxyz quaternions, with NumPy broadcasting."""
  xyz = q[..., 1:]
  uv = np.cross(xyz, vector)
  return vector + 2 * (q[..., :1] * uv + np.cross(xyz, uv))


def foot_clearance(model, positions, quaternions, body_names):
  """Lowest point of each foot's seven collision capsules relative to z=0."""
  clearances = []
  for side in ("left", "right"):
    bottoms = []
    for number in range(1, 8):
      gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                             f"{side}_foot{number}_collision")
      if gid < 0 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CAPSULE:
        raise ValueError("Expected the G1's capsule foot collision geometry")
      bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid]))
      bid = body_names.index(bname)
      q = quaternions[:, bid].astype(np.float64)
      q /= np.linalg.norm(q, axis=-1, keepdims=True)
      center = positions[:, bid] + rotate(q, model.geom_pos[gid])
      local_axis = rotate(model.geom_quat[gid], np.array([0., 0., 1.]))
      axis = rotate(q, local_axis)
      radius, half_length = model.geom_size[gid, :2]
      bottoms.append(center[:, 2] - radius - half_length * np.abs(axis[:, 2]))
    clearances.append(np.min(bottoms, axis=0))
  return np.stack(clearances, axis=-1)


def sha256(path: Path) -> str:
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-subset", type=Path, required=True)
  parser.add_argument("--training-subset", type=Path, required=True)
  parser.add_argument("--policy", type=Path, required=True)
  parser.add_argument("--previous-report", type=Path, required=True)
  parser.add_argument("--snapshot-hashes", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--fk-samples", type=int, default=64)
  args = parser.parse_args()
  source = yaml.safe_load(args.source_subset.read_text())
  selected = yaml.safe_load(args.training_subset.read_text())
  policy = DifficultyCfg(**yaml.safe_load(args.policy.read_text()))
  previous = json.loads(args.previous_report.read_text())
  old_rows = {r["clip"]: r for r in previous["records"]}
  snapshot = {r["clip"]: r["sha256"] for r in json.loads(args.snapshot_hashes.read_text())}
  source_names, selected_names = source["clips"], selected["clips"]
  if len(set(source_names)) != len(source_names) or len(set(selected_names)) != len(selected_names):
    raise ValueError("Duplicate entries in a manifest")
  if not set(selected_names) <= set(source_names):
    raise ValueError("Selected manifest contains a clip outside the original dataset")
  selected_names = set(selected_names)
  model = get_spec().compile()
  model_data = mujoco.MjData(model)
  soft_factor = get_g1_robot_cfg().articulation.soft_joint_pos_limit_factor
  records, duplicates = [], defaultdict(list)
  root = Path(source["root"])
  for number, clip in enumerate(source_names, 1):
    path = root / clip
    flags = []
    with np.load(path, allow_pickle=False) as z:
      fps = float(np.asarray(z["fps"]).item())
      body_names, joint_names = [str(x) for x in z["body_names"]], [str(x) for x in z["joint_names"]]
      arrays = {k: z[k] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    p, q, jp, jv = (arrays[k] for k in ("body_pos_w", "body_quat_w", "joint_pos", "joint_vel"))
    frames = len(jp)
    expected = {"joint_pos": (frames, 29), "joint_vel": (frames, 29),
                "body_pos_w": (frames, len(body_names), 3), "body_quat_w": (frames, len(body_names), 4),
                "body_lin_vel_w": (frames, len(body_names), 3), "body_ang_vel_w": (frames, len(body_names), 3)}
    if any(arrays[k].shape != shape for k, shape in expected.items()):
      raise ValueError(f"Invalid array shape: {clip}")
    if fps != 50 or not all(np.isfinite(a).all() for a in arrays.values()):
      raise ValueError(f"Non-finite array or wrong fps: {clip}")
    if len(set(joint_names)) != 29 or set(joint_names) != set(JOINT_NAMES):
      raise ValueError(f"Wrong G1 joint schema: {clip}")
    if not set(TRACKED_BODY_NAMES) <= set(body_names) or len(set(body_names)) != len(body_names):
      raise ValueError(f"Wrong G1 body schema: {clip}")
    norms = np.linalg.norm(q.astype(np.float64), axis=-1)
    if np.any(norms < 1e-8):
      raise ValueError(f"Zero-length quaternion: {clip}")
    normalized_q = q / norms[..., None]
    quat_deviation = float(np.max(np.abs(norms - 1)))
    if quat_deviation > .001:
      flags.append("quaternion_norm")
    old_reasons, old_stats = assess_motion(path.stem, p, q, body_names, fps, policy)
    if old_reasons != old_rows[clip]["reasons"] or (not old_reasons) != (clip in selected_names):
      raise ValueError(f"Filtering no longer reproduces its saved decisions: {clip}")
    if clip in snapshot and sha256(path) != snapshot[clip]:
      raise ValueError(f"Source changed since the stopped training run: {clip}")
    digest = hashlib.sha256()
    for array in (jp, p, normalized_q):
      digest.update(np.ascontiguousarray(array).tobytes())
    duplicates[digest.hexdigest()].append(clip)
    jids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names])
    if np.any(jids < 0):
      raise ValueError(f"Joint missing from the CPU model: {clip}")
    lo, hi = model.jnt_range[jids].T
    hard_error = np.maximum(np.maximum(lo-jp, jp-hi), 0)
    midpoint, width = (lo+hi)/2, (hi-lo)*soft_factor/2
    soft_error = np.maximum(np.maximum(midpoint-width-jp, jp-midpoint-width), 0)
    if np.any(hard_error > np.radians(3)):
      flags.append("hard_joint_limit_over_3deg")
    joint_step = np.abs(np.diff(jp.astype(np.float64), axis=0))
    body_step = np.linalg.norm(np.diff(p.astype(np.float64), axis=0), axis=-1)
    angle_step = np.degrees(2*np.arccos(np.clip(np.abs(np.sum(normalized_q[1:]*normalized_q[:-1], axis=-1)), 0, 1)))
    if joint_step.max() > np.radians(30) or body_step.max() > .3 or angle_step.max() > 45:
      flags.append("temporal_jump_review")
    fd_joint = np.gradient(jp.astype(np.float64), 1/fps, axis=0)
    fd_body = np.gradient(p.astype(np.float64), 1/fps, axis=0)
    feet = foot_clearance(model, p, q, body_names)
    both_above = np.all(feet > .02, axis=-1)
    penetration = np.any(feet < -.05, axis=-1)
    if np.any(feet < -.08) or longest_run(penetration)/fps >= .1:
      flags.append("foot_penetration_review")
    if longest_run(both_above)/fps >= .3:
      flags.append("both_feet_above_ground_review")
    bids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names])
    if np.any(bids < 0):
      raise ValueError(f"Body missing from the CPU model: {clip}")
    root_index = body_names.index("pelvis")
    samples = np.unique(np.r_[np.linspace(0, frames-1, min(args.fk_samples, frames), dtype=int),
                              np.argmin(p[:,root_index,2]), np.argmin(feet.min(axis=1)),
                              np.argmax(feet.min(axis=1))])
    fk_max, fk_sum = 0., 0.
    for frame in samples:
      model_data.qpos[:3] = p[frame,root_index]
      model_data.qpos[3:7] = normalized_q[frame,root_index]
      model_data.qpos[model.jnt_qposadr[jids]] = jp[frame]
      mujoco.mj_kinematics(model, model_data)
      error = np.linalg.norm(model_data.xpos[bids]-p[frame], axis=-1)
      fk_max, fk_sum = max(fk_max,float(error.max())), fk_sum + float(error.mean())
    if fk_max > .005:
      flags.append("forward_kinematics_mismatch")
    up = 1-2*(normalized_q[...,1]**2+normalized_q[...,2]**2)
    tilt = np.degrees(np.arccos(np.clip(up[:,[root_index,body_names.index('torso_link')]], -1, 1))).max(axis=1)
    threshold_mask = (tilt>policy.maximum_tilt_degrees) | (p[:,root_index,2]<policy.minimum_pelvis_height_m)
    threshold_mask |= np.all(p[:,[body_names.index('left_ankle_roll_link'),body_names.index('right_ankle_roll_link')],2]>policy.maximum_both_ankles_height_m,axis=1)
    if clip not in selected_names and all(r in ['excessive_tilt','low_pelvis','raised_ankles'] for r in old_reasons) and threshold_mask.mean()<.02:
      flags.append("whole_clip_exclusion_review")
    record={"clip":clip,"source":"lafan1" if '/lafan1/' in clip else 'bones_seed',
            "included":clip in selected_names,"frames":frames,"seconds":frames/fps,
            "old_reasons":old_reasons,"flags":flags,
            "quaternion_max_norm_deviation":quat_deviation,
            "hard_limit_max_excess_degrees":float(np.degrees(hard_error).max()),
            "hard_limit_frame_fraction_over_3deg":float(np.any(hard_error>np.radians(3),axis=-1).mean()),
            "soft_limit_frame_fraction_over_3deg":float(np.any(soft_error>np.radians(3),axis=-1).mean()),
            "soft_limit_max_excess_degrees":float(np.degrees(soft_error).max()),
            "maximum_joint_step_degrees":float(np.degrees(joint_step).max()),
            "maximum_body_step_m":float(body_step.max()),
            "maximum_body_orientation_step_degrees":float(angle_step.max()),
            "maximum_joint_velocity_rad_s":float(np.abs(jv).max()),
            "joint_velocity_derivative_rmse_rad_s":float(np.sqrt(np.mean((fd_joint[1:-1]-jv[1:-1])**2))),
            "body_velocity_derivative_rmse_m_s":float(np.sqrt(np.mean((fd_body[1:-1]-arrays['body_lin_vel_w'][1:-1])**2))),
            "minimum_foot_collision_z_m":float(feet.min()),
            "longest_foot_penetration_seconds":longest_run(penetration)/fps,
            "longest_both_feet_clear_seconds":longest_run(both_above)/fps,
            "foot_penetration_frame_fraction":float(penetration.mean()),
            "old_threshold_frame_fraction":float(threshold_mask.mean()),
            "fk_frames_checked":len(samples),"fk_max_error_m":fk_max,
            "fk_mean_error_m":fk_sum/len(samples),**{k:v for k,v in old_stats.items() if k not in ['frames','fps','seconds']}}
    records.append(record)
    if number%40==0:
      print(f"Audited {number}/{len(source_names)} clips",flush=True)
  def summary(rows):
    return {"clips":len(rows),"frames":sum(r['frames'] for r in rows),
            "minutes":sum(r['seconds'] for r in rows)/60,
            "flag_counts":dict(Counter(f for r in rows for f in r['flags'])),
            "by_source":{s:{"clips":sum(r['source']==s for r in rows),
                            "minutes":sum(r['seconds'] for r in rows if r['source']==s)/60} for s in ['lafan1','bones_seed']}}
  report={"created_at_utc":datetime.now(timezone.utc).isoformat(),
          "training_started":False,"processing":"CPU-only static geometry and array audit; no physics rollout",
          "source_manifest_sha256":sha256(args.source_subset),"training_manifest_sha256":sha256(args.training_subset),
          "all_array_frames_checked":sum(r['frames'] for r in records),
          "all_arrays_finite":True,"all_clip_fps":50,"filter_decisions_reproduced":True,
          "selected_files_match_training_snapshot":len(snapshot),"soft_joint_limit_factor":soft_factor,
          "all":summary(records),"included":summary([r for r in records if r['included']]),
          "excluded":summary([r for r in records if not r['included']]),
          "exact_motion_duplicate_groups":[v for v in duplicates.values() if len(v)>1],
          "notes":["Flags identify review candidates, not automatic exclusions.",
                   "Foot clearance uses collision capsule surfaces over every frame, not ankle-link origin heights.",
                   "Foot clearance does not measure contact force or prove dynamic feasibility.",
                   "Forward kinematics checks are sampled; other temporal and geometry statistics cover all frames.",
                   "The source data and existing training manifest are not modified."],"records":records}
  args.output_dir.mkdir(parents=True,exist_ok=True)
  (args.output_dir/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
  for name,rows in [('all_clips.csv',records),('review_candidates.csv',[r for r in records if r['flags']])]:
    with (args.output_dir/name).open('w',newline='') as stream:
      writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader()
      for row in rows:writer.writerow({**row,'flags':';'.join(row['flags']),'old_reasons':';'.join(row['old_reasons'])})
  print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':
  main()
