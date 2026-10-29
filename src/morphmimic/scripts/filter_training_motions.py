"""Build a filtered training manifest without changing any source NPZ files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from morphmimic.data.difficulty import DifficultyCfg, assess_motion


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-subset", type=Path, required=True)
  parser.add_argument("--policy", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--report-dir", type=Path, required=True)
  parser.add_argument("--allow-jumps-and-kicks", action="store_true")
  args = parser.parse_args()
  if args.output.resolve() == args.source_subset.resolve():
    raise ValueError("Output must differ from the original manifest")
  source = yaml.safe_load(args.source_subset.read_text())
  policy = DifficultyCfg(**yaml.safe_load(args.policy.read_text()))
  if args.allow_jumps_and_kicks:
    policy = replace(policy, exclude_jumps_and_kicks=False)
  root = Path(source["root"]).resolve()
  if len(source["clips"]) != len(set(source["clips"])):
    raise ValueError("Duplicate input clips")
  records = []
  for clip in source["clips"]:
    path = root / clip
    with np.load(path, allow_pickle=False) as z:
      positions, quaternions = z["body_pos_w"], z["body_quat_w"]
      reasons, stats = assess_motion(path.stem, positions, quaternions,
                                    [str(s) for s in z["body_names"]],
                                    float(np.asarray(z["fps"]).item()), policy)
      for key in ("joint_pos", "joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        if z[key].shape[0] != stats["frames"] or not np.isfinite(z[key]).all():
          raise ValueError(f"Invalid {key} in {path}")
    records.append({"clip": clip, "source": "lafan1" if "/lafan1/" in clip else "bones_seed",
                    "included": not reasons, "reasons": reasons, **stats})
  kept = [r for r in records if r["included"]]
  rejected = [r for r in records if not r["included"]]
  if not kept:
    raise ValueError("The policy excluded every input clip")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.report_dir.mkdir(parents=True, exist_ok=True)
  output = {"root": str(root), "hours": sum(r["seconds"] for r in kept) / 3600,
            "clips": [r["clip"] for r in kept]}
  args.output.write_text(yaml.safe_dump(output, sort_keys=False))
  def summarize(rows):
    return {"clips": len(rows), "frames": sum(r["frames"] for r in rows),
            "hours": sum(r["seconds"] for r in rows) / 3600,
            "by_source": dict(Counter(r["source"] for r in rows))}
  report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_subset": str(args.source_subset.resolve()),
            "source_subset_sha256": hashlib.sha256(args.source_subset.read_bytes()).hexdigest(),
            "output_subset": str(args.output.resolve()),
            "output_subset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "policy": asdict(policy),
            "input": summarize(records), "kept": summarize(kept), "excluded": summarize(rejected),
            "exclusion_reason_counts": dict(Counter(reason for r in rejected for reason in r["reasons"])),
            "note": "Whole-clip exclusion; source NPZ files are unchanged. Raised ankles are a pose heuristic, not a measurement of foot contact.",
            "records": records}
  (args.report_dir / "filter_report.json").write_text(json.dumps(report, indent=2) + "\n")
  for filename, rows in (("included.csv", kept), ("excluded.csv", rejected)):
    with (args.report_dir / filename).open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(records[0]))
      writer.writeheader()
      for row in rows:
        writer.writerow({**row, "reasons": ";".join(row["reasons"])})
  print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
  main()
