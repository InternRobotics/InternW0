#!/usr/bin/env python3
"""Load a LIBERO posttrain checkpoint and run one dummy 7-D action."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from experiments.libero.w0_policy.deploy_policy import get_model

DEFAULT_CKPT = os.environ.get("W0_LIBERO_CHECKPOINT", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, required=not bool(DEFAULT_CKPT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    policy = get_model({"ckpt": args.ckpt, "device": args.device})
    obs = {
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, -0.04], dtype=np.float32),
    }
    action = np.asarray(
        policy.act(obs, "pick up the black bowl between the plate and the ramekin and place it on the plate"),
        dtype=np.float32,
    )
    report = {
        "ckpt": args.ckpt,
        "action7": action.tolist(),
        "finite": bool(np.isfinite(action).all()),
        "shape": list(action.shape),
    }
    print(json.dumps(report, indent=2), flush=True)
    if not report["finite"] or action.shape != (7,):
        raise SystemExit("policy smoke produced an invalid 7-D action")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
