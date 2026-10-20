#!/usr/bin/env python3
"""LIBERO env client. Run this with the LIBERO python, not w0."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.libero_utils import (
    DUMMY_ACTION, MAX_STEPS, action_rollout_stats, encode_image_b64,
    get_libero_env, save_rollout_mp4, stitch_obs_frame, write_libero_config,
)


def _request(url: str, payload: dict | None = None, timeout: float = 300.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    method = "GET" if payload is None else "POST"
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _obs_payload(obs: dict, instruction: str) -> dict:
    return {
        "instruction": instruction,
        "agentview_image": encode_image_b64(obs["agentview_image"]),
        "robot0_eye_in_hand_image": encode_image_b64(obs["robot0_eye_in_hand_image"]),
        "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32).tolist(),
        "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32).tolist(),
        "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--trial-start", type=int, default=0,
                        help="Global trial/init index used when sharding an evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--binarize-gripper", action="store_true",
        help="Threshold the runtime gripper command to {-1,+1}, matching InternVLA-A1.5.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(os.environ.get("W0_LIBERO_EVAL_OUTPUT", "outputs/libero/evaluation")),
    )
    parser.add_argument("--save-video", action="store_true", help="Save every trial as mp4.")
    parser.add_argument("--save-failed-video", action="store_true",
                        help="Save failed (and a few success) rollouts as mp4 for debugging.")
    parser.add_argument("--save-failed-max", type=int, default=8,
                        help="Max failed mp4s to keep per task.")
    parser.add_argument("--save-ok-max", type=int, default=2,
                        help="Max success mp4s to keep per task (for comparison).")
    parser.add_argument("--keep-videos-below", type=float, default=0.80,
                        help="If final success_rate >= this, delete saved mp4s. <0 keeps them.")
    parser.add_argument("--video-fps", type=int, default=20)
    args = parser.parse_args()
    write_libero_config()
    from libero.libero import benchmark
    base = f"http://{args.host}:{args.port}"
    for _ in range(60):
        try:
            if _request(f"{base}/health").get("ok"):
                break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5)
    else:
        raise SystemExit(f"policy server not ready at {base}")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    env, instruction = get_libero_env(task, seed=args.seed)
    init_states = suite.get_task_init_states(args.task_id)
    max_steps = int(args.max_steps or MAX_STEPS[args.suite])
    successes = []
    saved_fail, saved_ok = [], []
    video_dir = args.output_dir / "videos"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for local_trial in range(args.num_trials):
        trial = int(args.trial_start) + local_trial
        _request(f"{base}/reset", {})
        env.reset()
        obs = env.set_init_state(init_states[trial % len(init_states)])
        done = False
        for _ in range(args.settle_steps):
            obs, _, done, _ = env.step(list(DUMMY_ACTION))
        steps = 0
        actions, eefs, frames = [], [], []
        need_frames = args.save_video or (
            args.save_failed_video
            and (len(saved_fail) < int(args.save_failed_max) or len(saved_ok) < int(args.save_ok_max))
        )
        for steps in range(1, max_steps + 1):
            if done:
                break
            action = np.asarray(
                _request(f"{base}/act", _obs_payload(obs, instruction))["action"],
                dtype=np.float32,
            )
            if args.binarize_gripper:
                # The policy adapter has already converted native open-scale to
                # robosuite convention: -1=open, +1=close.
                action[6] = 1.0 if action[6] >= 0.0 else -1.0
            actions.append(np.asarray(action, dtype=np.float32))
            eefs.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
            if need_frames:
                xyz = np.asarray(action[:3], dtype=np.float32)
                frames.append(stitch_obs_frame(obs, [
                    f"{args.suite} t{args.task_id} trial={trial} step={steps}",
                    instruction[:90],
                    f"grip={float(action[6]):+.2f} dxyz={xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}",
                    f"eef={float(obs['robot0_eef_pos'][0]):+.3f},{float(obs['robot0_eef_pos'][1]):+.3f},{float(obs['robot0_eef_pos'][2]):+.3f}",
                ]))
            obs, _, done, _ = env.step(action)
            if steps == 1 or steps % 50 == 0:
                print(f"trial={trial} step={steps}/{max_steps} done={int(done)}", flush=True)
        successes.append(bool(done))
        stats = action_rollout_stats(actions, eefs)
        print(
            f"trial={trial} success={int(done)} steps={steps} running={float(np.mean(successes)):.3f} "
            f"xyz_abs_mean={stats.get('xyz_abs_mean')} travel={stats.get('eef_travel_m')}",
            flush=True,
        )
        keep_this = args.save_video or (
            args.save_failed_video and frames and (
                (not done and len(saved_fail) < int(args.save_failed_max))
                or (done and len(saved_ok) < int(args.save_ok_max))
            )
        )
        if keep_this and frames:
            tag = "ok" if done else "fail"
            stem = f"{args.suite}_task{args.task_id}_trial{trial:02d}_{tag}"
            mp4 = save_rollout_mp4(video_dir / f"{stem}.mp4", frames, fps=args.video_fps)
            npy = video_dir / f"{stem}_actions.npy"
            if actions:
                np.save(npy, np.stack(actions).astype(np.float32))
            meta = {
                "suite": args.suite, "task_id": args.task_id, "trial": trial,
                "success": bool(done), "instruction": instruction,
                "steps": steps, "max_steps": max_steps, "stats": stats,
                "ckpt": os.environ.get("W0_LIBERO_CHECKPOINT"), "mp4": str(mp4),
                "actions_npy": str(npy) if actions else None,
            }
            (video_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
            (saved_ok if done else saved_fail).append(str(mp4))
            print(f"saved {mp4}", flush=True)
    try:
        env.close()
    except Exception:
        pass
    rate = float(np.mean(successes)) if successes else 0.0
    kept_videos = saved_fail + saved_ok
    if (
        args.save_failed_video
        and float(args.keep_videos_below) >= 0
        and rate >= float(args.keep_videos_below)
        and video_dir.is_dir()
    ):
        for path in video_dir.glob("*"):
            try:
                path.unlink()
            except OSError:
                pass
        kept_videos = []
        print(f"dropped videos: success_rate={rate:.3f} >= {float(args.keep_videos_below):.3f}", flush=True)
    summary = {
        "suite": args.suite, "task_id": args.task_id, "instruction": instruction,
        "num_trials": args.num_trials, "trial_start": int(args.trial_start),
        "successes": int(np.sum(successes)),
        "success_rate": rate, "policy": base,
        "ckpt": os.environ.get("W0_LIBERO_CHECKPOINT"),
        "seed": args.seed, "settle_steps": args.settle_steps,
        "max_steps": max_steps, "binarize_gripper": bool(args.binarize_gripper),
        "failed_videos": saved_fail if kept_videos else [],
        "ok_videos": saved_ok if kept_videos else [],
    }
    out = args.output_dir / f"{args.suite}_task{args.task_id}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
