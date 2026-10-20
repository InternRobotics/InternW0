"""LIBERO env helpers for w0 posttrain evaluation."""
from __future__ import annotations

import base64
import io
import os
import pathlib
import time

import numpy as np

LIBERO_REPO = pathlib.Path(os.environ.get("LIBERO_PATH", "external/LIBERO"))
LIBERO_CONFIG_ROOT = pathlib.Path(
    os.environ.get("LIBERO_CONFIG_PATH", ".cache/libero_config")
)


def write_libero_config(repo: pathlib.Path | None = None) -> pathlib.Path:
    repo = pathlib.Path(repo or LIBERO_REPO).resolve()
    benchmark_root = repo / "libero" / "libero"
    LIBERO_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LIBERO_CONFIG_ROOT / "config.yaml"
    path.write_text(
        "\n".join([
            f"benchmark_root: {benchmark_root}",
            f"bddl_files: {benchmark_root / 'bddl_files'}",
            f"init_states: {benchmark_root / 'init_files'}",
            f"datasets: {repo / 'datasets'}",
            f"assets: {benchmark_root / 'assets'}",
            "",
        ])
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_CONFIG_ROOT)
    return path


def encode_image_b64(image: np.ndarray) -> str:
    from PIL import Image
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    codec = str(os.environ.get("W0_LIBERO_IMAGE_CODEC", "JPEG")).strip().upper()
    if codec == "PNG":
        Image.fromarray(array).save(buffer, format="PNG")
    else:
        Image.fromarray(array).save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image_b64(payload: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB"))

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
LIBERO_ENV_RESOLUTION = 256
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
MAX_STEPS = {
    "libero_spatial": 600,
    "libero_object": 600,
    "libero_goal": 600,
    "libero_10": 700,
}


def get_libero_env(task, resolution=LIBERO_ENV_RESOLUTION, seed=42, env_num=1):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

    write_libero_config()
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    if env_num > 1:
        env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
    else:
        env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def rotate_180_hwc(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


def get_libero_images(obs: dict, rotate: bool = True) -> dict[str, np.ndarray]:
    head = np.asarray(obs["agentview_image"])
    wrist = np.asarray(obs["robot0_eye_in_hand_image"])
    if rotate:
        head, wrist = rotate_180_hwc(head), rotate_180_hwc(wrist)
    return {"image": head, "wrist_image": wrist}


def _to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def annotate_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw
    image = Image.fromarray(_to_uint8(frame))
    draw = ImageDraw.Draw(image)
    y = 6
    for line in lines:
        draw.text((8, y), line, fill=(255, 255, 0))
        y += 14
    return np.asarray(image)


def stitch_obs_frame(obs: dict, lines: list[str] | None = None, rotate: bool = True) -> np.ndarray:
    views = get_libero_images(obs, rotate=rotate)
    frame = np.concatenate([_to_uint8(views["image"]), _to_uint8(views["wrist_image"])], axis=1)
    if lines:
        frame = annotate_frame(frame, lines)
    return frame


def save_rollout_mp4(path: pathlib.Path, frames: list[np.ndarray], fps: int = 20) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError(f"No frames to write: {path}")
    try:
        import imageio
        writer = imageio.get_writer(str(path), fps=fps)
        for frame in frames:
            writer.append_data(_to_uint8(frame))
        writer.close()
        return path
    except Exception:
        import cv2
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {path}")
        for frame in frames:
            writer.write(cv2.cvtColor(_to_uint8(frame), cv2.COLOR_RGB2BGR))
        writer.release()
        return path


def action_rollout_stats(actions: list, eefs: list) -> dict:
    action = np.asarray(actions, dtype=np.float32)
    eef = np.asarray(eefs, dtype=np.float32)
    if action.size == 0:
        return {}
    travel = float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum()) if len(eef) > 1 else 0.0
    return {
        "steps": int(len(action)),
        "xyz_abs_mean": np.abs(action[:, :3]).mean(axis=0).tolist(),
        "xyz_abs_max": np.abs(action[:, :3]).max(axis=0).tolist(),
        "rot_abs_mean": np.abs(action[:, 3:6]).mean(axis=0).tolist(),
        "rot_abs_max": np.abs(action[:, 3:6]).max(axis=0).tolist(),
        "gripper_mean": float(action[:, 6].mean()),
        "gripper_min": float(action[:, 6].min()),
        "gripper_max": float(action[:, 6].max()),
        "eef_start": eef[0].tolist() if len(eef) else [],
        "eef_end": eef[-1].tolist() if len(eef) else [],
        "eef_net_m": float(np.linalg.norm(eef[-1] - eef[0])) if len(eef) else 0.0,
        "eef_travel_m": travel,
    }
