"""LIBERO posttraining: native EEF10 scattered into the 37D left-arm slots."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .color_jitter import VideoColorJitter
from .domain_video_dataset import decode_native_frames
from .lerobot.robot_video_dataset import DEFAULT_PROMPT
from .libero_eef import (
    EEF10_DIM, EEF10_NAMES, GRIPPER_CONVENTION, GRIPPER_SOURCE_OSC7, OUTPUT_REPRESENTATION, ROT6D_DIMS,
    SLOTS, osc7_to_eef10, state8_to_eef10,
)
from .posttrain_temporal import PosttrainTemporalSampling
from .robotwin_posttrain import camera_canvas

CAMERA_CANDIDATES = (
    ("observation.images.image", "observation.images.wrist_image"),
    ("observation.images.image", "observation.images.image2"),
    ("image", "wrist_image"),
    ("observation.images.agentview_image", "observation.images.eye_in_hand_image"),
)
DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
RIGHT_WRIST_EXCLUSION = torch.zeros(384, 320, dtype=torch.bool)
RIGHT_WRIST_EXCLUSION[256:, 160:] = True
DOMAIN = {
    "name": "libero/franka", "dataset": "libero", "embodiment": "franka",
    "action_dim": 37, "proprio_dim": 37, "action_valid_indices": SLOTS,
    "proprio_valid_indices": SLOTS, "raw_eef_names": EEF10_NAMES,
    "action_semantics": "native_delta_eef10_row_aligned",
}
RECIPE = {
    "version": 2, "source_fps": 20, "target_fps": 20, "stride": 1, "phases": [0],
    "action_horizon": 64, "action_chunk_size": 16, "max_action_offset": 15,
    "video_plan_horizon": 96, "num_history_frames": 0, "short_episodes": "pad_mask",
    "slots": SLOTS, "domain": DOMAIN, "rot6d_identity_dims": list(ROT6D_DIMS),
    "gripper_convention": GRIPPER_CONVENTION, "gripper_source_osc7": GRIPPER_SOURCE_OSC7,
    "output_representation": OUTPUT_REPRESENTATION,
    "normalization": "bounds_q99_minmax_degenerate_clip5_rot6d_identity", "sigma": 1e-8,
    "camera_layout": "disk_native_cover_center_crop_main_top_wrist_bottom_left_384x320",
    "text": "episode_first_instruction_default_prompt_128_wan22ti2v5b",
}


SUITE_ALIASES = {
    "goal": "libero_goal",
    "libero_goal": "libero_goal",
    "long": "libero_10",
    "libero_long": "libero_10",
    "libero_10": "libero_10",
}


def is_validation(suite: str, episode: int, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"libero:{seed}:{suite}:{episode}".encode()).hexdigest()
    return int(digest[:16], 16) / 2**64 < fraction


def suite_allowed(suite: str, include_suites: list[str] | None) -> bool:
    """Keep stats fingerprint on all suites; optionally sample only some of them."""
    if not include_suites:
        return True
    tokens = [SUITE_ALIASES.get(str(token), str(token)) for token in include_suites]
    return any(token in suite for token in tokens)


def full_window_counts(episodes: list[dict], action_horizon: int) -> np.ndarray:
    """Number of non-padded action windows contributed by each episode."""
    return np.asarray([
        max(1, int(episode["length"]) - int(action_horizon) + 1)
        for episode in episodes
    ], dtype=np.int64)


def frame_counts(episodes: list[dict]) -> np.ndarray:
    """Number of frame anchors contributed by each episode."""
    return np.asarray([max(1, int(episode["length"])) for episode in episodes], dtype=np.int64)


def locate_global_window(cumulative: np.ndarray, global_window: int) -> tuple[int, int]:
    """Map a global window index to (episode index, episode-local start)."""
    if cumulative.ndim != 1 or not len(cumulative) or int(cumulative[-1]) <= 0:
        raise ValueError("cumulative window counts must be a non-empty positive vector")
    if not 0 <= int(global_window) < int(cumulative[-1]):
        raise IndexError(global_window)
    episode_index = int(np.searchsorted(cumulative, int(global_window), side="right"))
    previous = int(cumulative[episode_index - 1]) if episode_index else 0
    return episode_index, int(global_window) - previous


def discover_roots(root: Path) -> list[Path]:
    root = Path(root)
    if (root / "meta/info.json").is_file():
        return [root]
    suites = sorted(
        path for path in root.iterdir()
        if path.is_dir() and (path / "meta/info.json").is_file()
    )
    if suites:
        return suites
    raise FileNotFoundError(
        f"No LIBERO LeRobot dataset at {root}. Point dataset_root at a converted "
        "native_delta_eef10 directory or a parent of the four suite folders."
    )


def _feature_shape(info: dict, key: str) -> tuple[int, ...]:
    feature = info["features"][key]
    shape = feature.get("shape") or feature.get("info", {}).get("shape")
    return tuple(int(v) for v in shape)


def _camera_keys(info: dict) -> tuple[str, str]:
    features = info.get("features", {})
    for main, wrist in CAMERA_CANDIDATES:
        if main in features and wrist in features:
            return main, wrist
    raise ValueError(f"Could not find LIBERO head/wrist cameras in {sorted(features)}")


def _read_table(path: Path, columns: list[str]):
    try:
        return pq.read_table(path, columns=columns)
    except Exception as exc:
        if "Dot path" not in str(exc):
            raise
        return pq.read_table(path)


def file_local_offsets(chunk, file_index, length, from_index) -> np.ndarray:
    chunk, file_index, length, from_index = [np.asarray(v) for v in (chunk, file_index, length, from_index)]
    order = np.lexsort((from_index, file_index, chunk))
    offsets = np.empty(len(length), dtype=np.int64)
    running: dict[tuple[int, int], int] = {}
    for index in order:
        key = (int(chunk[index]), int(file_index[index]))
        offsets[index] = running.get(key, 0)
        running[key] = offsets[index] + int(length[index])
    return offsets


def _v3_task_map(root: Path) -> dict[int, str]:
    path = root / "meta/tasks.parquet"
    if not path.is_file():
        return {}
    table = pq.read_table(path)
    names = set(table.column_names)
    task_index = table.column("task_index").to_pylist() if "task_index" in names else list(range(len(table)))
    if "task" in names:
        texts = table.column("task").to_pylist()
    elif "instruction" in names:
        texts = table.column("instruction").to_pylist()
    else:
        texts = table.to_pandas().index.to_list()
    mapping = {}
    for index, text in zip(task_index, texts):
        if text in (None, ""):
            continue
        mapping[int(index)] = str(text)
    return mapping


def _episode_instruction(record: dict, task_map: dict[int, str]) -> str | None:
    tasks = record.get("tasks") or record.get("task") or record.get("instruction")
    if isinstance(tasks, (list, tuple, np.ndarray)) and len(tasks) and tasks[0] not in (None, ""):
        return str(tasks[0])
    if isinstance(tasks, str) and tasks:
        return tasks
    if "task_index" in record and int(record["task_index"]) in task_map:
        return task_map[int(record["task_index"])]
    return None


def _representation(root: Path, info: dict) -> str:
    conversion_path = root / "meta/conversion.json"
    if conversion_path.is_file():
        conversion = json.loads(conversion_path.read_text())
        declared = conversion.get("output_representation")
        if declared != OUTPUT_REPRESENTATION:
            raise ValueError(
                f"{conversion_path} declares {declared!r}, expected {OUTPUT_REPRESENTATION!r}"
            )
        return OUTPUT_REPRESENTATION
    action_dim = _feature_shape(info, "action")[-1]
    if action_dim == EEF10_DIM:
        raise ValueError(
            f"{root} has 10-D actions but no meta/conversion.json; "
            "refuse to guess the rotation/gripper convention."
        )
    if action_dim == 7:
        return "libero_osc7"
    raise ValueError(f"Unsupported LIBERO action dim {action_dim} in {root}")


def _v2_episodes(root: Path, info: dict, cameras: tuple[str, str], representation: str, digest) -> list[dict]:
    episodes = []
    with (root / "meta/episodes.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            record = json.loads(line)
            length = record.get("length") or record.get("episode_length")
            instruction = _episode_instruction(record, {})
            if length is None or not instruction:
                raise ValueError(f"Incomplete episode record in {root}: {record}")
            if int(length) >= 2:
                episodes.append({
                    "episode_index": int(record["episode_index"]),
                    "length": int(length),
                    "instruction": instruction,
                    "suite": root.name,
                    "root": str(root),
                    "cameras": cameras,
                    "representation": representation,
                    "layout": "v2",
                })
    return episodes


def _v3_episode_paths(root: Path) -> list[Path]:
    paths = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not paths and (root / "meta/episodes.parquet").is_file():
        paths = [root / "meta/episodes.parquet"]
    if not paths:
        raise FileNotFoundError(f"No LeRobot episode index under {root}/meta")
    return paths


def _v3_episodes(root: Path, info: dict, cameras: tuple[str, str], representation: str, digest) -> list[dict]:
    frames = []
    for path in _v3_episode_paths(root):
        digest.update(path.read_bytes())
        frames.append(pq.read_table(path).to_pandas())
    import pandas as pd
    table = pd.concat(frames, ignore_index=True).sort_values("episode_index").reset_index(drop=True)
    task_map = _v3_task_map(root)
    tasks_path = root / "meta/tasks.parquet"
    if tasks_path.is_file():
        digest.update(tasks_path.read_bytes())
    if "data/chunk_index" not in table.columns or "data/file_index" not in table.columns:
        raise ValueError(f"{root} looks like LeRobot v3 but is missing data shard coordinates.")
    from_index = table["dataset_from_index"].to_numpy() if "dataset_from_index" in table.columns else table["episode_index"].to_numpy()
    data_offsets = file_local_offsets(
        table["data/chunk_index"].to_numpy(), table["data/file_index"].to_numpy(),
        table["length"].to_numpy(), from_index)
    video_meta = {camera: {} for camera in cameras}
    for camera in cameras:
        chunk_col, file_col = f"videos/{camera}/chunk_index", f"videos/{camera}/file_index"
        if chunk_col in table.columns and file_col in table.columns:
            offsets = file_local_offsets(
                table[chunk_col].to_numpy(), table[file_col].to_numpy(),
                table["length"].to_numpy(), from_index)
            for row, offset in enumerate(offsets):
                video_meta[camera][row] = {
                    "chunk": int(table.iloc[row][chunk_col]),
                    "file": int(table.iloc[row][file_col]),
                    "offset": int(offset),
                }
    episodes = []
    for row_id, record in table.iterrows():
        payload = record.to_dict()
        length = payload.get("length") or payload.get("episode_length")
        instruction = _episode_instruction(payload, task_map)
        if length is None or not instruction:
            raise ValueError(f"Incomplete v3 episode record in {root}: episode_index={payload.get('episode_index')}")
        if int(length) < 2:
            continue
        cameras_meta = {}
        for camera in cameras:
            cameras_meta[camera] = video_meta[camera].get(int(row_id), {"chunk": 0, "file": 0, "offset": 0})
        episodes.append({
            "episode_index": int(payload["episode_index"]),
            "length": int(length),
            "instruction": instruction,
            "suite": root.name,
            "root": str(root),
            "cameras": cameras,
            "representation": representation,
            "layout": "v3",
            "data_chunk": int(payload["data/chunk_index"]),
            "data_file": int(payload["data/file_index"]),
            "data_row_offset": int(data_offsets[int(row_id)]),
            "video_meta": cameras_meta,
        })
    return episodes


def read_metadata(root: Path) -> tuple[dict, list[dict], str, str]:
    info_bytes = (root / "meta/info.json").read_bytes()
    info = json.loads(info_bytes)
    fps = info.get("fps")
    if abs(float(fps) - 20) > 1e-6:
        raise ValueError(f"{root} must be recorded at 20 Hz, got {fps}")
    representation = _representation(root, info)
    cameras = _camera_keys(info)
    digest = hashlib.sha256(info_bytes)
    conversion = root / "meta/conversion.json"
    if conversion.is_file():
        digest.update(conversion.read_bytes())
    if (root / "meta/episodes.jsonl").is_file():
        episodes = _v2_episodes(root, info, cameras, representation, digest)
    else:
        episodes = _v3_episodes(root, info, cameras, representation, digest)
    return info, episodes, digest.hexdigest(), representation


def _parquet_path(root: Path, info: dict, episode: dict) -> Path:
    if episode.get("layout") == "v3":
        template = info.get("data_path") or "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        return root / template.format(chunk_index=episode["data_chunk"], file_index=episode["data_file"])
    index = episode["episode_index"]
    chunk_size = int(info.get("chunks_size") or 1000)
    template = info.get("data_path") or DEFAULT_DATA_PATH
    return root / template.format(episode_chunk=index // chunk_size, episode_index=index)


def _as_array(column, length: int, dim: int) -> np.ndarray:
    values = np.asarray(column.to_pylist(), dtype=np.float32)
    if values.ndim == 1:
        values = np.stack(column.to_pylist()).astype(np.float32)
    if values.shape != (length, dim):
        raise ValueError(f"Expected ({length}, {dim}), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite robot signal")
    return values


def read_numeric(root: Path, info: dict, episode: dict) -> tuple[np.ndarray, np.ndarray]:
    path = _parquet_path(root, info, episode)
    table = _read_table(path, ["action", "observation.state"])
    if episode.get("layout") == "v3":
        table = table.slice(int(episode["data_row_offset"]), int(episode["length"]))
    length = episode["length"]
    action_dim = _feature_shape(info, "action")[-1]
    state_dim = _feature_shape(info, "observation.state")[-1]
    action = _as_array(table["action"], length, action_dim)
    state = _as_array(table["observation.state"], length, state_dim)
    if episode["representation"] == OUTPUT_REPRESENTATION:
        if action.shape[-1] != EEF10_DIM or state.shape[-1] != EEF10_DIM:
            raise ValueError(f"native_delta_eef10 requires 10-D action/state in {path}")
        return action, state
    if action.shape[-1] != 7:
        raise ValueError(f"7-D OSC conversion requires action dim 7, got {action.shape}")
    converted_action = np.stack([osc7_to_eef10(row) for row in action])
    if state.shape[-1] == EEF10_DIM:
        converted_state = state
    elif state.shape[-1] == 8:
        converted_state = np.stack([state8_to_eef10(row) for row in state])
    else:
        raise ValueError(f"Cannot convert state dim {state.shape[-1]} in {path}")
    return converted_action, converted_state


def rotate_180(images: torch.Tensor) -> torch.Tensor:
    return images.flip(-1).flip(-2)


class LiberoPosttrainProcessor:
    """Normalize xyz/gripper, keep rot6d identical, scatter into 37D slots 7-16."""
    action_output_dim = proprio_output_dim = 37
    recipe = RECIPE

    def __init__(self, stats_path: str):
        self.stats_path = str(stats_path)
        self.set_normalizer_from_stats(json.loads(Path(stats_path).read_text()))

    def set_normalizer_from_stats(self, stats: dict) -> None:
        if stats.get("recipe") != self.recipe:
            raise ValueError("Statistics do not match the LIBERO posttrain recipe.")
        self.statistics = stats
        self.bounds = {}
        for kind in ("action", "proprio"):
            entry = stats[kind]
            low, high = torch.tensor(entry["q01"]), torch.tensor(entry["q99"])
            for index, bounds in entry["minmax_bounds"].items():
                low[int(index)], high[int(index)] = bounds
            if (low.shape != (EEF10_DIM,) or high.shape != (EEF10_DIM,) or not torch.isfinite(low).all()
                    or not torch.isfinite(high).all() or (high < low).any()):
                raise ValueError(f"Invalid {kind} normalization bounds.")
            self.bounds[kind] = low, high

    def eval(self):
        return self

    def _normalize(self, native: torch.Tensor, kind: str) -> torch.Tensor:
        low, high = [v.to(native) for v in self.bounds[kind]]
        scaled = (2 * (native - low) / (high - low + 1e-8) - 1).clamp(-5, 5)
        scaled[..., list(ROT6D_DIMS)] = native[..., list(ROT6D_DIMS)]
        return scaled

    def pack(self, native: torch.Tensor, kind: str, valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        native = native.float()
        if native.shape[-1] != EEF10_DIM:
            raise ValueError("Expected 10 native EEF values.")
        valid = torch.ones_like(native, dtype=torch.bool) if valid is None else valid.expand_as(native)
        if not torch.isfinite(native.masked_select(valid)).all():
            raise ValueError("Nonfinite valid robot signal.")
        normalized = self._normalize(native.masked_fill(~valid, 0), kind).masked_fill(~valid, 0)
        packed = native.new_zeros(*native.shape[:-1], 37)
        mask = torch.zeros_like(packed, dtype=torch.bool)
        packed[..., SLOTS] = normalized
        mask[..., SLOTS] = valid
        return packed, mask

    def unpack_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != 37:
            raise ValueError("Expected 37 action slots.")
        native = action[..., SLOTS]
        low, high = [v.to(native) for v in self.bounds["action"]]
        restored = (native + 1) * (high - low + 1e-8) / 2 + low
        restored[..., list(ROT6D_DIMS)] = native[..., list(ROT6D_DIMS)]
        return restored


class LiberoPosttrainDataset(PosttrainTemporalSampling, torch.utils.data.Dataset):
    uses_epoch_indices = True
    domain_ids, domain_id = [0], 0
    action_dim = proprio_dim = 37

    def __init__(self, dataset_root: str, processor: LiberoPosttrainProcessor,
                 is_training_set: bool = True, samples_per_epoch: int | None = 100000,
                 seed: int = 42, val_set_proportion: float = 0.01,
                 pretrained_norm_stats: str | None = None, *, action_horizon: int = 64,
                 action_chunk_size: int = 16, max_action_offset: int = 15,
                 video_plan_horizon: int = 96, action_video_freq_ratio: int = 8,
                 epoch_sampling: str = "random_frame", text_embedding_cache_dir: str | None = None,
                 dataset_roots: list[str] | None = None, color_jitter=None,
                 include_suites: list[str] | None = None):
        self._configure_temporal_sampling(
            action_horizon=action_horizon, action_chunk_size=action_chunk_size,
            max_action_offset=max_action_offset, video_plan_horizon=video_plan_horizon,
            action_video_freq_ratio=action_video_freq_ratio,
        )
        self.color_jitter = VideoColorJitter(color_jitter, training=bool(is_training_set))
        roots = [Path(p) for p in (dataset_roots or [dataset_root])]
        discovered = []
        for root in roots:
            discovered.extend(discover_roots(root))
        if not discovered:
            raise FileNotFoundError("No LIBERO datasets discovered.")
        self.roots = discovered
        self.processor = processor
        self.seed = int(seed)
        self.is_training_set = bool(is_training_set)
        if epoch_sampling not in (
            "random_frame", "random_window", "random_episode", "all_valid_windows"
        ):
            raise ValueError(
                "epoch_sampling must be random_frame, random_window, random_episode, "
                "or all_valid_windows."
            )
        self.epoch_sampling = epoch_sampling
        self.domain_configs = [json.loads(json.dumps(DOMAIN))]
        episodes, fingerprint = [], hashlib.sha256()
        self.infos = {}
        representations = set()
        for root in self.roots:
            info, root_episodes, digest, representation = read_metadata(root)
            fingerprint.update(digest.encode())
            fingerprint.update(Path(root).name.encode())
            self.infos[str(root)] = info
            representations.add(representation)
            episodes.extend(root_episodes)
        if len(representations) != 1:
            raise ValueError(f"Mixed LIBERO representations: {sorted(representations)}")
        stats = processor.statistics
        if (stats["metadata_sha256"] != fingerprint.hexdigest() or stats["split_seed"] != seed or
                stats["val_set_proportion"] != val_set_proportion):
            raise ValueError("Statistics metadata or trajectory split does not match dataset.")
        self.include_suites = list(include_suites) if include_suites else None
        self.episodes = [
            e for e in episodes
            if is_validation(e["suite"], e["episode_index"], seed, val_set_proportion) != is_training_set
            and suite_allowed(e["suite"], self.include_suites)
        ]
        if not self.episodes:
            raise ValueError("Empty LIBERO split.")
        # W0 / LeRobot indexes every frame, including anchors whose future
        # targets require episode-end padding. Sampling episodes uniformly
        # underweights long manipulation trajectories (the exact failure mode
        # of LIBERO Goal drawer/rack and Long task 8). Keep exact frame counts
        # for W0 alignment and valid-window counts as an optional ablation.
        self.random_frame_counts = frame_counts(self.episodes)
        self.random_frame_cumulative = np.cumsum(self.random_frame_counts, dtype=np.int64)
        self.random_window_counts = full_window_counts(self.episodes, self.action_horizon)
        self.random_window_cumulative = np.cumsum(self.random_window_counts, dtype=np.int64)
        if epoch_sampling == "all_valid_windows":
            if samples_per_epoch is not None:
                raise ValueError("all_valid_windows requires samples_per_epoch=null.")
            self.cumulative_windows = np.cumsum(
                [e["length"] - 1 for e in self.episodes], dtype=np.int64)
            self.samples_per_epoch = int(self.cumulative_windows[-1])
        else:
            if (isinstance(samples_per_epoch, bool) or not isinstance(samples_per_epoch, int)
                    or samples_per_epoch < 1):
                raise ValueError("Random sampling requires a positive integer samples_per_epoch.")
            self.samples_per_epoch = samples_per_epoch
        cache_root = Path(text_embedding_cache_dir) if text_embedding_cache_dir else self.roots[0] / "text_embeds_cache"
        self.text_cache_dir = cache_root
        self.domain_data_recipe = dict(
            self._sampling_recipe(RECIPE), metadata_sha256=fingerprint.hexdigest(), split_seed=seed,
            val_set_proportion=val_set_proportion, source_representation=next(iter(representations)),
            stats_sha256=hashlib.sha256(Path(processor.stats_path).read_bytes()).hexdigest(),
            text_embedding_cache_dir=str(self.text_cache_dir),
            dataset_roots=[str(root) for root in self.roots],
            include_suites=self.include_suites,
            anchor_sampling=("all_valid_native_anchors_stride1_v1"
                             if epoch_sampling == "all_valid_windows"
                             else "random_frames_global_with_episode_end_padding_v1"
                             if epoch_sampling == "random_frame"
                             else "random_full_windows_global_v1"
                             if epoch_sampling == "random_window"
                             else "random_episode_then_full_window_v1"),
            samples_per_epoch=self.samples_per_epoch,
        )
        self.domain_data_recipe.update(self.color_jitter.recipe())
        self._numeric_cache, self._text_cache = OrderedDict(), OrderedDict()

    def __len__(self):
        return self.samples_per_epoch

    def configure_video_history_memory(self, *, num_history_frames):
        if num_history_frames != 0:
            raise ValueError("LIBERO posttrain requires num_history_frames=0.")

    def _numeric(self, episode):
        key = (episode["root"], episode["episode_index"])
        if key not in self._numeric_cache:
            root = Path(episode["root"])
            self._numeric_cache[key] = read_numeric(root, self.infos[episode["root"]], episode)
            if len(self._numeric_cache) > 2:
                self._numeric_cache.popitem(last=False)
        return self._numeric_cache[key]

    def _text(self, instruction):
        prompt = DEFAULT_PROMPT.format(task=instruction)
        if prompt not in self._text_cache:
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            path = self.text_cache_dir / f"{digest}.t5_len128.wan22ti2v5b.pt"
            value = torch.load(path, map_location="cpu", weights_only=True)
            context, mask = value["context"].clone(), value["mask"].bool()
            if context.shape != (128, 4096) or mask.shape != (128,):
                raise ValueError(f"Invalid text embedding: {path}")
            context[~mask] = 0
            self._text_cache[prompt] = (context, torch.ones_like(mask))
            if len(self._text_cache) > 32:
                self._text_cache.popitem(last=False)
        return prompt, *self._text_cache[prompt]

    def _video_path(self, episode, camera):
        info = self.infos[episode["root"]]
        root = Path(episode["root"])
        if episode.get("layout") == "v3":
            meta = episode["video_meta"][camera]
            template = info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            return root / template.format(video_key=camera, chunk_index=meta["chunk"], file_index=meta["file"])
        ep = episode["episode_index"]
        chunk_size = int(info.get("chunks_size") or 1000)
        template = info.get("video_path") or DEFAULT_VIDEO_PATH
        return root / template.format(episode_chunk=ep // chunk_size, episode_index=ep, video_key=camera)

    def __getitem__(self, index):
        index, epoch = index if isinstance(index, tuple) else (index, 0)
        if not 0 <= index < len(self):
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch), int(index)]))
        if self.epoch_sampling == "all_valid_windows":
            ep_index = int(np.searchsorted(self.cumulative_windows, index, side="right"))
            episode = self.episodes[ep_index]
            previous = int(self.cumulative_windows[ep_index - 1]) if ep_index else 0
            start = int(index) - previous
        elif self.epoch_sampling == "random_frame":
            global_frame = int(rng.integers(int(self.random_frame_cumulative[-1])))
            ep_index, start = locate_global_window(
                self.random_frame_cumulative, global_frame
            )
            episode = self.episodes[ep_index]
        elif self.epoch_sampling == "random_window":
            global_window = int(rng.integers(int(self.random_window_cumulative[-1])))
            ep_index, start = locate_global_window(
                self.random_window_cumulative, global_window
            )
            episode = self.episodes[ep_index]
        else:
            episode = self.episodes[int(rng.integers(len(self.episodes)))]
            start = int(rng.integers(max(1, episode["length"] - self.action_horizon + 1)))
        raw_action, raw_state = self._numeric(episode)
        length = len(raw_action)
        remaining = length - start
        offset = self._sample_action_offset(rng, remaining)
        ticks = start + np.arange(self.action_candidate_length)
        pad = ticks >= length
        action, action_mask = self.processor.pack(
            torch.from_numpy(raw_action[np.minimum(ticks, length - 1)]),
            "action", torch.from_numpy(~pad)[:, None])
        state, state_mask = self.processor.pack(
            torch.from_numpy(raw_state[np.minimum(ticks, length - 1)]),
            "proprio", torch.from_numpy(~pad)[:, None])
        video_positions = start + self.video_sample_indices
        chunk_positions = start + offset + self._chunk_start_offsets
        all_positions = np.minimum(np.concatenate([video_positions, chunk_positions]), length - 1)
        unique, inverse = np.unique(all_positions, return_inverse=True)
        cameras = []
        fps = 20
        for camera in episode["cameras"]:
            base = int(episode.get("video_meta", {}).get(camera, {}).get("offset", 0))
            frames = decode_native_frames(self._video_path(episode, camera), unique + base, fps)
            cameras.append(frames)
        cameras.append(torch.zeros_like(cameras[1]))
        canvas = camera_canvas(cameras)[torch.from_numpy(inverse)]
        canvas = self.color_jitter(
            canvas, seed=self.seed, epoch=epoch, index=index, exclusion_mask=RIGHT_WRIST_EXCLUSION)
        prompt, context, context_mask = self._text(episode["instruction"])
        video_count = len(self.video_sample_indices)
        return {
            "video": canvas[:video_count].permute(1, 0, 2, 3), "chunk_obs_images": canvas[video_count:],
            "action": action, "proprio": state, "action_valid_mask": action_mask,
            "proprio_valid_mask": state_mask, "action_is_pad": torch.from_numpy(pad),
            "proprio_is_pad": torch.from_numpy(pad),
            "image_is_pad": torch.from_numpy(video_positions >= length),
            "domain_id": torch.tensor(0), "action_offset": torch.tensor(offset),
            "action_video_freq_ratio": torch.tensor(self.action_video_freq_ratio),
            "prompt": prompt, "context": context, "context_mask": context_mask,
        }

    def denormalize_action(self, action, domain_id):
        if int(domain_id) != 0:
            raise ValueError("This dataset has only target domain 0.")
        return self.processor.unpack_action(action)


def collect_prompts(dataset_root: str, dataset_roots: list[str] | None = None) -> list[str]:
    roots = []
    for root in (dataset_roots or [dataset_root]):
        roots.extend(discover_roots(Path(root)))
    prompts, seen = [], set()
    for root in roots:
        _, episodes, _, _ = read_metadata(root)
        for episode in episodes:
            prompt = DEFAULT_PROMPT.format(task=episode["instruction"])
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
    return prompts


def metadata_fingerprint(dataset_root: str, dataset_roots: list[str] | None = None) -> str:
    digest = hashlib.sha256()
    roots = []
    for root in (dataset_roots or [dataset_root]):
        roots.extend(discover_roots(Path(root)))
    for root in roots:
        _, _, fingerprint, _ = read_metadata(root)
        digest.update(fingerprint.encode())
        digest.update(str(root).encode())
    return digest.hexdigest()
