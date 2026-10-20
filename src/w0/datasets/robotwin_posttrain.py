"""RoboTwin LeRobot 50 Hz -> 12.5/25 Hz, native commands and train-only bounds."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from .domain_video_dataset import decode_native_frames
from .color_jitter import VideoColorJitter
from .lerobot.robot_video_dataset import DEFAULT_PROMPT
from .posttrain_temporal import PosttrainTemporalSampling

SLOTS = [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 22, 33]
JOINT_NAMES = [f"{side}_{name}" for side in ("left", "right") for name in
               ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")]
DOMAIN = {"name": "robotwin_wam/aloha", "dataset": "robotwin_wam", "embodiment": "aloha",
          "action_dim": 37, "proprio_dim": 37, "action_valid_indices": SLOTS,
          "proprio_valid_indices": SLOTS, "raw_joint_names": JOINT_NAMES,
          "action_semantics": "native_absolute_joint_command_interval_end"}
# Historical statistics contract, retained so existing train-only bounds load
# unchanged, including shared bounds for sampling-rate ablations. Statistics
# select train-only stride-4 reference positions; sampling a different stride
# does not refit these bounds. Effective timing is in domain_data_recipe.
RECIPE = {"version": 1, "source_fps": 50, "target_fps": 12.5, "stride": 4,
          "phases": [0, 1, 2, 3], "action_horizon": 64, "action_chunk_size": 16,
          "max_action_offset": 15, "video_plan_horizon": 96, "num_history_frames": 0,
          "short_episodes": "pad_mask", "slots": SLOTS, "domain": DOMAIN,
          "normalization": "bounds_q99_minmax_degenerate_clip5", "sigma": 1e-8,
          "camera_layout": "cover_center_crop_main_top_wrists_bottom_384x320",
          "text": "episode_first_instruction_default_prompt_128_wan22ti2v5b"}
CAMERAS = ["observation.images." + name for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")]


def is_validation(episode: int, seed: int, fraction: float) -> bool:
    value = int(hashlib.sha256(f"robotwin_wam:{seed}:{episode}".encode()).hexdigest()[:16], 16)
    return value / 2**64 < fraction


def read_metadata(root: Path) -> tuple[dict, list[dict], str]:
    info_bytes = (root / "meta/info.json").read_bytes()
    info = json.loads(info_bytes)
    if info["fps"] != 50:
        raise ValueError("This recipe requires recorded 50 Hz data.")
    for key in ("action", "observation.state"):
        feature = info["features"][key]
        names = feature["names"]
        if names and isinstance(names[0], list):
            names = names[0]
        if feature["shape"] != [14] or names != JOINT_NAMES:
            raise ValueError(f"Unexpected RoboTwin joint layout in {key}")
    digest = hashlib.sha256(info_bytes)
    episodes = []
    with (root / "meta/episodes.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            record = json.loads(line)
            if record["length"] >= 5:
                episodes.append({"episode_index": record["episode_index"], "length": record["length"],
                                 "instruction": record["tasks"][0]})
    return info, episodes, digest.hexdigest()


def read_numeric(root: Path, info: dict, episode: dict) -> tuple[np.ndarray, np.ndarray]:
    index = episode["episode_index"]
    path = root / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
    table = pq.read_table(path, columns=["action", "observation.state", "timestamp"])
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    timestamps = table["timestamp"].to_numpy()
    if (action.shape != (episode["length"], 14) or state.shape != action.shape or
            not np.isfinite(action).all() or not np.isfinite(state).all() or
            not np.allclose(timestamps, np.arange(len(table)) / 50, rtol=0, atol=1e-4)):
        raise ValueError(f"Invalid action/state/timestamps: {path}")
    return action, state


def camera_canvas(images: list[torch.Tensor]) -> torch.Tensor:
    """Three [N,3,H,W] uint8 cameras -> [N,3,384,320] normalized canvas."""
    tiles = []
    for images_one, (height, width) in zip(images, [(256, 320), (128, 160), (128, 160)]):
        scale = max(height / images_one.shape[-2], width / images_one.shape[-1])
        resized = F.interpolate(images_one.float(), size=(math.ceil(images_one.shape[-2] * scale),
                               math.ceil(images_one.shape[-1] * scale)), mode="bilinear", align_corners=False)
        top, left = (resized.shape[-2] - height) // 2, (resized.shape[-1] - width) // 2
        tiles.append(resized[..., top:top + height, left:left + width] / 127.5 - 1)
    return torch.cat([tiles[0], torch.cat(tiles[1:], dim=-1)], dim=-2)


class RobotwinPosttrainProcessor:
    """Normalize native joints first, scatter into 37D; invert only real slots."""
    action_output_dim = proprio_output_dim = 37
    recipe = RECIPE

    def __init__(self, stats_path: str):
        self.stats_path = str(stats_path)
        self.set_normalizer_from_stats(json.loads(Path(stats_path).read_text()))

    def set_normalizer_from_stats(self, stats: dict) -> None:
        if stats.get("recipe") != self.recipe:
            raise ValueError("Statistics do not match the RoboTwin posttrain recipe.")
        self.statistics = stats
        self.bounds = {}
        for kind in ("action", "proprio"):
            entry = stats[kind]
            low, high = torch.tensor(entry["q01"]), torch.tensor(entry["q99"])
            for index, bounds in entry["minmax_bounds"].items():
                low[int(index)], high[int(index)] = bounds
            if (low.shape != (14,) or high.shape != (14,) or not torch.isfinite(low).all()
                    or not torch.isfinite(high).all() or (high < low).any()):
                raise ValueError(f"Invalid {kind} normalization bounds.")
            self.bounds[kind] = low, high

    def eval(self):
        return self

    def pack(self, native: torch.Tensor, kind: str, valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        native = native.float()
        if native.shape[-1] != 14:
            raise ValueError("Expected 14 native joint/gripper values.")
        valid = torch.ones_like(native, dtype=torch.bool) if valid is None else valid.expand_as(native)
        if not torch.isfinite(native.masked_select(valid)).all():
            raise ValueError("Nonfinite valid robot signal.")
        low, high = [v.to(native) for v in self.bounds[kind]]
        normalized = (2 * (native.masked_fill(~valid, 0) - low) / (high - low + 1e-8) - 1).clamp(-5, 5)
        packed = native.new_zeros(*native.shape[:-1], 37)
        mask = torch.zeros_like(packed, dtype=torch.bool)
        packed[..., SLOTS] = normalized.masked_fill(~valid, 0)
        mask[..., SLOTS] = valid
        return packed, mask

    def unpack_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != 37:
            raise ValueError("Expected 37 action slots.")
        low, high = [v.to(action) for v in self.bounds["action"]]
        return (action[..., SLOTS] + 1) * (high - low + 1e-8) / 2 + low


class RobotwinPosttrainDataset(PosttrainTemporalSampling, torch.utils.data.Dataset):
    uses_epoch_indices = True
    domain_ids, domain_id = [0], 0
    action_dim = proprio_dim = 37

    def __init__(self, dataset_root: str, processor: RobotwinPosttrainProcessor,
                 is_training_set: bool = True, samples_per_epoch: int | None = 100000,
                 seed: int = 42, val_set_proportion: float = 0.01,
                 pretrained_norm_stats: str | None = None, *, action_horizon: int = 64,
                 action_chunk_size: int = 16, max_action_offset: int = 15,
                 video_plan_horizon: int = 96, action_video_freq_ratio: int = 8,
                 epoch_sampling: str = "random",
                 sampling_stride: int = 4,
                 color_jitter=None, text_embedding_cache_dir: str | None = None):
        if isinstance(sampling_stride, bool) or not isinstance(sampling_stride, Integral) or sampling_stride not in (2, 4):
            raise ValueError("sampling_stride must be 2 (25 Hz) or 4 (12.5 Hz).")
        self.sampling_stride = int(sampling_stride)
        self._configure_temporal_sampling(
            action_horizon=action_horizon, action_chunk_size=action_chunk_size,
            max_action_offset=max_action_offset, video_plan_horizon=video_plan_horizon,
            action_video_freq_ratio=action_video_freq_ratio,
        )
        self.color_jitter = VideoColorJitter(color_jitter, training=bool(is_training_set))
        self.root, self.processor = Path(dataset_root), processor
        self.text_embedding_cache_dir = (
            Path(text_embedding_cache_dir)
            if text_embedding_cache_dir is not None
            else self.root / "text_embeds_cache"
        )
        self.info, episodes, fingerprint = read_metadata(self.root)
        self.seed = int(seed)
        if epoch_sampling not in ("random", "all_valid_windows"):
            raise ValueError("epoch_sampling must be random or all_valid_windows.")
        self.epoch_sampling = epoch_sampling
        self.is_training_set = bool(is_training_set)
        self.domain_configs = [json.loads(json.dumps(DOMAIN))]
        stats = processor.statistics
        if (stats["metadata_sha256"] != fingerprint or stats["split_seed"] != seed or
                stats["val_set_proportion"] != val_set_proportion):
            raise ValueError("Statistics metadata or trajectory split does not match dataset.")
        self.episodes = [e for e in episodes if is_validation(e["episode_index"], seed, val_set_proportion) != is_training_set]
        if not self.episodes:
            raise ValueError("Empty dataset split.")
        if epoch_sampling == "all_valid_windows":
            if samples_per_epoch is not None:
                raise ValueError("all_valid_windows requires samples_per_epoch=null.")
            # Union of all sampling phases: every native anchor i with
            # i + stride < length contributes one window. Tails are masked.
            self.cumulative_windows = np.cumsum(
                [e["length"] - self.sampling_stride for e in self.episodes], dtype=np.int64)
            self.samples_per_epoch = int(self.cumulative_windows[-1])
        else:
            if (isinstance(samples_per_epoch, bool) or not isinstance(samples_per_epoch, int)
                    or samples_per_epoch < 1):
                raise ValueError("Random sampling requires a positive integer samples_per_epoch.")
            self.samples_per_epoch = samples_per_epoch
        self.domain_data_recipe = dict(self._sampling_recipe(RECIPE), metadata_sha256=fingerprint, split_seed=seed,
                                      stride=self.sampling_stride, target_fps=50 / self.sampling_stride,
                                      phases=list(range(self.sampling_stride)),
                                      val_set_proportion=val_set_proportion,
                                      stats_sha256=hashlib.sha256(Path(processor.stats_path).read_bytes()).hexdigest(),
                                      anchor_sampling=(("all_valid_native_anchors_four_phases_v1" if self.sampling_stride == 4
                                                        else "all_valid_native_anchors_two_phases_v1")
                                                       if epoch_sampling == "all_valid_windows"
                                                       else "prefer_full_horizon_v1"),
                                      samples_per_epoch=self.samples_per_epoch)
        self.domain_data_recipe.update(self.color_jitter.recipe())
        self._numeric_cache, self._text_cache = OrderedDict(), OrderedDict()

    def __len__(self):
        return self.samples_per_epoch

    def configure_video_history_memory(self, *, num_history_frames):
        if num_history_frames != 0:
            raise ValueError("RoboTwin posttrain requires num_history_frames=0.")

    def _numeric(self, episode):
        key = episode["episode_index"]
        if key not in self._numeric_cache:
            self._numeric_cache[key] = read_numeric(self.root, self.info, episode)
            if len(self._numeric_cache) > 2:
                self._numeric_cache.popitem(last=False)
        return self._numeric_cache[key]

    def _text(self, instruction):
        prompt = DEFAULT_PROMPT.format(task=instruction)
        if prompt not in self._text_cache:
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            path = self.text_embedding_cache_dir / f"{digest}.t5_len128.wan22ti2v5b.pt"
            value = torch.load(path, map_location="cpu", weights_only=True)
            context, mask = value["context"].clone(), value["mask"].bool()
            if context.shape != (128, 4096) or mask.shape != (128,):
                raise ValueError(f"Invalid text embedding: {path}")
            context[~mask] = 0
            self._text_cache[prompt] = (context, torch.ones_like(mask))
            if len(self._text_cache) > 32:
                self._text_cache.popitem(last=False)
        return prompt, *self._text_cache[prompt]

    def __getitem__(self, index):
        index, epoch = index if isinstance(index, tuple) else (index, 0)
        if not 0 <= index < len(self):
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch), int(index)]))
        stride = self.sampling_stride
        if self.epoch_sampling == "all_valid_windows":
            ep_index = int(np.searchsorted(self.cumulative_windows, index, side="right"))
            episode = self.episodes[ep_index]
            previous = int(self.cumulative_windows[ep_index - 1]) if ep_index else 0
            anchor = int(index) - previous
            phase, start = anchor % stride, anchor // stride
            intervals = (episode["length"] - 1 - phase) // stride
        else:
            episode = self.episodes[int(rng.integers(len(self.episodes)))]
            phase = int(rng.integers(min(stride, episode["length"] - stride)))
            intervals = (episode["length"] - 1 - phase) // stride
            start = int(rng.integers(max(1, intervals - self.action_horizon + 1)))
        raw_action, raw_state = self._numeric(episode)
        length = len(raw_action)
        offset = self._sample_action_offset(rng, intervals - start)
        ticks = start + np.arange(self.action_candidate_length)
        source = phase + stride * ticks
        # i -> j: recorded command at j-1, proprio at i.
        action_pad = source + stride >= length
        state_pad = source >= length
        action, action_mask = self.processor.pack(torch.from_numpy(raw_action[np.minimum(source + stride - 1, length - 1)]),
                                                  "action", torch.from_numpy(~action_pad)[:, None])
        state, state_mask = self.processor.pack(torch.from_numpy(raw_state[np.minimum(source, length - 1)]),
                                                "proprio", torch.from_numpy(~state_pad)[:, None])
        video_positions = phase + stride * (start + self.video_sample_indices)
        chunk_positions = phase + stride * (start + offset + self._chunk_start_offsets)
        all_positions = np.minimum(np.concatenate([video_positions, chunk_positions]), length - 1)
        unique, inverse = np.unique(all_positions, return_inverse=True)
        cameras = []
        ep = episode["episode_index"]
        for camera in CAMERAS:
            path = self.root / self.info["video_path"].format(
                episode_chunk=ep // self.info["chunks_size"], episode_index=ep, video_key=camera)
            cameras.append(decode_native_frames(path, unique, 50))
        canvas = camera_canvas(cameras)[torch.from_numpy(inverse)]
        canvas = self.color_jitter(canvas, seed=self.seed, epoch=epoch, index=index)
        prompt, context, context_mask = self._text(episode["instruction"])
        video_count = len(self.video_sample_indices)
        # Keep an explicit zero offset when disabled: the action/video timelines
        # can have different horizons and chunk observations are sampled exactly.
        return {"video": canvas[:video_count].permute(1, 0, 2, 3), "chunk_obs_images": canvas[video_count:],
                "action": action, "proprio": state, "action_valid_mask": action_mask,
                "proprio_valid_mask": state_mask, "action_is_pad": torch.from_numpy(action_pad),
                "proprio_is_pad": torch.from_numpy(state_pad),
                "image_is_pad": torch.from_numpy(video_positions >= length),
                "domain_id": torch.tensor(0), "action_offset": torch.tensor(offset),
                "action_video_freq_ratio": torch.tensor(self.action_video_freq_ratio), "prompt": prompt,
                "context": context, "context_mask": context_mask}

    def denormalize_action(self, action, domain_id):
        if int(domain_id) != 0:
            raise ValueError("This dataset has only target domain 0.")
        return self.processor.unpack_action(action)
