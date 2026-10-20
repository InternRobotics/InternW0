"""Dataset adapter for the native RoboTwin Motus ``clean`` layout.

Each episode is stored as a packed three-camera MP4, a joint-position tensor,
100 prompt strings, and their precomputed UMT5 embeddings.  This adapter keeps
that source data untouched and presents the same sample contract as
``RobotVideoDataset``.
"""

from __future__ import annotations

import bisect
import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torchvision.transforms.functional as transforms_F
from torchvision.io import read_video
from w0.datasets.domain import validate_domain_id, domain_data_recipe

from w0.datasets.lerobot.utils.normalizer import (
    LinearNormalizer,
    load_dataset_stats_from_json,
)


@dataclass(frozen=True)
class _Episode:
    task_name: str
    episode_id: int
    video_path: Path
    qpos_path: Path
    prompts_path: Path
    embeddings_path: Path
    source_length: int
    target_length: int
    domain_id: Optional[int] = None


class MotusCleanVideoDataset(torch.utils.data.Dataset):
    """Read clean-setting RoboTwin episodes without converting them to LeRobot."""

    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames: int = 65,
        video_size=(384, 320),
        processor=None,
        pretrained_norm_stats: Optional[str] = None,
        val_set_proportion: float = 0.01,
        is_training_set: bool = False,
        global_sample_stride: int = 1,
        action_video_freq_ratio: int = 8,
        context_len: int = 128,
        source_fps: float = 30.0,
        target_fps: float = 50.0,
        prompt_sampling: str = "random",
        qpos_cache_size: int = 32,
        text_cache_size: int = 2,
        max_action_offset: int = 0,
        action_chunk_size: int = 16,
        action_horizon: int = 0,
        num_history_frames: int = 0,
        domain_id: Optional[int] = None,
        domain_ids=None,
        **_unused,
    ):
        if int(max_action_offset) != 0:
            raise ValueError("MotusCleanVideoDataset currently requires max_action_offset=0.")
        if int(num_history_frames) != 0:
            raise ValueError("MotusCleanVideoDataset currently requires num_history_frames=0.")
        if len(dataset_dirs) == 0:
            raise ValueError("At least one Motus clean dataset directory is required.")
        if pretrained_norm_stats in (None, "", "null"):
            raise ValueError("pretrained_norm_stats is required for Motus clean training.")
        if not (0.0 <= float(val_set_proportion) < 1.0):
            raise ValueError("val_set_proportion must be in [0, 1).")

        self.dataset_dirs = [Path(path) for path in dataset_dirs]
        self.domain_id = validate_domain_id(domain_id)
        if domain_ids is not None and domain_id is not None:
            raise ValueError("Specify domain_id or per-directory domain_ids, not both.")
        self.domain_ids = ([validate_domain_id(d) for d in domain_ids] if domain_ids is not None
                           else [self.domain_id] * len(self.dataset_dirs))
        if len(self.domain_ids) != len(self.dataset_dirs):
            raise ValueError("domain_ids must align with dataset_dirs.")
        if domain_ids is not None and any(d is None for d in self.domain_ids):
            raise ValueError("Every directory requires an explicit domain ID.")
        self.domain_data_recipe = domain_data_recipe(self.dataset_dirs, self.domain_ids, pretrained_norm_stats)
        self.shape_meta = shape_meta
        self.num_frames = int(num_frames)
        self.video_size = tuple(int(x) for x in video_size)
        self.global_sample_stride = int(global_sample_stride)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.context_len = int(context_len)
        self.source_fps = float(source_fps)
        self.target_fps = float(target_fps)
        self.is_training_set = bool(is_training_set)
        self.prompt_sampling = str(prompt_sampling)
        self.num_history_frames = 0
        self.max_action_offset = int(max_action_offset)
        self.action_chunk_size = int(action_chunk_size)
        self.action_horizon = int(action_horizon) or self.num_frames - 1
        self._action_offset_enabled = False
        if self.action_horizon != self.num_frames - 1:
            raise ValueError("Native clean training requires action_horizon=num_frames-1.")
        if min(self.source_fps, self.target_fps, self.global_sample_stride) <= 0:
            raise ValueError("Frame rates and sample stride must be positive.")
        if min(qpos_cache_size, text_cache_size) < 1:
            raise ValueError("Cache sizes must be positive.")
        self.video_sample_indices = list(
            range(0, self.num_frames, self.action_video_freq_ratio)
        )
        if (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError("num_frames-1 must be divisible by action_video_freq_ratio.")
        if len(self.video_sample_indices) < 2:
            raise ValueError("At least two video frames are required.")

        action_dim = int(shape_meta["action"][0]["shape"])
        state_dim = int(shape_meta["state"][0]["shape"])
        if action_dim != 14 or state_dim != 14:
            raise ValueError(
                f"Native Motus qpos has 14 dimensions, got action={action_dim}, state={state_dim}."
            )

        stats = load_dataset_stats_from_json(str(pretrained_norm_stats))
        self.normalizer = LinearNormalizer(
            shape_meta=shape_meta,
            use_stepwise_action_norm=False,
            default_mode="z-score",
            exception_mode=None,
            stats=stats,
        )
        if processor is not None:
            processor.set_normalizer_from_stats(stats)
            processor.train() if self.is_training_set else processor.eval()
            self.normalizer = processor.normalizer
            # Existing evaluation uses the dataset's processor to invert normalization.
            self.lerobot_dataset = SimpleNamespace(processor=processor)
        self._qpos_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self._qpos_cache_size = int(qpos_cache_size)
        self._text_cache: OrderedDict[Path, tuple[list[str], list[torch.Tensor]]] = OrderedDict()
        self._text_cache_size = int(text_cache_size)

        all_episodes = self._discover_episodes()
        rng = random.Random(42)
        rng.shuffle(all_episodes)
        split = int(len(all_episodes) * (1.0 - float(val_set_proportion)))
        if float(val_set_proportion) < 1e-6:
            selected = all_episodes
        elif self.is_training_set:
            selected = all_episodes[:split]
        else:
            selected = all_episodes[split:]
        if not selected:
            raise ValueError("The requested Motus clean train/validation split is empty.")
        self.episodes = selected
        self._cumulative_lengths = []
        total = 0
        for episode in self.episodes:
            total += episode.target_length
            self._cumulative_lengths.append(total)

    def configure_video_history_memory(self, *, num_history_frames: int) -> None:
        if int(num_history_frames) != 0:
            raise ValueError("Motus clean training currently supports num_history_frames=0 only.")

    def _discover_episodes(self) -> list[_Episode]:
        episodes: list[_Episode] = []
        for root, domain_id in zip(self.dataset_dirs, self.domain_ids):
            if not root.is_dir():
                raise FileNotFoundError(f"Motus clean dataset directory does not exist: {root}")
            manifest_path = root / "dataset_manifest.json"
            if manifest_path.is_file():
                records = json.loads(manifest_path.read_text(encoding="utf-8"))["episodes"]
            else:
                records = []
                for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                    qpos_dir = task_dir / "qpos"
                    for qpos_path in sorted(
                        qpos_dir.glob("*.pt"), key=lambda p: int(p.stem)
                    ):
                        qpos = torch.load(
                            qpos_path, map_location="cpu", weights_only=False
                        )
                        if (
                            not torch.is_tensor(qpos)
                            or qpos.ndim != 2
                            or qpos.shape[1] != 14
                        ):
                            raise ValueError(
                                f"Expected qpos [T,14], got {type(qpos)} at {qpos_path}"
                            )
                        records.append(
                            {
                                "task_name": task_dir.name,
                                "episode_id": int(qpos_path.stem),
                                "source_length": int(qpos.shape[0]),
                            }
                        )
            for record in records:
                task_dir = root / record["task_name"]
                episode_id = int(record["episode_id"])
                qpos_path = task_dir / "qpos" / f"{episode_id}.pt"
                video_path = task_dir / "videos" / f"{episode_id}.mp4"
                prompts_path = task_dir / "metas" / f"{episode_id}.txt"
                embeddings_path = task_dir / "umt5_wan" / f"{episode_id}.pt"
                for required in (qpos_path, video_path, prompts_path, embeddings_path):
                    if not required.is_file():
                        raise FileNotFoundError(
                            f"Incomplete Motus episode; missing {required}"
                        )
                source_length = int(record["source_length"])
                target_length = max(
                    1,
                    int(
                        math.floor(
                            (source_length - 1) * self.target_fps / self.source_fps
                        )
                    )
                    + 1,
                )
                episodes.append(
                    _Episode(
                        task_name=str(record["task_name"]),
                        episode_id=episode_id,
                        video_path=video_path,
                        qpos_path=qpos_path,
                        prompts_path=prompts_path,
                        embeddings_path=embeddings_path,
                        source_length=source_length,
                        target_length=target_length,
                        domain_id=domain_id,
                    )
                )
        if not episodes:
            raise ValueError(f"No Motus episodes found under {self.dataset_dirs}")
        return episodes

    def __len__(self) -> int:
        return self._cumulative_lengths[-1]

    def denormalize_action(
        self, action: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        """Convert normalized Motus actions back to native 14-D qpos space."""
        if action.ndim != 3 or proprio.ndim != 3:
            raise ValueError(
                "`action` and `proprio` must have shape [B,T,D], got "
                f"{tuple(action.shape)} and {tuple(proprio.shape)}."
            )
        batch = {
            "action": {"default": action},
            "state": {"default": proprio},
        }
        batch = self.normalizer.backward(batch)
        return batch["action"]["default"]

    def _resolve_index(self, index: int) -> tuple[_Episode, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_idx = bisect.bisect_right(self._cumulative_lengths, index)
        previous = self._cumulative_lengths[episode_idx - 1] if episode_idx else 0
        return self.episodes[episode_idx], int(index - previous)

    @staticmethod
    def _cache_put(cache: OrderedDict, key, value, max_size: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_size:
            cache.popitem(last=False)

    def _load_qpos(self, episode: _Episode) -> torch.Tensor:
        if episode.qpos_path not in self._qpos_cache:
            value = torch.load(
                episode.qpos_path, map_location="cpu", weights_only=False
            ).float()
            self._cache_put(
                self._qpos_cache, episode.qpos_path, value, self._qpos_cache_size
            )
        self._qpos_cache.move_to_end(episode.qpos_path)
        return self._qpos_cache[episode.qpos_path]

    def _load_text(self, episode: _Episode) -> tuple[list[str], list[torch.Tensor]]:
        key = episode.embeddings_path
        if key not in self._text_cache:
            prompts = [
                line.strip()
                for line in episode.prompts_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            embeddings = torch.load(key, map_location="cpu", weights_only=True, mmap=True)
            if not isinstance(embeddings, list) or not embeddings:
                raise ValueError(f"Expected a non-empty embedding list in {key}")
            if len(prompts) != len(embeddings):
                raise ValueError(
                    f"Prompt/embedding count mismatch in {episode.task_name}/{episode.episode_id}: "
                    f"{len(prompts)} != {len(embeddings)}"
                )
            self._cache_put(
                self._text_cache, key, (prompts, embeddings), self._text_cache_size
            )
        self._text_cache.move_to_end(key)
        return self._text_cache[key]

    def _target_to_source(self, target_positions: torch.Tensor) -> torch.Tensor:
        return (
            target_positions.float()
            * self.source_fps
            / self.target_fps
        )

    @staticmethod
    def _interpolate(sequence: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.clamp(0, sequence.shape[0] - 1)
        lo = positions.floor().long()
        hi = positions.ceil().long()
        weight = (positions - lo.float()).unsqueeze(-1)
        return sequence[lo] * (1.0 - weight) + sequence[hi] * weight

    def _decode_video(self, episode: _Episode, source_indices: torch.Tensor) -> torch.Tensor:
        source_indices = source_indices.clamp(0, episode.source_length - 1).long()
        first = int(source_indices.min().item())
        last = int(source_indices.max().item())
        frames, _, _ = read_video(
            str(episode.video_path),
            start_pts=first / self.source_fps,
            end_pts=(last + 0.5) / self.source_fps,
            pts_unit="sec",
            output_format="THWC",
        )
        if frames.shape[0] == 0:
            raise RuntimeError(f"No frames decoded from {episode.video_path}")
        relative = (source_indices - first).clamp(0, frames.shape[0] - 1)
        packed = frames.index_select(0, relative).permute(0, 3, 1, 2)
        if tuple(packed.shape[-2:]) != (360, 320):
            raise ValueError(
                f"Expected packed Motus video size 360x320, got {tuple(packed.shape[-2:])} "
                f"for {episode.video_path}"
            )

        top = transforms_F.resize(
            packed[:, :, :240, :], [256, 320], antialias=True
        )
        left = transforms_F.resize(
            packed[:, :, 240:, :160], [128, 160], antialias=True
        )
        right = transforms_F.resize(
            packed[:, :, 240:, 160:], [128, 160], antialias=True
        )
        composite = torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)
        if tuple(composite.shape[-2:]) != self.video_size:
            composite = transforms_F.resize(composite, list(self.video_size), antialias=True)
        composite = composite.float().div_(255.0).sub_(0.5).div_(0.5)
        return composite.permute(1, 0, 2, 3).contiguous()

    def _get(self, index: int) -> dict:
        episode, start = self._resolve_index(index)
        qpos = self._load_qpos(episode)
        action_offsets = torch.arange(1, self.num_frames, dtype=torch.float32)
        state_offsets = torch.arange(0, self.num_frames - 1, dtype=torch.float32)
        video_offsets = torch.tensor(self.video_sample_indices, dtype=torch.float32)
        action_target = start + action_offsets * self.global_sample_stride
        state_target = start + state_offsets * self.global_sample_stride
        video_target = start + video_offsets * self.global_sample_stride

        action_pad = action_target >= episode.target_length
        state_pad = state_target >= episode.target_length
        image_pad = video_target >= episode.target_length
        action = self._interpolate(qpos, self._target_to_source(action_target))
        proprio = self._interpolate(qpos, self._target_to_source(state_target))
        source_video_indices = self._target_to_source(video_target).round().long()
        video = self._decode_video(episode, source_video_indices)

        normalized = self.normalizer.forward(
            {"action": {"default": action}, "state": {"default": proprio}}
        )
        action = normalized["action"]["default"]
        proprio = normalized["state"]["default"]

        prompts, embeddings = self._load_text(episode)
        if self.is_training_set and self.prompt_sampling == "random":
            prompt_idx = random.randrange(len(prompts))
        else:
            prompt_idx = 0
        prompt = prompts[prompt_idx]
        embedding = embeddings[prompt_idx]
        if not torch.is_tensor(embedding) or embedding.ndim != 2:
            raise ValueError(f"Expected text embedding [L,D], got {type(embedding)}")
        context = embedding.new_zeros((self.context_len, embedding.shape[1]))
        valid = min(self.context_len, int(embedding.shape[0]))
        context[:valid] = embedding[:valid]
        # Match RobotVideoDataset/Wan behavior: padded context is zero, while the
        # downstream attention mask remains enabled for the full fixed length.
        context_mask = torch.ones(self.context_len, dtype=torch.bool)

        current_frame_index = start // self.action_video_freq_ratio
        temporal_offsets = torch.tensor(
            self.video_sample_indices[::4], dtype=torch.long
        ) * self.global_sample_stride // self.action_video_freq_ratio
        result = {
            "video": video,
            "action": action,
            "proprio": proprio,
            # The bundled UMT5 tensor was computed from this exact prompt.
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_pad,
            "action_is_pad": action_pad,
            "proprio_is_pad": state_pad,
            "action_video_freq_ratio": torch.tensor(self.action_video_freq_ratio),
            "video_current_frame_index": torch.tensor(current_frame_index),
            "video_temporal_position_ids": current_frame_index + temporal_offsets,
        }
        if getattr(episode, "domain_id", None) is not None:
            result["domain_id"] = torch.tensor(episode.domain_id, dtype=torch.long)
        return result

    def __getitem__(self, index: int) -> dict:
        return self._get(index)
