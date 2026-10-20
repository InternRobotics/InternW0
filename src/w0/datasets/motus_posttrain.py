"""Clean Motus joint posttraining: native 30 Hz -> 15 Hz, explicit 37D slots."""
from __future__ import annotations

from collections import OrderedDict
import bisect
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .domain_video_dataset import decode_native_frames
from .color_jitter import VideoColorJitter
from .robotwin_posttrain import (
    DOMAIN as BASE_DOMAIN, RECIPE as BASE_RECIPE, RobotwinPosttrainDataset,
    RobotwinPosttrainProcessor, camera_canvas,
)

DOMAIN = dict(BASE_DOMAIN, name="robotwin_motus_clean/aloha", dataset="robotwin_motus_clean",
              action_semantics="absolute_joint_position_next_sample")
RECIPE = dict(BASE_RECIPE, source_fps=30, target_fps=15, stride=2, phases=[0, 1], domain=DOMAIN,
              text="paired_motus_prompt_umt5_128",
              sampling="all_valid_native_anchors_two_phases_v1")


def validation_episode(task, episode, seed=42, fraction=.01):
    digest = hashlib.sha256(f"motus_clean:{seed}:{task}:{episode}".encode()).hexdigest()
    return int(digest[:16], 16) / 2**64 < fraction


def read_manifest(root: Path):
    content = (root / "dataset_manifest.json").read_bytes()
    episodes = sorted(json.loads(content)["episodes"], key=lambda e: (e["task_name"], e["episode_id"]))
    keys = [(e["task_name"], e["episode_id"]) for e in episodes]
    if len(keys) != len(set(keys)) or any(e["source_length"] < 3 for e in episodes):
        raise ValueError("Duplicate or too-short clean episodes.")
    return episodes, hashlib.sha256(content).hexdigest()


def read_qpos(root, episode):
    path = root / episode["task_name"] / "qpos" / f"{episode['episode_id']}.pt"
    qpos = torch.load(path, map_location="cpu", weights_only=True).float()
    if qpos.shape != (episode["source_length"], 14) or not torch.isfinite(qpos).all():
        raise ValueError(f"Invalid native qpos: {path}")
    return qpos


class MotusPosttrainProcessor(RobotwinPosttrainProcessor):
    recipe = RECIPE


class MotusPosttrainDataset(RobotwinPosttrainDataset):
    """An epoch visits every valid native anchor once, including both stride-2 phases."""
    def __init__(self, dataset_root, processor, is_training_set=True, seed=42,
                 val_set_proportion=.01, pretrained_norm_stats=None, *, action_horizon=64,
                 action_chunk_size=16, max_action_offset=15, video_plan_horizon=96,
                 action_video_freq_ratio=8, color_jitter=None):
        self._configure_temporal_sampling(
            action_horizon=action_horizon, action_chunk_size=action_chunk_size,
            max_action_offset=max_action_offset, video_plan_horizon=video_plan_horizon,
            action_video_freq_ratio=action_video_freq_ratio,
        )
        self.color_jitter = VideoColorJitter(color_jitter, training=bool(is_training_set))
        self.root, self.processor = Path(dataset_root), processor
        episodes, fingerprint = read_manifest(self.root)
        self.seed, self.is_training_set = int(seed), bool(is_training_set)
        if not 0 < val_set_proportion < 1:
            raise ValueError("A nonempty train/validation split is required.")
        stats = processor.statistics
        if (stats['metadata_sha256'] != fingerprint or stats['split_seed'] != seed
                or stats['val_set_proportion'] != val_set_proportion):
            raise ValueError("Clean statistics and manifest/split disagree.")
        self.episodes = [e for e in episodes if validation_episode(
            e['task_name'], e['episode_id'], seed, val_set_proportion) != is_training_set]
        if not self.episodes:
            raise ValueError("Empty clean split.")
        self.cumulative = np.cumsum([e['source_length'] - 2 for e in self.episodes]).tolist()
        self.domain_configs = [json.loads(json.dumps(DOMAIN))]
        self.domain_data_recipe = dict(self._sampling_recipe(RECIPE), metadata_sha256=fingerprint, split_seed=seed,
                                      val_set_proportion=val_set_proportion,
                                      stats_sha256=hashlib.sha256(Path(processor.stats_path).read_bytes()).hexdigest())
        self.domain_data_recipe.update(self.color_jitter.recipe())
        self._qpos_cache, self._text_cache = OrderedDict(), OrderedDict()

    def __len__(self):
        return self.cumulative[-1]

    def _text_for_episode(self, episode, rng):
        key = (episode['task_name'], episode['episode_id'])
        if key not in self._text_cache:
            base = self.root / key[0]
            prompts = (base / 'metas' / f'{key[1]}.txt').read_text().splitlines()
            embeddings = torch.load(base / 'umt5_wan' / f'{key[1]}.pt', map_location='cpu',
                                    weights_only=True, mmap=True)
            if not prompts or len(prompts) != len(embeddings):
                raise ValueError(f"Mismatched clean prompt/embedding pairs: {key}")
            self._text_cache[key] = prompts, embeddings
            if len(self._text_cache) > 2:
                self._text_cache.popitem(last=False)
        prompts, embeddings = self._text_cache[key]
        index = int(rng.integers(len(prompts))) if self.is_training_set else 0
        embedding = embeddings[index]
        if embedding.ndim != 2 or embedding.shape[1] != 4096 or not torch.isfinite(embedding).all():
            raise ValueError(f"Invalid clean embedding: {key}/{index}")
        context = embedding.new_zeros(128, 4096)
        count = min(len(embedding), 128)
        context[:count] = embedding[:count]
        return prompts[index], context, torch.ones(128, dtype=torch.bool)

    def __getitem__(self, index):
        index, epoch = index if isinstance(index, tuple) else (index, 0)
        if not 0 <= index < len(self):
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch), int(index)]))
        ep_index = bisect.bisect_right(self.cumulative, index)
        episode = self.episodes[ep_index]
        anchor = index - (self.cumulative[ep_index - 1] if ep_index else 0)
        key = (episode['task_name'], episode['episode_id'])
        if key not in self._qpos_cache:
            self._qpos_cache[key] = read_qpos(self.root, episode)
            if len(self._qpos_cache) > 8:
                self._qpos_cache.popitem(last=False)
        qpos = self._qpos_cache[key]
        length = len(qpos)
        remaining = (length - 1 - anchor) // 2
        offset = self._sample_action_offset(rng, remaining)
        positions = anchor + 2 * np.arange(self.action_candidate_length)
        action_pad, state_pad = positions + 2 >= length, positions >= length
        action, action_mask = self.processor.pack(qpos[np.minimum(positions + 2, length - 1)],
                                                 'action', torch.from_numpy(~action_pad)[:, None])
        proprio, proprio_mask = self.processor.pack(qpos[np.minimum(positions, length - 1)],
                                                    'proprio', torch.from_numpy(~state_pad)[:, None])
        video_positions = anchor + 2 * self.video_sample_indices
        chunk_positions = anchor + 2 * (offset + self._chunk_start_offsets)
        positions = np.minimum(np.concatenate([video_positions, chunk_positions]), length - 1)
        unique, inverse = np.unique(positions, return_inverse=True)
        path = self.root / key[0] / 'videos' / f'{key[1]}.mp4'
        packed = decode_native_frames(path, unique, 30)
        if packed.shape[-2:] != (360, 320):
            raise ValueError(f"Invalid Motus camera canvas: {path}")
        canvas = camera_canvas([packed[..., :240, :], packed[..., 240:, :160],
                                packed[..., 240:, 160:]])[torch.from_numpy(inverse)]
        canvas = self.color_jitter(canvas, seed=self.seed, epoch=epoch, index=index)
        prompt, context, context_mask = self._text_for_episode(episode, rng)
        video_count = len(self.video_sample_indices)
        return dict(video=canvas[:video_count].permute(1, 0, 2, 3), chunk_obs_images=canvas[video_count:],
                    action=action, proprio=proprio, action_valid_mask=action_mask,
                    proprio_valid_mask=proprio_mask, action_is_pad=torch.from_numpy(action_pad),
                    proprio_is_pad=torch.from_numpy(state_pad), image_is_pad=torch.from_numpy(video_positions >= length),
                    domain_id=torch.tensor(0), action_offset=torch.tensor(offset),
                    action_video_freq_ratio=torch.tensor(self.action_video_freq_ratio),
                    prompt=prompt, context=context, context_mask=context_mask)
