"""Native LeRobot parquet/video reader with dataset x embodiment routing.

Uses recorded action commands at t (not next-state relabeling), per-domain
normalization, explicit camera slots, and padded vectors for mixed-domain batches.
"""

import bisect
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import av
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from .domain_registry import read_registry, registry_fingerprint, episode_is_validation
from .lerobot.robot_video_dataset import DEFAULT_PROMPT


def decode_native_frames(path, positions, fps):
    """Seek by PTS and retain only requested frames, including repeated padding."""
    requested = [float(p) / fps for p in positions]
    selected, index = [], 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        container.seek(max(0, int((origin + requested[0]) / stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base) - origin
            if timestamp < requested[index] - 0.5 / fps:
                continue
            rgb = None
            while index < len(requested) and timestamp >= requested[index] - 0.5 / fps:
                if abs(timestamp - requested[index]) > 0.51 / fps:
                    raise ValueError(f"Video timestamp does not match native FPS at {path}: {timestamp} vs {requested[index]}")
                if rgb is None:
                    rgb = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                selected.append(rgb)
                index += 1
            if index == len(requested):
                break
    if index != len(requested):
        raise ValueError(f"Missing requested frames at {path}: {index}/{len(requested)}")
    return torch.stack(selected)


class DomainNormalizer:
    def __init__(self, configs, statistics):
        self.configs = configs
        self.statistics = statistics
        for config in configs:
            for kind, dim in (("action", config["action_dim"]), ("proprio", config["proprio_dim"])):
                stats = statistics[config["name"]][kind]
                mean, std = torch.tensor(stats["mean"]), torch.tensor(stats["std"])
                if mean.shape != (dim,) or std.shape != (dim,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
                    raise ValueError(f"Invalid {kind} normalization for {config['name']}")

    def transform(self, x, domain_id, kind, inverse=False):
        config = self.configs[int(domain_id)]
        dim = config[kind + "_dim"] if kind == "action" else config["proprio_dim"]
        stats = self.statistics[config["name"]][kind]
        mean, std = x.new_tensor(stats["mean"]), x.new_tensor(stats["std"]).clamp_min(1e-6)
        x = x[..., :dim]
        return x * std + mean if inverse else (x - mean) / std


class DomainVideoDataset(torch.utils.data.Dataset):
    uses_epoch_indices = True
    def __init__(self, domain_registry_path, domain_stats_path, text_embedding_cache_dir,
                 action_horizon=64, action_chunk_size=16, action_video_freq_ratio=8,
                 target_fps=50.0, video_size=(384, 320), context_len=128,
                 is_training_set=True, val_set_proportion=0.05, seed=42,
                 samples_per_epoch=100000, domain_weights=None, max_sources_per_domain=None,
                 text_encoder_id="wan22ti2v5b", episode_cache_size=2):
        self.registry = read_registry(domain_registry_path)
        self.domain_configs = self.registry["domain_configs"]
        self.domain_ids = list(range(len(self.domain_configs)))
        self.names = [d["name"] for d in self.domain_configs]
        self.action_dim = max(d["action_dim"] for d in self.domain_configs)
        self.proprio_dim = max(d["proprio_dim"] for d in self.domain_configs)
        self.action_horizon, self.action_chunk_size = int(action_horizon), int(action_chunk_size)
        self.num_frames = self.action_horizon + 1
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.target_fps, self.video_size = float(target_fps), tuple(video_size)
        self.context_len, self.seed = int(context_len), int(seed)
        self.samples_per_epoch = int(samples_per_epoch)
        self.max_action_offset, self.num_history_frames, self._action_offset_enabled = 0, 0, False
        self.is_training_set, self.val_set_proportion = bool(is_training_set), float(val_set_proportion)
        if self.samples_per_epoch < 1 or self.target_fps <= 0 or not 0 < self.val_set_proportion < 1:
            raise ValueError("Invalid epoch size, FPS or validation proportion.")
        if self.action_horizon <= 0 or self.action_chunk_size <= 0 or self.action_video_freq_ratio <= 0:
            raise ValueError("Action horizon, chunk size and video frequency ratio must be positive.")
        if self.video_size[0] % 3 or self.video_size[1] % 2:
            raise ValueError("Camera canvas requires height divisible by 3 and even width.")
        if self.action_horizon % self.action_chunk_size or self.action_horizon % (4 * self.action_video_freq_ratio):
            raise ValueError("Horizon must align to action chunks and VAE temporal groups.")
        stats_bytes = Path(domain_stats_path).read_bytes()
        stats = json.loads(stats_bytes)
        if stats["registry_fingerprint"] != registry_fingerprint(self.registry):
            raise ValueError("Domain statistics were prepared for a different registry.")
        if stats["split_seed"] != self.seed or stats["val_set_proportion"] != self.val_set_proportion:
            raise ValueError("Domain statistics and dataset must use the same trajectory split.")
        self.domain_normalization = stats["domains"]
        self.normalizer = DomainNormalizer(self.domain_configs, self.domain_normalization)
        import hashlib
        self.domain_data_recipe = {
            "registry_fingerprint": registry_fingerprint(self.registry),
            "stats_sha256": hashlib.sha256(stats_bytes).hexdigest(),
            "domain_names": self.names, "target_fps": self.target_fps,
            "split_seed": self.seed, "val_set_proportion": self.val_set_proportion,
            "domain_weights": domain_weights, "samples_per_epoch": self.samples_per_epoch,
            "max_sources_per_domain": max_sources_per_domain,
            "sampling_version": "epoch_domain_episode_window_v1",
            "video_size": list(self.video_size), "camera_layout": "main_top_two_wrists_bottom",
            "context_len": self.context_len, "text_encoder_id": text_encoder_id,
        }
        self.text_embedding_cache_dir = str(text_embedding_cache_dir)
        self.text_encoder_id = text_encoder_id
        self._text_context_cache, self._episode_cache = OrderedDict(), OrderedDict()
        self._text_context_cache_maxsize, self.episode_cache_size = 128, int(episode_cache_size)
        self.episodes = [[] for _ in self.domain_configs]
        counts = [0] * len(self.names)
        self.sources = []
        for source in self.registry["sources"]:
            d = self.names.index(source["domain"])
            if max_sources_per_domain is not None and counts[d] >= int(max_sources_per_domain):
                continue
            counts[d] += 1
            source_idx = len(self.sources)
            self.sources.append(source)
            with (Path(source["root"]) / "meta/episodes.jsonl").open() as episode_file:
                for line in episode_file:
                    ep = json.loads(line)
                    if ep["length"] < 1 or len(ep.get("tasks", [])) != 1:
                        raise ValueError(f"Expected nonempty, single-task trajectory at {source['root']}: {ep}")
                    is_val = episode_is_validation(source, ep["episode_index"], self.val_set_proportion, self.seed)
                    if is_val != (not self.is_training_set):
                        continue
                    self.episodes[d].append((source_idx, ep))
        missing_domains = [name for name, episodes in zip(self.names, self.episodes) if not episodes]
        if missing_domains:
            raise ValueError(f"Domains without trajectories in the selected split: {missing_domains}. Increase the source limit or use the full registry.")
        weights = np.asarray(domain_weights if domain_weights is not None else [1.] * len(self.names), dtype=float)
        if weights.shape != (len(self.names),) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("domain_weights must be positive and align with bank order.")
        self._domain_cdf = np.cumsum(weights / weights.sum())

    def __len__(self):
        return self.samples_per_epoch

    def configure_video_history_memory(self, *, num_history_frames):
        if num_history_frames:
            raise ValueError("Native domain reader currently supports num_history_frames=0.")

    def _get_cached_text_context(self, prompt):
        # Same cache contract as the existing precompute script, configurable encoder name.
        if prompt in self._text_context_cache:
            self._text_context_cache.move_to_end(prompt)
            return self._text_context_cache[prompt]
        import hashlib
        hashed = hashlib.sha256(prompt.encode()).hexdigest()
        path = Path(self.text_embedding_cache_dir) / f"{hashed}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing text cache {path}; run scripts/precompute_text_embeds.py with this task.")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        context, mask = payload["context"].clone(), payload["mask"].bool()
        if context.shape[0] != self.context_len or mask.shape != (self.context_len,):
            raise ValueError("Text cache context length mismatch.")
        context[~mask] = 0
        result = context, torch.ones_like(mask)
        self._text_context_cache[prompt] = result
        if len(self._text_context_cache) > self._text_context_cache_maxsize:
            self._text_context_cache.popitem(last=False)
        return result

    def _episode_arrays(self, source_idx, ep, domain_id):
        cache_key = (source_idx, ep["episode_index"])
        if cache_key not in self._episode_cache:
            source, config = self.sources[source_idx], self.domain_configs[domain_id]
            episode = ep["episode_index"]
            path = Path(source["root"]) / source["data_path"].format(
                episode_chunk=episode // source["chunks_size"], episode_index=episode)
            keys = list(dict.fromkeys(f["key"] for kind in ("action_fields", "proprio_fields") for f in config[kind]))
            table = pq.read_table(path, columns=keys)
            arrays = []
            for kind in ("action_fields", "proprio_fields"):
                columns = [torch.tensor(np.asarray(table[f["key"]].to_pylist()), dtype=torch.float32).reshape(len(table), f["dim"])
                           for f in config[kind]]
                arrays.append(torch.cat(columns, dim=-1))
            if len(table) != ep["length"] or not all(torch.isfinite(x).all() for x in arrays):
                raise ValueError(f"Invalid native trajectory: {path}")
            self._episode_cache[cache_key] = arrays
            if len(self._episode_cache) > self.episode_cache_size:
                self._episode_cache.popitem(last=False)
        return self._episode_cache[cache_key]

    @staticmethod
    def _interpolate(x, positions):
        positions = positions.clamp(0, len(x) - 1)
        lo, hi = positions.floor().long(), positions.ceil().long()
        alpha = (positions - lo).unsqueeze(-1)
        return x[lo] * (1 - alpha) + x[hi] * alpha

    def _video(self, source, episode, positions):
        height, width = self.video_size
        # Fixed three-slot canvas: main view above, left/right wrists below.
        sizes = [(height * 2 // 3, width), (height // 3, width // 2), (height // 3, width // 2)]
        frames = []
        for key, size in zip(source["videos"], sizes):
            if key is None:
                frames.append(torch.full((len(positions), 3, *size), -1.))
                continue
            path = Path(source["root"]) / source["video_path"].format(
                episode_chunk=episode // source["chunks_size"], episode_index=episode, video_key=key)
            rgb = decode_native_frames(path, positions, source["fps"])
            rgb = F.interpolate(rgb.float(), size=size, mode="bilinear", align_corners=False) / 127.5 - 1
            frames.append(rgb)
        return torch.cat([frames[0], torch.cat(frames[1:], dim=-1)], dim=-2).permute(1, 0, 2, 3)

    def __getitem__(self, index):
        index, epoch = index if isinstance(index, tuple) else (index, 0)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        domain = min(bisect.bisect_right(self._domain_cdf, (index + 0.5) / len(self)), len(self.names) - 1)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch), int(index)]))
        source_idx, ep = self.episodes[domain][int(rng.integers(len(self.episodes[domain])))]
        source = self.sources[source_idx]
        action, proprio = self._episode_arrays(source_idx, ep, domain)
        target_length = int((ep["length"] - 1) * self.target_fps / source["fps"]) + 1
        start = int(rng.integers(max(1, target_length - self.action_horizon)))
        ticks = torch.arange(self.action_horizon + 1) + start
        positions = ticks.float() * source["fps"] / self.target_fps
        actions = self.normalizer.transform(self._interpolate(action, positions[:-1]), domain, "action")
        states = self.normalizer.transform(self._interpolate(proprio, positions), domain, "proprio")
        video_pos = positions[::self.action_video_freq_ratio].round().clamp(max=ep["length"] - 1)
        prompt = DEFAULT_PROMPT.format(task=ep["tasks"][0])
        context, context_mask = self._get_cached_text_context(prompt)
        return {"video": self._video(source, ep["episode_index"], video_pos),
                "action": F.pad(actions, (0, self.action_dim - actions.shape[-1])),
                "proprio": F.pad(states, (0, self.proprio_dim - states.shape[-1])),
                "domain_id": torch.tensor(domain, dtype=torch.long), "prompt": prompt,
                "context": context, "context_mask": context_mask,
                "action_is_pad": ticks[:-1] >= target_length, "proprio_is_pad": ticks >= target_length,
                "image_is_pad": ticks[::self.action_video_freq_ratio] >= target_length,
                "action_video_freq_ratio": torch.tensor(self.action_video_freq_ratio)}

    def denormalize_action(self, action, domain_id):
        return self.normalizer.transform(action, domain_id, "action", inverse=True)
