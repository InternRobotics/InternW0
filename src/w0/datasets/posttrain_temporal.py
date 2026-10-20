"""Shared temporal sampling for the RoboTwin posttraining adapters."""
from numbers import Integral

import numpy as np


class PosttrainTemporalSampling:
    def _configure_temporal_sampling(self, *, action_horizon, action_chunk_size,
                                    max_action_offset, video_plan_horizon,
                                    action_video_freq_ratio):
        values = dict(action_horizon=action_horizon, action_chunk_size=action_chunk_size,
                      max_action_offset=max_action_offset, video_plan_horizon=video_plan_horizon,
                      action_video_freq_ratio=action_video_freq_ratio)
        for name, value in values.items():
            minimum = 0 if name == "max_action_offset" else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
            setattr(self, name, int(value))
        if self.action_horizon % self.action_chunk_size:
            raise ValueError("action_horizon must be divisible by action_chunk_size.")
        if self.action_horizon + self.max_action_offset > self.video_plan_horizon:
            raise ValueError("video_plan_horizon must cover action_horizon + max_action_offset.")
        # These adapters use the Wan 2.2 VAE (temporal downsample factor 4).
        if self.video_plan_horizon % (4 * self.action_video_freq_ratio):
            raise ValueError("video_plan_horizon must be divisible by 4 * action_video_freq_ratio.")
        self.num_history_frames = 0
        self.num_frames = self.video_plan_horizon + 1
        self._action_offset_enabled = self.max_action_offset > 0
        self.action_candidate_length = self.action_horizon + self.max_action_offset
        self.video_sample_indices = np.arange(0, self.num_frames, self.action_video_freq_ratio)
        self._chunk_start_offsets = np.arange(0, self.action_horizon, self.action_chunk_size)

    def _sample_action_offset(self, rng, remaining):
        if remaining < 1:
            raise ValueError("An anchor must have at least one real action transition.")
        if not self._action_offset_enabled:
            return 0
        # Keep short episodes; the first selected action must remain real.
        return int(rng.integers(min(self.max_action_offset, remaining - 1) + 1))

    def _sampling_recipe(self, base):
        # Statistics retain their original recipe; runtime sampling is recorded
        # separately because the bounds cover all real train transitions.
        return dict(base, action_horizon=self.action_horizon, action_chunk_size=self.action_chunk_size,
                    max_action_offset=self.max_action_offset, video_plan_horizon=self.video_plan_horizon)
