"""Native clean data timing, padding and evaluation normalization contract."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from w0.datasets.motus_clean_video_dataset import MotusCleanVideoDataset


class MotusCleanDatasetTests(unittest.TestCase):
    def test_next_action_timing_padding_and_prompt_pair(self):
        dataset = object.__new__(MotusCleanVideoDataset)
        dataset.num_frames = 65
        dataset.action_video_freq_ratio = 8
        dataset.video_sample_indices = list(range(0, 65, 8))
        dataset.global_sample_stride = 1
        dataset.source_fps, dataset.target_fps = 30.0, 50.0
        dataset.context_len = 128
        dataset.is_training_set = False
        dataset.prompt_sampling = "random"
        dataset.normalizer = SimpleNamespace(forward=lambda x: x)
        episode = SimpleNamespace(target_length=11)
        qpos = torch.arange(7).float()[:, None].expand(-1, 14)
        embedding = torch.ones(3, 4096, dtype=torch.bfloat16)
        with patch.object(dataset, "_resolve_index", return_value=(episode, 0)), \
             patch.object(dataset, "_load_qpos", return_value=qpos), \
             patch.object(dataset, "_load_text", return_value=(["paired prompt"], [embedding])), \
             patch.object(dataset, "_decode_video", return_value=torch.zeros(3, 9, 384, 320)) as decode:
            sample = dataset[0]
        self.assertEqual(sample["prompt"], "paired prompt")
        torch.testing.assert_close(sample["action"][0], torch.full((14,), 0.6))
        torch.testing.assert_close(sample["proprio"][0], torch.zeros(14))
        self.assertEqual(int(sample["action_is_pad"].sum()), 54)
        self.assertEqual(int(sample["proprio_is_pad"].sum()), 53)
        torch.testing.assert_close(sample["video_temporal_position_ids"], torch.tensor([0, 4, 8]))
        self.assertEqual(int(sample["action_video_freq_ratio"]), 8)
        torch.testing.assert_close(sample["context"][:3], embedding)
        self.assertEqual(int(sample["context"][3:].count_nonzero()), 0)
        torch.testing.assert_close(decode.call_args.args[1], torch.tensor([0, 5, 10, 14, 19, 24, 29, 34, 38]))


if __name__ == "__main__":
    unittest.main()
