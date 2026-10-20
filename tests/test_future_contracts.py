import unittest

import torch

from w0.trainer import Wan22Trainer
from test_future_video import make_model, sample


class EvalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def batched_sample(self, **kwargs):
        data = sample(**kwargs)
        data["prompt"] = ["test task"]
        return data

    def test_eval_preserves_metadata_and_masks(self):
        model, data = make_model(), self.batched_sample()
        data["video_temporal_position_ids"] = torch.tensor([[8, 12, 16]])
        data["action_is_pad"] = torch.zeros(1, 64, dtype=torch.bool)
        data["image_is_pad"] = torch.zeros(1, 9, dtype=torch.bool)
        unbatched = {k: v[0] if torch.is_tensor(v) else v[0] for k, v in data.items()}
        batched = Wan22Trainer._to_batched_eval_sample(unbatched)
        for key in ("action_video_freq_ratio", "video_temporal_position_ids", "action_is_pad", "image_is_pad"):
            torch.testing.assert_close(batched[key], data[key])
        model._evaluate_action_metrics = lambda **kwargs: {}
        metrics = model.evaluate_validation(batched, eval_num_inference_steps=1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["val_loss"])))

    def test_offset_eval_uses_shifted_action_time(self):
        model, data = make_model(horizon=96), self.batched_sample(frames=13, actions=96)
        model.max_action_offset = 15
        data["action_offset"] = torch.tensor([15])
        data["chunk_obs_images"] = torch.randn(1, 4, 3, 16, 16)
        batched = Wan22Trainer._to_batched_eval_sample(data)
        kwargs = model._build_eval_infer_kwargs(sample=batched, eval_num_inference_steps=1)
        self.assertEqual(kwargs["action_position_offset"], 15)
        positions = []
        original = model._build_prefilled_action_pre_and_mask

        def record(**params):
            positions.append(params["chunk_start"])
            pre, mask = original(**params)
            start = params["chunk_start"]
            torch.testing.assert_close(pre["freqs"], model.action_expert.freqs[start:start + 16].view(16, 1, -1))
            return pre, mask

        model._build_prefilled_action_pre_and_mask = record
        result = model.infer(**kwargs)
        self.assertEqual(positions, [15, 31, 47, 63])
        self.assertEqual(tuple(result["action"].shape), (64, 4))

    def test_history_generation_matches_training_positions(self):
        model, data = make_model(), self.batched_sample()
        model.configure_chunk_history(num_history_frames=1, action_video_read_mode="future")
        data["video_current_frame_index"] = torch.tensor([8])
        data["video_temporal_position_ids"] = torch.tensor([[8, 12, 16]])
        expected = model._get_main_video_temporal_position_ids(data, batch_size=1, current_clip_latent_frames=3)
        del data["video_temporal_position_ids"]
        fallback = model._get_main_video_temporal_position_ids(data, batch_size=1, current_clip_latent_frames=3)
        torch.testing.assert_close(expected, fallback)
        positions = []
        original = model.video_expert.pre_dit

        def record(**params):
            if params["x"].shape[2] == 3:
                positions.append(params["temporal_position_ids"].clone())
            return original(**params)

        model.video_expert.pre_dit = record
        for index in (0, 8):
            state = model.prefill_video(input_image=data["video"][:, :, 0], action_horizon=64,
                                        context=data["context"], context_mask=data["context_mask"],
                                        video_frame_index=index)
        torch.testing.assert_close(positions[-1], expected[0])
        self.assertEqual(model._observed_frame_index, 16)
        self.assertEqual(len(model._history_kv_entries), 1)
        self.assertEqual(state["video_seq_len"], 12)

    def test_reusing_storage_slot_starts_from_noise(self):
        model, data = make_model(), self.batched_sample()
        state = model.prefill_video(input_image=data["video"][:, :, 0], action_horizon=64,
                                   context=data["context"], context_mask=data["context_mask"])
        seen = []
        original = model._predict_action_chunk_with_clean_cache

        def record(**kwargs):
            seen.append(kwargs["noisy_action_chunk"].clone())
            return original(**kwargs)

        model._predict_action_chunk_with_clean_cache = record
        for offset in (0, 4):
            state["action_position_offset"] = offset
            out = model.infer_action_chunk(inference_state=state, chunk_index=0,
                                           chunk_obs_image=data["video"][:, :, 0],
                                           chunk_proprio=data["proprio"][:, 0], num_inference_steps=1)
            state = out["inference_state"]
        torch.testing.assert_close(seen[0], seen[1])


if __name__ == "__main__":
    unittest.main()
