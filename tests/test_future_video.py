import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from w0.models.wan22.action_dit import ActionDiT
from w0.models.wan22.w0 import W0
from w0.models.wan22.mot import MoT
from w0.models.wan22.wan_video_dit import WanVideoDiT


class TinyVAE(nn.Module):
    temporal_downsample_factor = 4
    latent_dim = 4

    def encode(self, video, **kwargs):
        if isinstance(video, list):
            video = torch.stack(video)
        video = video[:, :, ::4]
        b, c, t, h, w = video.shape
        x = F.avg_pool2d(video.transpose(1, 2).reshape(b * t, c, h, w), 4)
        x = x.reshape(b, t, c, h // 4, w // 4).transpose(1, 2)
        return torch.cat([x, x.mean(1, keepdim=True)], dim=1)


def make_model(mode="future", horizon=64, *, action_kwargs=None, action_dim=4, proprio_dim=4):
    torch.manual_seed(17)
    video = WanVideoDiT(
        hidden_dim=48, in_dim=4, out_dim=4, ffn_dim=96, text_dim=32,
        freq_dim=16, eps=1e-6, patch_size=(1, 2, 2), num_heads=2,
        attn_head_dim=24, num_layers=3, has_image_input=False,
        seperated_timestep=True, fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
    )
    action = ActionDiT(
        hidden_dim=32, action_dim=action_dim, ffn_dim=64, text_dim=32, freq_dim=16,
        eps=1e-6, num_heads=2, attn_head_dim=24, num_layers=3,
        action_chunk_size=16, autoregressive_teacher_forcing=True,
        **(action_kwargs or {}),
    )
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)
    model = W0(video, action, mot, TinyVAE(), text_dim=32, proprio_dim=proprio_dim)
    model.configure_action_chunking(action_horizon=64, action_chunk_size=16)
    model.configure_chunk_obs_context(action_obs_downsample_factor=1, chunk_kv_editor_num_queries=3)
    model.configure_chunk_history(action_video_read_mode=mode)
    model.configure_future_video(video_plan_horizon=horizon, video_action_stride=8, video_plan_steps=2)
    return model


def sample(frames=9, actions=64, batch=1):
    return {
        "video": torch.randn(batch, 3, frames, 16, 16),
        "action": torch.randn(batch, actions, 4),
        "proprio": torch.randn(batch, actions + 1, 4),
        "context": torch.randn(batch, 3, 32),
        "context_mask": torch.ones(batch, 3, dtype=torch.bool),
        "image_is_pad": torch.zeros(batch, frames, dtype=torch.bool),
        "action_is_pad": torch.zeros(batch, actions, dtype=torch.bool),
        "action_video_freq_ratio": torch.full((batch,), 8),
    }


class FutureVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def prefill(self, model, data):
        return model.prefill_video(
            input_image=data["video"][:, :, 0], action_horizon=64,
            context=data["context"], context_mask=data["context_mask"], seed=3,
        )

    def test_horizon_and_four_chunks(self):
        model, data = make_model(), sample()
        state = self.prefill(model, data)
        self.assertEqual(state["video_plan_horizon"], 64)
        self.assertEqual(state["video_sampled_frames"], 9)
        self.assertEqual(state["video_latent_frames"], 3)
        self.assertEqual(state["video_seq_len"], 3 * state["video_tokens_per_frame"])
        first = model.vae.encode(data["video"][:, :, :1])
        torch.testing.assert_close(state["video_plan_latents"][:, :, :1], first)
        cache = state["video_kv_cache"]
        observed_positions = []
        original = model.action_expert.pre_dit

        def record(**kwargs):
            observed_positions.append(kwargs["noisy_position_offset"])
            return original(**kwargs)

        model.action_expert.pre_dit = record
        for i in range(4):
            out = model.infer_action_chunk(
                inference_state=state, chunk_index=i,
                chunk_obs_image=data["video"][:, :, i * 2],
                chunk_proprio=data["proprio"][:, i * 16], num_inference_steps=1,
            )
            self.assertEqual(tuple(out["action_chunk"].shape), (16, 4))
            state = out["inference_state"]
            self.assertIs(state["video_kv_cache"], cache)
            self.assertEqual(state["_chunk_video_kv_cache"][0]["k"].shape[1], state["video_seq_len"])
        self.assertEqual(observed_positions, [0, 16, 32, 48])
        with self.assertRaises(ValueError):
            model.infer_action_chunk(
                inference_state=state, chunk_index=4,
                chunk_obs_image=data["video"][:, :, 0], chunk_proprio=data["proprio"][:, 0],
            )

    def test_masks_and_legacy(self):
        for mode in ("future", "current_only"):
            model = make_model(mode)
            mask = model._build_mot_attention_mask(12, 16, 4, torch.device("cpu"), torch.ones(16, 16, dtype=torch.bool))
            self.assertFalse(mask[:12, 12:].any())
            self.assertFalse(mask[:4, 4:12].any())
            self.assertEqual(bool(mask[12:, 4:12].all()), mode == "future")
            state = self.prefill(model, sample())
            if mode == "current_only":
                self.assertEqual(state["video_seq_len"], state["video_tokens_per_frame"])

    def test_editor_uses_expert_dtype_not_legacy_embedding(self):
        model = make_model()
        model.video_expert.to(dtype=torch.bfloat16)
        model.action_expert.to(dtype=torch.bfloat16)
        self.assertEqual(model.mot.action_branch_embedding.dtype, torch.float32)
        model.mot.configure_chunk_kv_cache_editor(query_dim=32)
        self.assertEqual(next(model.mot.chunk_kv_cache_editor.parameters()).dtype, torch.bfloat16)

    def test_future_cache_changes_action(self):
        model, data = make_model(), sample()
        state = self.prefill(model, data)
        modified = copy.deepcopy(state)
        for layer in modified["video_kv_cache"]:
            layer["v"][:, state["video_tokens_per_frame"]:] += 2
        def run(s):
            return model.infer_action_chunk(
                inference_state=s, chunk_index=0, chunk_obs_image=data["video"][:, :, 0],
                chunk_proprio=data["proprio"][:, 0], num_inference_steps=1,
            )["action_chunk"]
        self.assertGreater((run(state) - run(modified)).abs().max().item(), 1e-5)

    def test_training_backward_and_generated_conditions(self):
        model, data = make_model(), sample()
        cache_records = []
        original = model._generate_future_video
        def capture(**kwargs):
            plan = original(**kwargs)
            cache_records.append(plan["video_kv_cache"])
            return plan
        model._generate_future_video = capture
        model.train()
        loss, metrics = model.training_loss(data)
        self.assertTrue(model.training)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.video_expert.blocks[0].self_attn.q.weight.grad)
        self.assertIsNotNone(model.action_expert.action_encoder.weight.grad)
        self.assertGreater(model.mot.chunk_kv_cache_editor.layer_delta_proj[0][-1].weight.grad.abs().sum().item(), 0)
        self.assertFalse(cache_records[0][0]["k"].requires_grad)
        inputs = model.build_inputs(data)
        torch.manual_seed(4)
        a = original(first_frame_latents=inputs["first_frame_latents"], context=inputs["context"], context_mask=inputs["context_mask"])
        data["video"][:, :, 1:] += 100
        changed = model.build_inputs(data)
        torch.manual_seed(4)
        b = original(first_frame_latents=changed["first_frame_latents"], context=changed["context"], context_mask=changed["context_mask"])
        torch.testing.assert_close(a["video_kv_cache"][-1]["k"], b["video_kv_cache"][-1]["k"])

    def test_offset_training_and_coverage(self):
        model, data = make_model(horizon=96), sample(frames=13, actions=96, batch=2)
        model.max_action_offset = 15
        data["action_offset"] = torch.tensor([1, 15])
        data["chunk_obs_images"] = torch.randn(2, 4, 3, 16, 16)
        loss, metrics = model.training_loss(data)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["mean_offset"], 8)
        with self.assertRaises(ValueError):
            model.configure_future_video(video_plan_horizon=63)

    def test_reject_wrong_temporal_scale(self):
        model, data = make_model(), sample()
        data["action_video_freq_ratio"] = torch.tensor([4])
        with self.assertRaisesRegex(ValueError, "action_video_freq_ratio"):
            model.training_loss(data)

if __name__ == "__main__":
    unittest.main()
