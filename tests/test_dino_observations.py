from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from w0.trainer import Wan22Trainer
from test_future_video import make_model, sample


class FakeDino(nn.Module):
    """Small frozen feature extractor for OVCR routing/gradient tests."""
    feature_dim = 768

    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = nn.Linear(3, self.feature_dim).requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def encode_frames(self, frames):
        return self.backbone(nn.functional.adaptive_avg_pool2d(frames, (2, 2)).flatten(2).transpose(1, 2))

    def behavior_config(self):
        return {"model_name": "test-dino", "weights_sha256": "test-only"}


def dino_model():
    model = make_model()
    with patch("w0.models.wan22.dino_encoder.DinoVideoEncoder", FakeDino):
        model.configure_chunk_obs_context(
            action_obs_downsample_factor=1, chunk_kv_editor_num_queries=3,
            observation_encoder="dino", dino_config={"weights_path": "test-only"},
        )
    return model


class DinoObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_chunk_training_images_and_video_use_identical_dino_features_without_vae(self):
        model, data = dino_model(), sample()
        with patch.object(model, "_encode_input_image_latents_tensor", side_effect=AssertionError("OVCR used VAE")):
            from_video = model._build_chunk_aligned_obs_context_from_video(
                video=data["video"], action_horizon=64, tiled=False,
            )
            from_images = model._build_chunk_aligned_obs_context_from_images(
                chunk_obs_images=data["video"][:, :, [0, 2, 4, 6]].permute(0, 2, 1, 3, 4), tiled=False,
            )
        torch.testing.assert_close(from_video[0], from_images[0])
        self.assertEqual(from_video[0].shape, (1, 4, 4, model.text_dim))

    def test_train_eval_gradients_and_four_fast_chunks(self):
        model, data = dino_model(), sample()
        Wan22Trainer._apply_dit_only_train_mode(model)
        loss, _ = model.training_loss(data)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.dino_encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.action_obs_visual_proj.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.chunk_obs_query_encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.action_expert.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.video_expert.parameters()))
        data["prompt"] = ["test task"]
        model._evaluate_action_metrics = lambda **kwargs: {}
        metrics = model.evaluate_validation(Wan22Trainer._to_batched_eval_sample(data), eval_num_inference_steps=1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["val_loss"])))
        state = model.prefill_video(input_image=data["video"][:, :, 0], action_horizon=64,
                                    context=data["context"], context_mask=data["context_mask"])
        cache = state["video_kv_cache"]
        with patch.object(model, "_encode_input_image_latents_tensor", side_effect=AssertionError("Fast path used VAE")):
            for i in range(4):
                result = model.infer_action_chunk(
                    inference_state=state, chunk_index=i,
                    chunk_obs_image=data["video"][:, :, 2 * i], chunk_proprio=data["proprio"][:, 16 * i],
                    num_inference_steps=1,
                )
                state = result["inference_state"]
                self.assertTrue(torch.isfinite(result["action_chunk"]).all())
                self.assertIs(state["video_kv_cache"], cache)

    def test_checkpoint_reload_and_explicit_vae_migration_preserve_other_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            source, target = make_model(), dino_model()
            source.save_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "observation_encoder"):
                target.load_checkpoint(path)
            target.checkpoint_observation_encoder_adapt = True
            fresh = {key: value.clone() for key, value in target.action_obs_visual_proj.state_dict().items()}
            target.load_checkpoint(path)
            for name in ("mot", "proprio_encoder", "chunk_obs_query_encoder"):
                for key, value in getattr(source, name).state_dict().items():
                    torch.testing.assert_close(getattr(target, name).state_dict()[key], value, rtol=0, atol=0)
            for key, value in target.action_obs_visual_proj.state_dict().items():
                expected = fresh[key] if key in {"0.weight", "0.bias", "1.weight"} else source.action_obs_visual_proj.state_dict()[key]
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "not optimizer resume"):
                target.load_checkpoint(path, optimizer=torch.optim.AdamW(target.parameters()))
            target.save_checkpoint(path)
            restored = dino_model()
            restored.load_checkpoint(path)
            self.assertEqual(restored.checkpoint_behavior(), target.checkpoint_behavior())
            restored.dino_encoder.behavior_config = lambda: {"model_name": "test-dino", "weights_sha256": "different"}
            restored.checkpoint_observation_encoder_adapt = True
            with self.assertRaisesRegex(ValueError, "dino_encoder"):
                restored.load_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "observation_encoder"):
                source.load_checkpoint(path)

    def test_migration_does_not_bypass_other_behavior_checks(self):
        source, target = make_model("current_only"), dino_model()
        with self.assertRaisesRegex(ValueError, "action_video_read_mode"):
            target.validate_checkpoint_behavior(
                {"behavior_config": source.checkpoint_behavior()}, "unused.pt",
                allow_observation_encoder_migration=True,
            )

    def test_version_one_checkpoint_remains_vae_only(self):
        model = make_model()
        saved = model.checkpoint_behavior()
        del saved["observation_encoder"], saved["dino_encoder"]
        saved["schema_version"] = 1
        model.validate_checkpoint_behavior({"behavior_config": saved}, "unused.pt")
        with self.assertRaisesRegex(ValueError, "observation_encoder"):
            dino_model().validate_checkpoint_behavior({"behavior_config": saved}, "unused.pt")


if __name__ == "__main__":
    unittest.main()
