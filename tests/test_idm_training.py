"""IDM computation/gradient contracts, using real tiny DiT blocks and OVCR."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from w0.trainer import Wan22Trainer
from test_future_video import make_model, sample
from test_dino_observations import dino_model


def idm_model(*, horizon=64, dino=False, checkpoint=False):
    model = dino_model() if dino else make_model(horizon=horizon)
    model.configure_future_video(video_plan_horizon=horizon, future_training_mode="idm",
                                 idm_checkpoint_full_layer=checkpoint)
    Wan22Trainer._apply_dit_only_train_mode(model)
    return model


class IDMTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mask_validation_once_outside_checkpoint(self):
        model = idm_model(checkpoint=True)
        editor = model.mot.chunk_kv_cache_editor
        with patch.object(editor, "validate_video_key_mask", wraps=editor.validate_video_key_mask) as validate:
            model.training_loss(sample())[0].backward()
            self.assertEqual(validate.call_count, 1)
        with self.assertRaisesRegex(ValueError, "valid video key"):
            editor.validate_video_key_mask(torch.zeros(2, 12, dtype=torch.bool), (2, 12))

    def test_resume_rejects_model_and_data_offset_changes(self):
        model = idm_model(horizon=96)
        recipe = {"training_recipe": model.training_recipe()}
        model.max_action_offset = 15
        with self.assertRaisesRegex(ValueError, "recipe"):
            model.validate_training_recipe(recipe)
        model.max_action_offset = 0
        dataset = SimpleNamespace(max_action_offset=0, action_horizon=64,
                                  action_chunk_size=16, action_video_freq_ratio=8,
                                  _action_offset_enabled=False)
        model.bind_training_data_recipe(dataset, dataset)
        recipe = {"training_recipe": model.training_recipe()}
        model.validate_training_recipe(recipe)
        dataset._action_offset_enabled = True
        model.bind_training_data_recipe(dataset, dataset)
        with self.assertRaisesRegex(ValueError, "recipe"):
            model.validate_training_recipe(recipe)
        dataset.max_action_offset = 15
        with self.assertRaisesRegex(ValueError, "max_action_offset"):
            model.bind_training_data_recipe(dataset, dataset)

    def test_sampler_probability_prefix_rng_and_nonmutation(self):
        model = idm_model()
        gt = torch.randn(16, 4, 3, 4, 4)
        first = gt[:, :, :1].clone()
        original = gt.clone()
        for probability in (0, 1, 0.5):
            model.idm_video_cond_noise_prob = probability
            torch.manual_seed(31)
            cond, t, stats = model._sample_idm_video_condition(gt, first)
            torch.testing.assert_close(cond[:, :, :1], first, atol=0, rtol=0)
            torch.testing.assert_close(gt, original, atol=0, rtol=0)
            if probability == 0:
                self.assertEqual(t.count_nonzero(), 0)
                torch.testing.assert_close(cond, gt, atol=0, rtol=0)
            elif probability == 1:
                self.assertTrue((t > 0).all())
            else:
                self.assertTrue(0 < stats["idm_cond_noisy_fraction"] < 1)
            torch.manual_seed(31)
            again = model._sample_idm_video_condition(gt, first)
            torch.testing.assert_close(cond, again[0], atol=0, rtol=0)
            self.assertTrue(torch.is_grad_enabled())
        for probability in (-1, 1.01, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "prob"):
                make_model().configure_future_video(future_training_mode="idm", idm_video_cond_noise_prob=probability)
        with self.assertRaisesRegex(ValueError, "requires future"):
            make_model("current_only").configure_future_video(future_training_mode="idm")

    def test_bf16_condition_timestep_matches_actual_sigma(self):
        model = idm_model()
        model.idm_video_cond_noise_prob = 1.0
        gt = torch.full((2, 4, 3, 4, 4), 0.25, dtype=torch.bfloat16)
        first = gt[:, :, :1].clone()
        noise = torch.ones_like(gt)
        sampled = torch.tensor([356.73, 811.29])
        with patch.object(model.train_video_scheduler, "sample_training_t", return_value=sampled), \
             patch("torch.randn_like", return_value=noise):
            cond, timestep, stats = model._sample_idm_video_condition(gt, first)
        sigma = (timestep / model.train_video_scheduler.num_train_timesteps).reshape(2, 1, 1, 1, 1)
        expected = (1 - sigma) * gt + sigma * noise
        expected[:, :, :1] = first
        torch.testing.assert_close(cond, expected, atol=0, rtol=0)
        self.assertEqual(stats["idm_cond_sigma_mean"], float(sigma.float().mean()))
        with torch.no_grad():
            self.assertFalse(model._sample_idm_video_condition(gt, first)[0].requires_grad)

    def test_action_only_reaches_full_condition_kv_and_shared_video(self):
        model, data = idm_model(), sample(batch=2)
        model.loss_lambda_video = 0
        captured = []
        editor = model.mot.chunk_kv_cache_editor
        original = editor.build_layer_updated_cache

        def record(**kwargs):
            for name in ("first_frame_keys", "first_frame_values"):
                kwargs[name].retain_grad()
            captured.append(kwargs)
            return original(**kwargs)

        with patch.object(editor, "build_layer_updated_cache", side_effect=record), \
             patch.object(model, "_generate_future_video", side_effect=AssertionError("IDM sampled video")):
            loss, metrics = model.training_loss(data)
            loss.backward()
        self.assertEqual(metrics["loss_video"], 0)
        for layer in captured:
            self.assertEqual(layer["first_frame_keys"].shape[1], 12)
            for name in ("first_frame_keys", "first_frame_values"):
                grad = layer[name].grad[:, 4:]
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum(), 0)
        for name in ("k", "v"):
            grad = getattr(model.video_expert.blocks[1].self_attn, name).weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.norm(), 0)
        self.assertGreater(model.action_expert.blocks[1].self_attn.v.weight.grad.norm(), 0)
        self.assertGreater(editor.layer_delta_proj[1][-1].weight.grad.norm(), 0)

    def test_stream_independence_layer_count_and_planner_steps(self):
        model, data = idm_model(), sample(batch=2)
        state_keys = set(model.state_dict())
        original = model._sample_idm_video_condition
        records = []
        pre = model.video_expert.pre_dit

        def record(**kwargs):
            result = pre(**kwargs)
            records.append((kwargs, result))
            return result

        torch.manual_seed(80)
        with patch.object(model.video_expert, "pre_dit", side_effect=record), \
             patch.object(model.mot, "_build_expert_attention_io", wraps=model.mot._build_expert_attention_io) as io:
            loss, metrics = model.training_loss(data)
            self.assertEqual(io.call_count, 3 * model.mot.num_layers)
        self.assertEqual(len(records), 2)
        self.assertIsNot(records[0][1]["tokens"], records[1][1]["tokens"])
        self.assertFalse(torch.equal(records[0][0]["timestep"], records[1][0]["timestep"]))
        torch.testing.assert_close(records[0][0]["x"][:, :, :1], records[1][0]["x"][:, :, :1])
        torch.testing.assert_close(records[0][1]["t_mod"][:, :4], records[1][1]["t_mod"][:, :4])

        def different_condition(*args):
            cond, t, stats = original(*args)
            cond[:, :, 1:] += 3
            return cond, t, stats

        torch.manual_seed(80)
        with patch.object(model, "_sample_idm_video_condition", side_effect=different_condition):
            _, changed = model.training_loss(data)
        self.assertEqual(metrics["loss_video"], changed["loss_video"])
        self.assertNotEqual(metrics["loss_action"], changed["loss_action"])
        model.video_plan_steps = 99
        torch.manual_seed(80)
        repeated, _ = model.training_loss(data)
        torch.testing.assert_close(loss, repeated, atol=0, rtol=0)
        self.assertEqual(state_keys, set(model.state_dict()))

    def test_full_layer_checkpoint_matches_all_gradients_and_frozen_dino(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                a = idm_model(dino=True).to(dtype=dtype)
                a.torch_dtype = dtype
                a.dino_encoder.float()  # The fake encoder receives raw FP32 images, like the real wrapper.
                # A zero-initialized editor intentionally blocks upstream query gradients initially.
                for proj in a.mot.chunk_kv_cache_editor.layer_delta_proj:
                    torch.nn.init.normal_(proj[-1].weight, std=0.01)
                b = copy.deepcopy(a)
                b.idm_checkpoint_full_layer = True
                b.mot.mot_checkpoint_mixed_attn = True
                data = sample(batch=2)
                results = []
                for model in (a, b):
                    torch.manual_seed(45)
                    loss, _ = model.training_loss(data)
                    loss.backward()
                    results.append(loss)
                tolerance = 1e-6 if dtype == torch.float32 else 1e-3
                torch.testing.assert_close(*results, atol=tolerance, rtol=tolerance)
                for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
                    self.assertEqual(pa.grad is None, pb.grad is None, name)
                    if pa.grad is not None:
                        torch.testing.assert_close(pa.grad, pb.grad, atol=tolerance, rtol=tolerance, msg=name)
                for module in (a.action_obs_visual_proj, a.chunk_obs_query_encoder, a.proprio_encoder):
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in a.dino_encoder.parameters()))

    def test_padding_masks_editor_and_both_streams(self):
        model, data = idm_model(), sample(batch=2)
        data["image_is_pad"][:, 5:] = True
        inputs = model.build_inputs(data)
        valid = model._idm_video_valid_keys(inputs, 4)
        self.assertTrue(valid[:, :8].all())
        self.assertFalse(valid[:, 8:].any())
        # Perturb post-VAE padding only, leaving OVCR observations and temporal VAE boundaries unchanged.
        changed = dict(inputs, input_latents=inputs["input_latents"].clone())
        changed["input_latents"][:, :, 2:] += 100
        losses = []
        for prepared in (inputs, changed):
            torch.manual_seed(66)
            with patch.object(model, "build_inputs", return_value=prepared):
                loss, metrics = model.training_loss(data)
                losses.append(loss)
                self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(*losses, atol=1e-6, rtol=1e-6)
        data["image_is_pad"][:, 5] = False  # The group is valid unless ALL sampled images are padded.
        inputs["image_is_pad"] = data["image_is_pad"]
        self.assertTrue(model._idm_video_valid_keys(inputs, 4).all())
        data["image_is_pad"][:, 0] = True
        with self.assertRaisesRegex(ValueError, "valid clean first"):
            model._idm_video_valid_keys(inputs, 4)

    def test_fm_and_other_action_chunks_are_invisible_to_action(self):
        model, data = idm_model(), sample(batch=2)
        forward = model.mot.forward_idm_action_with_chunk_updated_kv

        def check(**kwargs):
            reference = forward(**kwargs)
            fm_pre = dict(kwargs["video_fm_pre"], tokens=kwargs["video_fm_pre"]["tokens"] + 3)
            changed = forward(**{**kwargs, "video_fm_pre": fm_pre})
            torch.testing.assert_close(reference["action_prior"], changed["action_prior"], atol=0, rtol=0)
            self.assertGreater((reference["video"] - changed["video"]).abs().max(), 0)
            action = kwargs["action_pre"]["tokens"].clone()
            action[:, :16] += 3
            changed = forward(**{**kwargs, "action_pre": dict(kwargs["action_pre"], tokens=action)})
            torch.testing.assert_close(reference["action_prior"][:, 16:], changed["action_prior"][:, 16:], atol=0, rtol=0)
            torch.testing.assert_close(reference["video"], changed["video"], atol=0, rtol=0)
            return reference

        with patch.object(model.mot, "forward_idm_action_with_chunk_updated_kv", side_effect=check):
            model.training_loss(data)[0].backward()

    def test_history_validity_and_positions_match_between_streams(self):
        model, data = idm_model(checkpoint=True), sample(batch=2)
        model.configure_chunk_history(num_history_frames=1, action_video_read_mode="future")
        data.update(video_history=torch.randn(2, 3, 1, 16, 16),
                    video_history_valid_len=torch.tensor([0, 1]),
                    video_history_frame_indices=torch.tensor([[0], [4]]),
                    video_current_frame_index=torch.tensor([8, 8]),
                    video_temporal_position_ids=torch.tensor([[8, 12, 16], [8, 12, 16]]))
        original = model.video_expert.pre_dit
        positions = []

        def record(**kwargs):
            if kwargs["x"].shape[2] == 3:
                positions.append(kwargs["temporal_position_ids"].clone())
            return original(**kwargs)

        torch.manual_seed(75)
        with patch.object(model.video_expert, "pre_dit", side_effect=record):
            expected, _ = model.training_loss(data)
        torch.testing.assert_close(positions[0], positions[1], atol=0, rtol=0)
        data["video_history"][0] += 20
        torch.manual_seed(75)
        actual, _ = model.training_loss(data)
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        actual.backward()
        self.assertTrue(torch.isfinite(actual))

    def test_video_only_loss_ignores_condition_and_editor_base_is_immutable(self):
        model, data = idm_model(), sample(batch=2)
        model.loss_lambda_action_prior = 0
        loss, _ = model.training_loss(data)
        loss.backward()
        self.assertGreater(model.video_expert.blocks[1].self_attn.v.weight.grad.norm(), 0)
        self.assertTrue(all(p.grad is None or not p.grad.count_nonzero() for p in model.action_expert.parameters()))
        editor = model.mot.chunk_kv_cache_editor
        k, v = torch.randn(2, 12, 48), torch.randn(2, 12, 48)
        before_k, before_v = k.clone(), v.clone()
        queries = torch.randn(2, 4, 3, 32)
        torch.nn.init.normal_(editor.layer_delta_proj[0][-1].weight, std=0.01)
        updated = editor.build_layer_updated_cache(layer_idx=0, chunk_queries=queries,
                                                  first_frame_keys=k, first_frame_values=v)
        for chunk in range(4):
            single = editor.build_layer_updated_cache(layer_idx=0, chunk_queries=queries[:, chunk:chunk + 1],
                                                     first_frame_keys=k, first_frame_values=v)
            torch.testing.assert_close(single["k"][:, 0], updated["k"][:, chunk])
        torch.testing.assert_close(k, before_k, atol=0, rtol=0)
        torch.testing.assert_close(v, before_v, atol=0, rtol=0)

    def test_offset_layout_and_no_generated_fallback(self):
        model, data = idm_model(horizon=96, dino=True), sample(frames=13, actions=96, batch=2)
        model.max_action_offset = 15
        data["action_offset"] = torch.tensor([1, 15])
        data["chunk_obs_images"] = torch.randn(2, 4, 3, 16, 16)
        with patch.object(model, "_generate_future_video", side_effect=AssertionError("IDM sampled")):
            loss, metrics = model.training_loss(data)
            loss.backward()
            self.assertEqual(metrics["mean_offset"], 8)
        bad = sample(batch=2)
        bad["action_video_freq_ratio"][:] = 4
        with self.assertRaisesRegex(ValueError, "action_video_freq_ratio"):
            idm_model().training_loss(bad)
        inputs = model.build_inputs(data)
        with self.assertRaisesRegex(ValueError, "exceeds video plan"):
            model._validate_training_future_layout(inputs=inputs, sample=data, action_horizon=96)
        with self.assertRaisesRegex(ValueError, "latent frames"):
            idm_model()._validate_training_future_layout(inputs=inputs, sample=data, action_horizon=64)

    def test_video_only_all_action_padding_and_freeze_rejection(self):
        model, data = idm_model(), sample(batch=2)
        data["action_is_pad"][:] = True
        loss, metrics = model.training_loss(data)
        loss.backward()
        self.assertEqual(metrics["loss_action"], 0)
        self.assertGreater(model.video_expert.blocks[1].self_attn.v.weight.grad.norm(), 0)
        with self.assertRaisesRegex(ValueError, "freeze_video_dit"):
            Wan22Trainer._apply_dit_only_train_mode(model, freeze_video_dit=True)
        model.video_expert.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "trainable shared"):
            model.training_loss(data)

    def test_recipe_weight_only_resume_and_no_mutation(self):
        model = idm_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            opt = torch.optim.AdamW(model.parameters())
            model.training_loss(sample())[0].backward()
            opt.step()
            model.save_checkpoint(path, optimizer=opt, step=1)
            other = make_model()
            other.load_checkpoint(path)  # Inference behavior is identical, recipe need not match.
            self.assertEqual(other.checkpoint_behavior(), model.checkpoint_behavior())
            before = {k: v.clone() for k, v in other.state_dict().items()}
            other_opt = torch.optim.AdamW(other.parameters())
            with self.assertRaisesRegex(ValueError, "recipe"):
                other.load_checkpoint(path, optimizer=other_opt)
            self.assertFalse(other_opt.state)
            for key, value in before.items():
                torch.testing.assert_close(other.state_dict()[key], value, atol=0, rtol=0)
            model.idm_checkpoint_full_layer = not model.idm_checkpoint_full_layer
            model.load_checkpoint(path, optimizer=opt)
            metadata = {"behavior_config": model.checkpoint_behavior()}
            (Path(directory) / "trainer_state.json").write_text(json.dumps(metadata))
            trainer = object.__new__(Wan22Trainer)
            with self.assertRaisesRegex(ValueError, "recipe"):
                trainer._read_trainer_state_metadata(directory, model=model)
            trainer.model = model
            with patch.object(model, "load_state_dict", side_effect=AssertionError("Mutated before validation")):
                with self.assertRaisesRegex(ValueError, "recipe"):
                    trainer._load_model_weights_from_ddp_state(Path(directory))
            metadata["training_recipe"] = model.training_recipe()
            (Path(directory) / "trainer_state.json").write_text(json.dumps(metadata))
            trainer._read_trainer_state_metadata(directory, model=model)

    def test_optimizer_rng_resume_matches_next_step(self):
        model, data = idm_model(checkpoint=True), sample(batch=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.training_loss(data)[0].backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model.save_checkpoint(path, optimizer=optimizer, step=1)
            rng = torch.get_rng_state()
            reference, metrics = model.training_loss(data)
            reference.backward()
            optimizer.step()
            restored = idm_model(checkpoint=True)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
            restored.load_checkpoint(path, optimizer=restored_optimizer)
            torch.set_rng_state(rng)
            actual, restored_metrics = restored.training_loss(data)
            actual.backward()
            restored_optimizer.step()
            self.assertEqual(metrics, restored_metrics)
            torch.testing.assert_close(reference, actual, atol=0, rtol=0)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)

    def test_eval_teacher_loss_generated_inference_and_mode_invariance(self):
        model, data = idm_model(), sample()
        data["prompt"] = ["test task"]
        model._evaluate_action_metrics = lambda **kwargs: {"action_l1": 1.0, "action_l2": 2.0}
        with patch.object(model, "_generate_future_video", wraps=model._generate_future_video) as generate:
            result = model.evaluate_validation(Wan22Trainer._to_batched_eval_sample(data), eval_num_inference_steps=1)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(result["val_idm_loss"], result["val_loss"])
        self.assertEqual(result["val_generated_action_l1"], 1.0)
        states = []
        for mode in ("generated", "idm"):
            model.future_training_mode = mode
            states.append(model.prefill_video(input_image=data["video"][:, :, 0], action_horizon=64,
                                             context=data["context"], context_mask=data["context_mask"], seed=21))
        for a, b in zip(states[0]["video_kv_cache"], states[1]["video_kv_cache"]):
            for key in ("k", "v"):
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
                self.assertFalse(b[key].requires_grad)


if __name__ == "__main__":
    unittest.main()
