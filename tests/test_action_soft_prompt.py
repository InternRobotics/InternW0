"""Soft prompts through real tiny ActionDiT/MoT/IDM and cached inference paths."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir

from w0.trainer import Wan22Trainer, group_action_prompt_parameters
from w0.datasets.domain import validate_domain_id
from test_future_video import make_model, sample


def prompt_model(*, domains=("a", "b"), k=8, mode="idm", horizon=64):
    model = make_model(horizon=horizon, action_kwargs={
        "num_soft_prompts": k, "soft_prompt_domains": list(domains),
    })
    model.configure_future_video(video_plan_horizon=horizon, future_training_mode=mode)
    Wan22Trainer._apply_dit_only_train_mode(model)
    return model


class ActionSoftPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_prompt_preserves_rng_parameters_and_loss(self):
        a = make_model()
        rng_a = torch.get_rng_state()
        b = make_model(action_kwargs={"num_soft_prompts": 0, "soft_prompt_domains": []})
        torch.testing.assert_close(rng_a, torch.get_rng_state(), atol=0, rtol=0)
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        data = sample()
        losses = []
        for model in (a, b):
            torch.manual_seed(77)
            losses.append(model.training_loss(data)[0])
        torch.testing.assert_close(*losses, atol=0, rtol=0)

    def test_packing_time_rope_masks_and_parameter_sharing(self):
        expert = prompt_model().action_expert
        data = sample(batch=2)
        times = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80.]])
        state = expert.pre_dit(action_tokens=data["action"], timestep=times,
                              context=data["context"], domain_id=torch.tensor([0, 1]))
        self.assertEqual(state["tokens"].shape, (2, 96, 32))
        self.assertEqual(expert.soft_prompt_hub.weight.numel(), 2 * 8 * 32)
        blocks = state["tokens"].reshape(2, 4, 24, 32)
        for c in range(4):
            torch.testing.assert_close(blocks[:, c, :8], expert.soft_prompt_hub.weight.reshape(2, 8, 32))
        mod = state["t_mod"].reshape(2, 4, 24, 6, 32)
        torch.testing.assert_close(mod[:, :, :8], mod[:, :, 8:9].expand(-1, -1, 8, -1, -1))
        self.assertFalse(state["self_attn_mask"][:24, 24:].any())
        self.assertTrue(state["self_attn_mask"][:24, :24].all())
        for offset in (0, 5, 15):
            original = expert.freqs[offset:offset + 64, None]
            freq = expert.pack_action_freqs(original, state["meta"]).reshape(4, 24, 1, -1)
            torch.testing.assert_close(freq[:, :8], torch.ones_like(freq[:, :8]), atol=0, rtol=0)
            torch.testing.assert_close(freq[:, 8:].reshape_as(original), original, atol=0, rtol=0)
        self.assertEqual(expert.post_dit(state["tokens"], state).shape, (2, 64, 4))

    def test_domains_timestep_and_teacher_contracts(self):
        expert = prompt_model().action_expert
        kwargs = dict(action_tokens=torch.randn(2, 64, 4), timestep=torch.ones(2),
                      context=torch.randn(2, 3, 32))
        for domain in (None, -1, 2, 0.5, True, [0., 1.], [0, 1, 0]):
            with self.subTest(domain=domain), self.assertRaises(ValueError):
                expert.pre_dit(**kwargs, domain_id=domain)
        with self.assertRaisesRegex(ValueError, "timestep"):
            expert.pre_dit(**dict(kwargs, timestep=torch.arange(64.)[None].expand(2, -1)), domain_id=0)
        with self.assertRaisesRegex(ValueError, "teacher"):
            expert.pre_dit(**kwargs, domain_id=0, clean_action_tokens=kwargs["action_tokens"], clean_timestep=torch.zeros(2))
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                validate_domain_id(value)

    def test_idm_gradient_checkpoint_and_video_independence(self):
        for dtype in (torch.float32, torch.bfloat16):
            a = prompt_model().to(dtype=dtype)
            a.torch_dtype = dtype
            for proj in a.mot.chunk_kv_cache_editor.layer_delta_proj:
                torch.nn.init.normal_(proj[-1].weight, std=0.01)
            b = copy.deepcopy(a)
            b.idm_checkpoint_full_layer = False
            data = dict(sample(batch=2), domain_id=torch.tensor([0, 1]))
            values = []
            for model in (a, b):
                model.loss_lambda_video = 0
                torch.manual_seed(8)
                loss, _ = model.training_loss(data)
                loss.backward()
                values.append(loss)
                for module in (model.action_expert.soft_prompt_hub, model.video_expert.blocks[0],
                               model.action_expert.blocks[0], model.chunk_obs_query_encoder):
                    gradients = [p.grad for p in module.parameters() if p.grad is not None]
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
            tol = 2e-3 if dtype == torch.bfloat16 else 1e-6
            torch.testing.assert_close(*values, atol=tol, rtol=tol)
            for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
                self.assertEqual(pa.grad is None, pb.grad is None, name)
                if pa.grad is not None:
                    torch.testing.assert_close(pa.grad, pb.grad, atol=tol, rtol=tol, msg=name)
        model = prompt_model()
        metrics = []
        for domain in (0, 1):
            torch.manual_seed(91)
            metrics.append(model.training_loss(dict(data, domain_id=torch.full((2,), domain)))[1])
        self.assertEqual(metrics[0]["loss_video"], metrics[1]["loss_video"])
        self.assertNotEqual(metrics[0]["loss_action"], metrics[1]["loss_action"])
        model.zero_grad(set_to_none=True)
        model.loss_lambda_action_prior = 0
        model.training_loss(dict(data, domain_id=torch.tensor([0, 1])))[0].backward()
        grad = model.action_expert.soft_prompt_hub.weight.grad
        self.assertTrue(grad is None or not grad.count_nonzero())

    def test_parallel_chunk_equivalence_and_isolation(self):
        model = prompt_model().eval()
        data = dict(sample(batch=2), domain_id=torch.tensor([0, 1]))
        # Capture the actual full-future OVCR KV once, then hold it fixed.
        inputs = model.build_inputs(data)
        _, _, proprio, proprio_mask = model._split_visual_obs_and_proprio_context(
            obs_context=inputs["obs_context"], obs_context_mask=inputs["obs_context_mask"])
        cache = []
        editor = model.mot.chunk_kv_cache_editor
        original_editor = editor.build_layer_updated_cache
        def record(**kwargs):
            out = original_editor(**kwargs)
            cache.append({k: v.detach() for k, v in out.items()})
            return out
        with patch.object(editor, "build_layer_updated_cache", side_effect=record):
            model.training_loss(data)
        state = dict(editor_cache=cache, context=inputs["context"], context_mask=inputs["context_mask"],
                     proprio_context=proprio, proprio_mask=proprio_mask, domain_id=data["domain_id"])
        action, times = torch.randn(2, 64, 4), torch.tensor([[10., 20, 30, 40], [15., 25, 35, 45]])
        full = model._predict_action_flow_with_video_state(action, times, state)
        pieces = []
        for c in range(4):
            local = dict(state, proprio_context=state["proprio_context"][:, c:c+1],
                         proprio_mask=state["proprio_mask"][:, c:c+1])
            local["editor_cache"] = [{key: value[:, c:c+1] for key, value in layer.items()}
                                     for layer in state["editor_cache"]]
            # Preserve physical positions while using local context routing.
            original = model.action_expert.pre_dit
            def pre(**kwargs):
                out = original(**kwargs)
                out["freqs"] = model.action_expert.pack_action_freqs(
                    model.action_expert.freqs[c * 16:(c + 1) * 16, None], out["meta"])
                return out
            with patch.object(model.action_expert, "pre_dit", side_effect=pre):
                pieces.append(model._predict_action_flow_with_video_state(action[:, c*16:(c+1)*16], times[:, c], local))
        torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=2e-6, rtol=2e-6)
        changed = action.clone()
        changed[:, 16:32] += 10
        other = model._predict_action_flow_with_video_state(changed, times, state)
        torch.testing.assert_close(full[:, :16], other[:, :16], atol=0, rtol=0)
        torch.testing.assert_close(full[:, 32:], other[:, 32:], atol=0, rtol=0)

    def test_cached_inference_domain_and_offset(self):
        model, data = prompt_model(horizon=96), sample()
        prefill = dict(input_image=data["video"][:, :, 0], action_horizon=64,
                       context=data["context"], context_mask=data["context_mask"], seed=3,
                       action_position_offset=5)
        with self.assertRaisesRegex(ValueError, "domain_id"):
            model.prefill_video(**prefill)
        a = model.prefill_video(**prefill, domain_id=0)
        b = model.prefill_video(**prefill, domain_id=1)
        for x, y in zip(a["video_kv_cache"], b["video_kv_cache"]):
            torch.testing.assert_close(x["k"], y["k"], atol=0, rtol=0)
        kwargs = dict(chunk_obs_image=data["video"][:, :, 0], chunk_proprio=data["proprio"][:, 0],
                      chunk_index=1, num_inference_steps=2)
        with self.assertRaisesRegex(ValueError, "Cannot change"):
            model.infer_action_chunk(inference_state=a, domain_id=1, **kwargs)
        captured = []
        original = model.action_expert.post_dit
        def post(tokens, pre):
            captured.append(pre)
            return original(tokens, pre)
        with patch.object(model.action_expert, "post_dit", side_effect=post):
            out = model.infer_action_chunk(inference_state=b, **kwargs)
        self.assertEqual(out["action_chunk"].shape, (16, 4))
        self.assertEqual(len(captured), 2)
        for pre in captured:
            self.assertEqual(pre["tokens"].shape[1], 24)
            torch.testing.assert_close(pre["freqs"][8:], model.action_expert.freqs[21:37, None])
        single = prompt_model(domains=("only",))
        self.assertEqual(int(single.action_expert.normalize_domain_id(None, batch_size=1, device="cpu", allow_default=True)), 0)

    def test_checkpoint_roundtrip_migration_and_corruption(self):
        model = prompt_model()
        data = dict(sample(), domain_id=torch.tensor([1]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save_checkpoint(path)
            restored = prompt_model()
            restored.load_checkpoint(path)
            values = []
            for m in (model, restored):
                torch.manual_seed(60)
                values.append(m.training_loss(data)[0])
            torch.testing.assert_close(*values, atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "behavior"):
                prompt_model(domains=("b", "a")).load_checkpoint(path)
            payload = torch.load(path, weights_only=False)
            del payload["mot"]["mixtures.action.soft_prompt_hub.weight"]
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "complete MoT"):
                restored.load_checkpoint(path)
            old = prompt_model(k=0)
            old.save_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "behavior"):
                restored.load_checkpoint(path)
            initial = restored.action_expert.soft_prompt_hub.weight.detach().clone()
            restored.load_checkpoint(path, allow_soft_prompt_init=True)
            torch.testing.assert_close(initial, restored.action_expert.soft_prompt_hub.weight, atol=0, rtol=0)
            for key, value in old.mot.state_dict().items():
                torch.testing.assert_close(value, restored.mot.state_dict()[key], atol=0, rtol=0)

    def test_lr_ratio_optimizer_update_and_resume(self):
        a, b = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(4))
        def build(params):
            trainer = object.__new__(Wan22Trainer)
            trainer.learning_rate = 5e-5
            groups = group_action_prompt_parameters(params, params[1:], lr=5e-5, lr_scale=0.1, max_numel=3)
            trainer.optimizer = torch.optim.AdamW(groups, weight_decay=0)
            trainer.scheduler = trainer._build_scheduler("cosine", 20, 4)
            return trainer
        trainer = build([a, b])
        for step in range(20):
            if step == 9:
                new = build([torch.nn.Parameter(a.detach().clone()), torch.nn.Parameter(b.detach().clone())])
                new.optimizer.load_state_dict(trainer.optimizer.state_dict())
                new.scheduler.load_state_dict(trainer.scheduler.state_dict())
            for obj in ([trainer, new] if step >= 9 else [trainer]):
                for group in obj.optimizer.param_groups:
                    for p in group["params"]:
                        p.grad = torch.ones_like(p)
                obj.optimizer.step()
                obj.scheduler.step()
                rates = [g["lr"] for g in obj.optimizer.param_groups]
                self.assertAlmostEqual(rates[1] / rates[0], 0.1)
            if step >= 9:
                self.assertEqual([g["lr"] for g in trainer.optimizer.param_groups],
                                 [g["lr"] for g in new.optimizer.param_groups])
        self.assertLess(float(b.mean()), 1)
        self.assertAlmostEqual(trainer.optimizer.param_groups[1]["lr"], 5e-8)

    def test_task_config(self):
        with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
            cfg = compose(config_name="pretrain")
        self.assertEqual(cfg.model.action_dit_config.num_soft_prompts, 8)
        self.assertEqual(cfg.model.action_chunk_size, 16)
        self.assertEqual(cfg.soft_prompt_lr_scale, 0.1)


if __name__ == "__main__":
    unittest.main()
