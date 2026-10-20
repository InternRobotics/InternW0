"""Numerical and collective contracts for training hot-path optimizations."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from accelerate.utils import DistributedType

from w0.trainer import Wan22Trainer
from test_idm_training import idm_model
from test_future_video import sample


class TrainingEfficiencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_frame_timestep_expansion_matches_legacy_and_reduces_projection_rows(self):
        model = idm_model().video_expert
        for prefix in (0, 1, 3, 7):
            inputs = dict(x=torch.randn(2, 4, 3, 4, 4), timestep=torch.tensor([0., 725.]),
                          context=torch.randn(2, 3, 32), fuse_vae_embedding_in_latents=True,
                          clean_prefix_frames=prefix)
            outputs, rows = [], []
            hook = model.time_embedding.register_forward_pre_hook(
                lambda module, args: rows.append(args[0].shape[0]))
            try:
                for compact in (False, True):
                    model.compact_timestep_embedding = compact
                    outputs.append(model.pre_dit(**inputs))
            finally:
                hook.remove()
            self.assertEqual(rows, [24, 6])
            for key in ('tokens', 't', 't_mod', 'context', 'freqs'):
                torch.testing.assert_close(outputs[0][key], outputs[1][key])

    def test_idm_loss_and_gradients_with_legacy_timestep_path(self):
        for dtype in (torch.float32, torch.bfloat16):
            for checkpoint in (False, True):
                with self.subTest(dtype=dtype, checkpoint=checkpoint):
                    old = idm_model(checkpoint=checkpoint).to(dtype=dtype)
                    old.torch_dtype = dtype
                    old.video_expert.compact_timestep_embedding = False
                    original_pre = old.video_expert.pre_dit
                    new = copy.deepcopy(old)
                    new.video_expert.compact_timestep_embedding = True
                    data = sample(batch=2)
                    losses = []
                    for model in (old, new):
                        torch.manual_seed(129)
                        def legacy_pre(**kwargs):
                            kwargs.pop('projected_context', None)
                            return original_pre(**kwargs)
                        if model is old:
                            with patch.object(old.video_expert, 'pre_dit', side_effect=legacy_pre):
                                loss, _ = model.training_loss(data)
                                loss.backward()
                        else:
                            loss, _ = model.training_loss(data)
                            loss.backward()
                        losses.append(loss)
                    torch.testing.assert_close(*losses, rtol=1e-4, atol=1e-5)
                    self.assertEqual(set(old.state_dict()), set(new.state_dict()))
                    for (name, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
                        self.assertEqual(a.grad is None, b.grad is None, name)
                        if a.grad is None:
                            continue
                        self.assertTrue(torch.isfinite(b.grad).all(), name)
                        # BF16 changes reduction order when shared activations expand.
                        tolerance = 0.02 if dtype == torch.bfloat16 else 2e-5
                        error = (a.grad.float() - b.grad.float()).norm()
                        scale = a.grad.float().norm().clamp_min(1e-6)
                        self.assertLessEqual(float(error / scale), tolerance, name)

    def test_idm_shares_text_projections_but_not_video_states(self):
        model = idm_model()
        counts = {'text': 0, 'k': 0, 'v': 0}
        def count(name):
            def hook(module, args):
                counts[name] += 1
            return hook
        hooks = [model.video_expert.text_embedding.register_forward_pre_hook(count('text'))]
        for block in model.video_expert.blocks:
            hooks.extend([block.cross_attn.k.register_forward_pre_hook(count('k')),
                          block.cross_attn.v.register_forward_pre_hook(count('v'))])
        try:
            model.training_loss(sample(batch=2))[0].backward()
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(counts, {'text': 1, 'k': 3, 'v': 3})

    def test_partial_checkpoint_preserves_gradients_and_resume_recipe(self):
        for dtype in (torch.float32, torch.bfloat16):
            model = idm_model(checkpoint=True).to(dtype=dtype)
            model.torch_dtype = dtype
            partial = copy.deepcopy(model)
            partial.idm_checkpoint_layer_stride = 2
            recipe = model.training_recipe()
            # Old checkpoints have no stride; recomputation is not the objective.
            recipe.pop('idm_checkpoint_layer_stride')
            partial.validate_training_recipe({'training_recipe': recipe})
            data = sample(batch=2)
            losses = []
            for current in (model, partial):
                torch.manual_seed(129)
                loss, _ = current.training_loss(data)
                loss.backward()
                losses.append(loss)
            torch.testing.assert_close(*losses, rtol=0, atol=0)
            for (name, a), (_, b) in zip(model.named_parameters(), partial.named_parameters()):
                self.assertEqual(a.grad is None, b.grad is None, name)
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, msg=name)
        for stride in (0, -1, 1.5):
            with self.assertRaisesRegex(ValueError, 'stride'):
                model.configure_future_video(idm_checkpoint_layer_stride=stride)

    def test_metrics_use_one_collective_and_keep_rank_mean(self):
        calls = []
        def gather(packed):
            calls.append(packed)
            return torch.cat([packed, packed + 2], dim=0)
        trainer = object.__new__(Wan22Trainer)
        trainer.accelerator = SimpleNamespace(gather=gather)
        loss = torch.tensor(3., requires_grad=True)
        result = trainer._gather_training_metrics(loss, {'z': 7., 'a': torch.tensor(5.)}, 9.)
        self.assertEqual(result, (4., {'a': 6., 'z': 8.}, 10.))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].shape, (1, 4))
        self.assertFalse(calls[0].requires_grad)

    def test_gc_cadence_and_profile_accounting(self):
        trainer = object.__new__(Wan22Trainer)
        trainer.gc_collect_every = 3
        trainer._profile_window = trainer._reset_profile_window()
        with patch('w0.trainer.gc.collect') as collect:
            for step in range(1, 8):
                trainer.global_step = step
                trainer._collect_step_garbage()
        self.assertEqual(collect.call_count, 2)
        trainer._update_profile_window(data_time=1, forward_time=2, backward_time=3,
                                       optimizer_time=4, gc_time=0.5)
        self.assertEqual(trainer._consume_profile_window()['gc_time'], 0.5)
        self.assertEqual(trainer._profile_window['steps'], 0)

    def test_deepspeed_norm_stays_on_device_and_other_backends_still_clip(self):
        trainer = object.__new__(Wan22Trainer)
        trainer.max_grad_norm = 1.0
        norm = torch.tensor(3.)
        trainer.model = Mock()
        trainer.model.get_global_grad_norm.return_value = norm
        trainer.accelerator = Mock(distributed_type=DistributedType.DEEPSPEED)
        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('device sync')):
            self.assertIs(trainer._clip_grad_norm(), norm)
        trainer.accelerator.clip_grad_norm_.assert_not_called()
        trainer.accelerator.distributed_type = DistributedType.NO
        trainer._clip_grad_norm()
        trainer.accelerator.clip_grad_norm_.assert_called_once_with(
            trainer.model.parameters.return_value, 1.0)


if __name__ == '__main__':
    unittest.main()
