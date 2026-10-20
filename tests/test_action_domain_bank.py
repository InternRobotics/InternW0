"""Mixed native dimensions, domain routing, and reader contracts."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from w0.datasets.domain_registry import read_registry, episode_is_validation
from w0.datasets.domain_video_dataset import DomainNormalizer
from w0.trainer import Wan22Trainer
from w0.utils.samplers import ResumableEpochSampler
from test_future_video import make_model, sample


CONFIGS = [dict(name="set/arm", dataset="set", embodiment="arm", action_dim=2, proprio_dim=2),
           dict(name="set/dual", dataset="set", embodiment="dual", action_dim=4, proprio_dim=4)]


def bank_model(configs=CONFIGS, k=8):
    model = make_model(action_kwargs={"num_soft_prompts": k, "domain_configs": configs},
                       action_dim=max(d["action_dim"] for d in configs),
                       proprio_dim=max(d["proprio_dim"] for d in configs))
    model.configure_future_video(future_training_mode="idm")
    Wan22Trainer._apply_dit_only_train_mode(model)
    return model


class ActionDomainBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mixed_domains_padding_loss_and_gradients(self):
        for k in (0, 8):
            model = bank_model(k=k)
            data = dict(sample(batch=2), domain_id=torch.tensor([0, 1]))
            altered = copy.deepcopy(data)
            altered["action"][0, :, 2:] = 1000
            altered["proprio"][0, :, 2:] = -1000
            losses = []
            for value in (data, altered):
                torch.manual_seed(123)
                losses.append(model.training_loss(value)[0])
            torch.testing.assert_close(*losses, atol=0, rtol=0)
            losses[0].backward()
            for module in (model.action_expert.action_encoder, model.action_expert.head, model.proprio_encoder):
                self.assertTrue(torch.isfinite(module.weight.grad).all())
                self.assertTrue(all(float(g.abs().sum()) > 0 for g in module.weight.grad))
            self.assertEqual(float(model.action_expert.action_encoder.weight.grad[0, :, 2:].abs().sum()), 0)
            self.assertEqual(float(model.action_expert.head.weight.grad[0, 2:].abs().sum()), 0)
            self.assertEqual(float(model.proprio_encoder.weight.grad[0, :, 2:].abs().sum()), 0)

    def test_mixed_adapter_matches_individual_routing(self):
        model = bank_model()
        expert = model.action_expert
        data = sample(batch=2)
        kwargs = dict(action_tokens=data["action"], timestep=torch.ones(2), context=data["context"], domain_id=torch.tensor([0, 1]))
        mixed = expert.pre_dit(**kwargs)
        decoded = expert.post_dit(mixed["tokens"], mixed)
        self.assertEqual(float(decoded[0, :, 2:].abs().sum()), 0)
        for i in range(2):
            state = expert.pre_dit(**{k: v[i:i + 1] for k, v in kwargs.items()})
            torch.testing.assert_close(mixed["tokens"][i:i + 1], state["tokens"])
            torch.testing.assert_close(decoded[i:i + 1], expert.post_dit(state["tokens"], state))

    def test_cached_inference_video_has_no_domain_condition(self):
        for k in (0, 8):
            model, data = bank_model(k=k), sample()
            states = [model.prefill_video(input_image=data["video"][:, :, 0], action_horizon=64,
                                         context=data["context"], context_mask=data["context_mask"],
                                         seed=7, domain_id=i) for i in (0, 1)]
            torch.testing.assert_close(states[0]["video_plan_latents"], states[1]["video_plan_latents"], atol=0, rtol=0)
            kwargs = dict(chunk_index=0, chunk_obs_image=data["video"][:, :, 0],
                          chunk_proprio=data["proprio"][:, 0], num_inference_steps=1)
            result = model.infer_action_chunk(inference_state=states[0], **kwargs)
            self.assertEqual(float(result["action_chunk"][:, 2:].abs().sum()), 0)
            with self.assertRaisesRegex(ValueError, "domain_id"):
                model.infer_action_chunk(inference_state=states[1], domain_id=0, **kwargs)

    def test_registry_and_hydra_task(self):
        root = Path(__file__).resolve().parents[1]
        registry = read_registry(root / "configs/data/domain_banks/internw0_domains.json")
        self.assertEqual(len(registry["domain_configs"]), 11)
        self.assertEqual(len(registry["sources"]), 1982)
        xsens = next(d for d in registry["domain_configs"] if d["embodiment"] == "tienkung_xsens_1rgb")
        self.assertEqual(xsens["action_dim"], 26)
        self.assertEqual(xsens["action_fields"][-1]["dim"], 12)
        with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
            cfg = compose(config_name="pretrain")
            resolved = OmegaConf.to_container(cfg, resolve=True)
        self.assertEqual(resolved["data"]["train"]["_target_"], "w0.datasets.domain_video_dataset.DomainVideoDataset")
        self.assertNotIn("processor", resolved["data"]["train"])
        self.assertFalse(resolved["model"]["video_dit_config"]["action_conditioned"])
        self.assertEqual(resolved["model"]["action_dit_config"]["num_soft_prompts"], 8)

    def test_epoch_and_split_determinism(self):
        class Dataset:
            uses_epoch_indices = True
            def __len__(self): return 11
        sampler = ResumableEpochSampler(Dataset(), seed=42, batch_size=1, num_processes=1)
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)
        self.assertTrue(all(epoch == 1 for index, epoch in second))
        self.assertNotEqual(first, second)
        sampler.set_epoch(0)
        sampler.set_epoch_offset(1)
        self.assertEqual(second, list(sampler))
        for ep in range(100):
            a = dict(root="RoboMIND2/benchmark1_0_compressed/franka_1rgb/task")
            b = dict(root="RoboMIND2/benchmark1_1_compressed/franka_3rgb/task")
            self.assertEqual(episode_is_validation(a, ep), episode_is_validation(b, ep))

    def test_eval_keeps_domain_and_uses_native_dimensions(self):
        data = sample()
        data["prompt"] = ["move"]
        data["domain_id"] = torch.tensor(0)
        batched = Wan22Trainer._to_batched_eval_sample(data)
        self.assertEqual(batched["domain_id"].tolist(), [0])
        stats = {d["name"]: {kind: dict(mean=[1.] * d[dim], std=[2.] * d[dim])
                            for kind, dim in (("action", "action_dim"), ("proprio", "proprio_dim"))} for d in CONFIGS}
        normalizer = DomainNormalizer(CONFIGS, stats)
        x = torch.randn(5, 4)
        torch.testing.assert_close(normalizer.transform(normalizer.transform(x, 0, "action"), 0, "action", inverse=True), x[:, :2])


if __name__ == "__main__":
    unittest.main()
