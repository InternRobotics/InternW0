"""LIBERO closed-loop adapter: 10-D EEF through w0 37D left-arm slots."""
from __future__ import annotations

from collections import deque
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from w0.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from w0.datasets.libero_eef import eef10_to_osc7, obs_to_eef10
from w0.datasets.robotwin_posttrain import camera_canvas

logger = logging.getLogger(__name__)


def _assert_compatible_statistics(local: dict, saved: dict) -> None:
    if local.get("recipe") != saved.get("recipe"):
        raise ValueError("Deployment statistics recipe differs from the checkpoint.")
    for kind in ("action", "proprio"):
        for key in ("q01", "q99"):
            if local[kind][key] != saved[kind][key]:
                raise ValueError(f"Deployment {kind}.{key} differs from the checkpoint.")


def _compose(name: str, overrides: list[str] | None = None) -> DictConfig:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str((PROJECT_ROOT / "configs").resolve())):
        return compose(config_name=name, overrides=overrides or [])


def _checkpoint_action_chunk_size(checkpoint_path: str | Path) -> int:
    """Read the deployment chunk contract before constructing the model."""
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True,
    )
    behavior = payload.get("behavior_config")
    if not isinstance(behavior, dict) or "action_chunk_size" not in behavior:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no embedded action_chunk_size; "
            "pass action_chunk_size explicitly or re-save it with behavior_config."
        )
    value = int(behavior["action_chunk_size"])
    if value < 1:
        raise ValueError(f"Invalid checkpoint action_chunk_size={value}.")
    return value


class LiberoPosttrainPolicy:
    def __init__(self, model_cfg, processor_cfg, checkpoint_path, dataset_stats_path,
                 device, model_dtype, action_horizon, chunks_per_video_prefill=1,
                 num_inference_steps=10, sigma_shift=None, seed=None, text_cache_dir=None,
                 image_transform="rotate_180", execute_n_actions=None):
        model_cfg = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg.load_text_encoder = text_cache_dir is None
        model_cfg.tokenizer_max_len = 128
        self.model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
        loaded = self.model.load_checkpoint(checkpoint_path)
        saved_statistics = loaded.get("posttrain_statistics")
        del loaded
        self.model = self.model.to(device).eval()
        processor_cfg = OmegaConf.create(OmegaConf.to_container(processor_cfg, resolve=True))
        processor_cfg.stats_path = str(dataset_stats_path)
        self.processor = instantiate(processor_cfg).eval()
        if saved_statistics is not None:
            _assert_compatible_statistics(self.processor.statistics, saved_statistics)
        self.action_horizon = int(action_horizon)
        if self.action_horizon != int(self.model.action_horizon):
            raise ValueError("Deployment action_horizon must match the training model.")
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cache_dir = Path(text_cache_dir) if text_cache_dir else None
        self.image_transform = image_transform
        self.execute_n_actions = None if execute_n_actions in (None, 0) else int(execute_n_actions)
        self.num_chunks = int(self.action_horizon // self.model.action_chunk_size)
        self.chunks_per_video_prefill = int(chunks_per_video_prefill)
        self.pending_actions: deque[np.ndarray] = deque()
        self._prompt_cache = {}
        self._episode_prefilled = False
        self._chunks_since_video_prefill = 0
        self._warmup()

    def _warmup(self) -> None:
        dummy = torch.zeros((1, 3, 384, 320), device=self.model.device, dtype=self.model.torch_dtype)
        proprio = torch.zeros((1, 37), device=self.model.device, dtype=torch.float32)
        warmup = {
            "prompt": None,
            "context": torch.zeros(128, 4096, device=self.model.device, dtype=self.model.torch_dtype),
            "context_mask": torch.ones(128, device=self.model.device, dtype=torch.bool),
        }
        with torch.no_grad():
            self.model.infer_action(
                **warmup, input_image=dummy, action_horizon=self.action_horizon,
                num_inference_steps=self.num_inference_steps, sigma_shift=self.sigma_shift,
                seed=0, phase="video", domain_id=0)
            self.model.infer_action(
                prompt=None, input_image=dummy, action_horizon=self.action_horizon,
                chunk_obs_image=dummy, chunk_proprio=proprio,
                num_inference_steps=self.num_inference_steps, sigma_shift=self.sigma_shift,
                seed=0, phase="action", domain_id=0)
        self.reset()

    def _prompt_kwargs(self, prompt: str) -> dict[str, Any]:
        if self.text_cache_dir is None:
            return {"prompt": prompt}
        if prompt not in self._prompt_cache:
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            path = self.text_cache_dir / f"{digest}.t5_len128.wan22ti2v5b.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing UMT5 cache for {prompt!r}: {path}")
            value = torch.load(path, map_location="cpu", weights_only=True)
            context, mask = value["context"].clone(), value["mask"].bool()
            context[~mask] = 0
            self._prompt_cache[prompt] = (context, torch.ones_like(mask))
        context, mask = self._prompt_cache[prompt]
        return {
            "prompt": None,
            "context": context.to(device=self.model.device, dtype=self.model.torch_dtype),
            "context_mask": mask.to(device=self.model.device),
        }

    def reset(self) -> None:
        if hasattr(self.model, "reset_history"):
            self.model.reset_history()
        if hasattr(self.model, "_inference_state"):
            self.model._inference_state = None
        self.pending_actions.clear()
        self._episode_prefilled = False
        self._chunks_since_video_prefill = 0

    def _reset_for_new_observation(self) -> None:
        """Start a fresh closed-loop plan from the current observation.

        LIBERO is configured without observation history.  Clearing only
        ``_inference_state`` leaves W0's video frame counter / KV history
        alive, so every replan advances temporal RoPE until it eventually
        overflows and conditions actions on stale observations.
        """
        if hasattr(self.model, "reset_history"):
            self.model.reset_history()
        if hasattr(self.model, "_inference_state"):
            self.model._inference_state = None
        self._episode_prefilled = False
        self._chunks_since_video_prefill = 0

    def _canvas(self, obs: dict) -> torch.Tensor:
        tensors = []
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            frame = np.asarray(obs[key])
            if np.issubdtype(frame.dtype, np.floating):
                if float(np.nanmax(frame)) <= 1.5:
                    frame = frame * 255.0
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            elif frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if self.image_transform == "rotate_180":
                frame = np.ascontiguousarray(frame[::-1, ::-1])
            tensors.append(torch.as_tensor(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0))
        tensors.append(torch.zeros_like(tensors[1]))
        return camera_canvas(tensors).to(device=self.model.device)

    def _proprio(self, obs: dict) -> torch.Tensor:
        native = obs_to_eef10(obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"])
        packed, _ = self.processor.pack(torch.as_tensor(native).float().unsqueeze(0), "proprio")
        return packed.to(device=self.model.device)

    def _predict_chunk(self, obs: dict, instruction: str) -> np.ndarray:
        image = self._canvas(obs)
        proprio = self._proprio(obs)
        prompt = DEFAULT_PROMPT.format(task=instruction)
        with torch.no_grad():
            if not self._episode_prefilled:
                self.model.infer_action(
                    **self._prompt_kwargs(prompt), input_image=image, action_horizon=self.action_horizon,
                    num_inference_steps=self.num_inference_steps, sigma_shift=self.sigma_shift,
                    seed=self.seed, phase="video", domain_id=0)
                self._episode_prefilled = True
                self._chunks_since_video_prefill = 0
            pred = self.model.infer_action(
                prompt=None, input_image=image, action_horizon=self.action_horizon,
                chunk_obs_image=image, chunk_proprio=proprio,
                num_inference_steps=self.num_inference_steps, sigma_shift=self.sigma_shift,
                seed=self.seed, phase="action", domain_id=0)
        packed = pred["action_chunk"].unsqueeze(0).float().cpu()
        native = self.processor.unpack_action(packed)[0].numpy()
        actions = np.stack([eef10_to_osc7(row) for row in native])
        if self.execute_n_actions is not None:
            actions = actions[: max(1, min(self.execute_n_actions, len(actions)))]
        return actions

    def act(self, obs: dict, instruction: str) -> np.ndarray:
        if not self.pending_actions:
            next_index = 0
            if getattr(self.model, "_inference_state", None) is not None:
                next_index = int(self.model._inference_state.get("next_chunk_index", 0))
            if next_index >= self.chunks_per_video_prefill or self._chunks_since_video_prefill >= self.chunks_per_video_prefill:
                self._reset_for_new_observation()
            for action in self._predict_chunk(obs, instruction):
                self.pending_actions.append(np.asarray(action, dtype=np.float32))
            self._chunks_since_video_prefill += 1
        return self.pending_actions.popleft()


def get_model(usr_args: dict[str, Any]) -> LiberoPosttrainPolicy:
    cfg = usr_args.get("cfg")
    if cfg is None:
        action_chunk_size = usr_args.get("action_chunk_size")
        if action_chunk_size is None:
            action_chunk_size = _checkpoint_action_chunk_size(usr_args["ckpt"])
        cfg = _compose(
            "eval_libero",
            overrides=[f"model.action_chunk_size={int(action_chunk_size)}"],
        )
    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    mixed = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision") or "bf16")
    dtype = {"no": torch.float32, "fp16": torch.float16}.get(mixed, torch.bfloat16)
    # Keep deployment deterministic by default, matching both this config and
    # the reference evaluator.  Previously serve_libero_policy always
    # supplied ``seed=None``, silently overriding cfg.seed=42 and making every
    # closed-loop replan draw unrelated diffusion noise.
    seed = usr_args["seed"] if "seed" in usr_args else cfg.get("seed")
    return LiberoPosttrainPolicy(
        model_cfg=cfg.model, processor_cfg=cfg.data.train.processor,
        checkpoint_path=usr_args["ckpt"],
        dataset_stats_path=usr_args.get("dataset_stats_path") or cfg.EVALUATION.dataset_stats_path,
        device=device, model_dtype=dtype,
        action_horizon=int(usr_args.get("action_horizon") or cfg.EVALUATION.action_horizon or cfg.model.action_horizon),
        chunks_per_video_prefill=int(usr_args.get("chunks_per_video_prefill") or cfg.EVALUATION.chunks_per_video_prefill or 1),
        num_inference_steps=int(usr_args.get("num_inference_steps") or cfg.EVALUATION.num_inference_steps or 10),
        sigma_shift=usr_args.get("sigma_shift"),
        seed=None if seed is None else int(seed),
        text_cache_dir=usr_args.get("text_cache_dir") or cfg.EVALUATION.get("text_cache_dir"),
        image_transform=str(cfg.EVALUATION.get("image_transform", "rotate_180")),
        execute_n_actions=usr_args.get("execute_n_actions", cfg.EVALUATION.get("execute_n_actions")),
    )
