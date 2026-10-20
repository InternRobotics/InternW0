"""Validate native checkpoint semantics before loading shape-compatible weights."""

import logging
from pathlib import Path

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _contract(*, mode, horizon, chunk, history, rope_stride, video_mask,
              action_conditioned, teacher_forcing, action_mask, obs_downsample,
              plan_horizon, video_stride, temporal_factor):
    future = mode == "future"
    return {
        "schema_version": 3,
        "num_soft_prompts": 0,
        "domain_configs": [],
        "soft_prompt_domains": [],
        "soft_prompt_layout": "per_chunk_prefix",
        "soft_prompt_position_mode": "identity_rope",
        "soft_prompt_timestep_mode": "follow_chunk",
        "observation_encoder": "vae",
        "dino_encoder": None,
        "architecture": "w0_one_way_ovcr",
        "action_video_read_mode": str(mode),
        "video_cache_mode": "generated_future_last_denoise" if future else "observed_frames",
        "action_position_unit": "action_step",
        "video_rope_position_unit": "sampled_video_frame" if history else "latent_frame",
        "action_horizon": int(horizon),
        "action_chunk_size": int(chunk),
        "num_history_frames": int(history),
        "video_rope_frame_stride": int(rope_stride) if history else None,
        "video_attention_mask_mode": str(video_mask),
        "video_action_conditioned": bool(action_conditioned),
        "autoregressive_teacher_forcing": bool(teacher_forcing),
        "action_teacher_forcing_mask_mode": str(action_mask),
        "action_obs_downsample_factor": int(obs_downsample),
        "video_plan_horizon": int(plan_horizon) if future else None,
        "video_action_stride": int(video_stride) if future else None,
        "vae_temporal_downsample_factor": int(temporal_factor),
    }


def model_behavior(model):
    contract = _contract(
        mode=getattr(model, "action_video_read_mode", "current_only"),
        horizon=model.action_horizon, chunk=model.action_chunk_size,
        history=getattr(model, "num_history_frames", 0),
        rope_stride=getattr(model, "video_rope_frame_stride", 1),
        video_mask=model.video_expert.video_attention_mask_mode,
        action_conditioned=model.video_expert.action_conditioned,
        teacher_forcing=model.action_expert.autoregressive_teacher_forcing,
        action_mask=model.action_expert.action_teacher_forcing_mask_mode,
        obs_downsample=model.action_obs_downsample_factor,
        plan_horizon=getattr(model, "video_plan_horizon", model.action_horizon),
        video_stride=getattr(model, "video_action_stride", 8),
        temporal_factor=model.vae.temporal_downsample_factor,
    )
    contract["observation_encoder"] = getattr(model, "observation_encoder", "vae")
    contract.update(model.action_expert.soft_prompt_behavior())
    if any("action_valid_indices" in d or "proprio_valid_indices" in d
           for d in model.action_expert.domain_configs):
        contract["schema_version"] = 4
    if contract["observation_encoder"] == "dino":
        contract["dino_encoder"] = model.dino_encoder.behavior_config()
    return contract


def legacy_config_behavior(path):
    cfg = OmegaConf.load(path)
    cfg = cfg.get("model", cfg)
    target = str(cfg.get("_target_", ""))
    legacy_target = "aha" + "wam.runtime.create_" + "aha" + "wam"
    if target == legacy_target:
        target = "w0.runtime.create_w0"
    if target != "w0.runtime.create_w0":
        raise ValueError(f"Legacy behavior config must describe create_w0: {path}")
    if cfg.get("observation_encoder", "vae") != "vae":
        raise ValueError("Legacy DINO checkpoints require embedded encoder identity; cannot infer it from shape/config alone.")
    video, action = cfg["video_dit_config"], cfg["action_dit_config"]
    horizon = int(cfg["action_horizon"])
    # Defaults mirror create_w0 and the expert constructors, not the target
    # model. Inferring these values from the target would hide mismatches.
    return _contract(
        mode=cfg.get("action_video_read_mode", "current_only"),
        horizon=horizon, chunk=cfg["action_chunk_size"],
        history=cfg.get("num_history_frames", 0),
        rope_stride=cfg.get("video_rope_frame_stride", 1),
        video_mask=video.get("video_attention_mask_mode", "bidirectional"),
        action_conditioned=video.get("action_conditioned", False),
        teacher_forcing=action.get("autoregressive_teacher_forcing", False),
        action_mask=action.get("action_teacher_forcing_mask_mode", "stage1_chunkwise"),
        obs_downsample=cfg.get("action_obs_downsample_factor", 2),
        plan_horizon=cfg.get("video_plan_horizon") or horizon,
        video_stride=cfg.get("video_action_stride", 8),
        temporal_factor=4,  # create_w0 loads the Wan 2.2 VAE.
    )


def _find_legacy_config(path):
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    candidates = [directory / "config.yaml"]
    # Known trainer layout only; never pick an unrelated ancestor's config.
    if (path.parent.name in ("weights", "state")
            and path.parent.parent.name == "checkpoints"):
        candidates.append(path.parent.parent.parent / "config.yaml")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def validate_checkpoint_behavior(model, payload, path, *, legacy_config=None,
                                 allow_observation_encoder_migration=False,
                                 allow_soft_prompt_init=False):
    saved = payload.get("behavior_config")
    if "behavior_config" not in payload:
        legacy_config = (legacy_config or getattr(model, "legacy_checkpoint_config", None)
                         or _find_legacy_config(path))
        if legacy_config is None:
            raise ValueError(
                f"Checkpoint {path} has no behavior_config or companion training config. "
                "Supply its original resolved training YAML through legacy_config=... "
                "or model.legacy_checkpoint_config; weight shapes alone cannot establish compatibility."
            )
        saved = legacy_config_behavior(legacy_config)
        logger.warning("Legacy checkpoint behavior is declared by %s (not embedded in weights). "
                       "Re-save the validated checkpoint to embed this contract.", legacy_config)
    if not isinstance(saved, dict) or saved.get("schema_version") not in (1, 2, 3, 4):
        raise ValueError("Unsupported or malformed checkpoint behavior_config schema.")
    architecture = str(saved.get("architecture", ""))
    legacy_prefix = "aha" + "wam_"
    if architecture.startswith(legacy_prefix):
        architecture = "w0_" + architecture[len(legacy_prefix):]
        saved = dict(saved, architecture=architecture)
    if saved["schema_version"] == 1:
        # Version 1 predates the DINO path and can only describe VAE observations.
        saved = dict(saved, schema_version=2, observation_encoder="vae", dino_encoder=None)
    if saved["schema_version"] == 2:
        if saved.get("num_soft_prompts", 0) != 0:
            raise ValueError("Legacy schema cannot describe enabled soft prompts.")
        saved = dict(saved, schema_version=3, num_soft_prompts=0, soft_prompt_domains=[],
                     domain_configs=[],
                     soft_prompt_layout="per_chunk_prefix", soft_prompt_position_mode="identity_rope",
                     soft_prompt_timestep_mode="follow_chunk")
    expected = model_behavior(model)
    if allow_soft_prompt_init:
        if saved.get("domain_configs", []) != expected["domain_configs"]:
            raise ValueError("Soft prompt initialization cannot change adapter banks; migrate the domain bank explicitly.")
        if saved.get("num_soft_prompts") != 0 or expected["num_soft_prompts"] <= 0:
            raise ValueError("Soft prompt initialization requires a K=0 source and a K>0 target.")
        saved = dict(saved, **model.action_expert.soft_prompt_behavior())
    migrate = (allow_observation_encoder_migration and saved.get("observation_encoder") == "vae"
               and expected["observation_encoder"] == "dino")
    if migrate:
        saved = dict(saved, observation_encoder="dino", dino_encoder=expected["dino_encoder"])
    differences = [f"{key}: checkpoint={saved.get(key)!r}, model={value!r}"
                   for key, value in expected.items() if key not in saved or saved[key] != value]
    if differences:
        raise ValueError("Checkpoint behavior mismatch (no weights loaded): " + "; ".join(differences))
    return migrate
