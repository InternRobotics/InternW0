"""One-way future-video planning for the existing AHA video/action experts."""

from __future__ import annotations

import math

import torch


class FutureVideoMixin:
    def uses_future_video(self) -> bool:
        return self._configured_action_video_read_mode() == "future"

    def configure_future_video(
        self, *, video_plan_horizon: int | None = None,
        video_action_stride: int = 8, video_plan_steps: int = 10,
        future_training_mode: str = "generated", idm_video_cond_noise_prob: float = 0.5,
        idm_checkpoint_full_layer: bool = True,
        idm_checkpoint_layer_stride: int = 1,
    ) -> None:
        self.video_plan_horizon = int(self.action_horizon if video_plan_horizon is None else video_plan_horizon)
        self.video_action_stride = int(video_action_stride)
        self.video_plan_steps = int(video_plan_steps)
        self.future_training_mode = str(future_training_mode)
        self.idm_video_cond_noise_prob = float(idm_video_cond_noise_prob)
        self.idm_checkpoint_full_layer = bool(idm_checkpoint_full_layer)
        if int(idm_checkpoint_layer_stride) != idm_checkpoint_layer_stride or idm_checkpoint_layer_stride < 1:
            raise ValueError("idm_checkpoint_layer_stride must be a positive integer.")
        self.idm_checkpoint_layer_stride = int(idm_checkpoint_layer_stride)
        if self.future_training_mode not in {"generated", "idm"}:
            raise ValueError("future_training_mode must be generated or idm.")
        if not math.isfinite(self.idm_video_cond_noise_prob) or not 0 <= self.idm_video_cond_noise_prob <= 1:
            raise ValueError("idm_video_cond_noise_prob must be finite and in [0, 1].")
        if self.uses_idm_training() and not self.uses_future_video():
            raise ValueError("IDM training requires future video reading.")
        if self.uses_future_video():
            self._future_video_layout()
            if self.video_plan_steps <= 0:
                raise ValueError("video_plan_steps must be positive.")
            if self.video_plan_horizon < self.action_horizon:
                raise ValueError("video_plan_horizon must cover action_horizon.")
            if getattr(self.video_expert, "action_conditioned", False):
                raise ValueError("Future video planning must not be action-conditioned.")

    def _future_video_layout(self) -> tuple[int, int]:
        factor = int(self.vae.temporal_downsample_factor)
        stride = int(self.video_action_stride)
        horizon = int(self.video_plan_horizon)
        if factor <= 0 or stride <= 0 or horizon <= 0 or horizon % (factor * stride):
            raise ValueError(
                "video_plan_horizon must be positive and divisible by "
                "video_action_stride * VAE temporal_downsample_factor."
            )
        return horizon // stride + 1, horizon // (stride * factor) + 1

    @torch.no_grad()
    def _generate_future_video(
        self, *, first_frame_latents, context, context_mask,
        temporal_position_ids=None, prefix_cache=None, attention_mask=None,
        seed=None, rand_device="cpu",
    ) -> dict:
        sampled_frames, latent_frames = self._future_video_layout()
        shape = list(first_frame_latents.shape)
        if shape[2] != 1:
            raise ValueError("Future planning needs exactly one clean observation latent.")
        shape[2] = latent_frames
        generator = None
        if seed is not None:
            generator = torch.Generator(device=rand_device).manual_seed(int(seed))
        latents = torch.randn(shape, generator=generator, device=rand_device).to(
            device=first_frame_latents.device, dtype=first_frame_latents.dtype,
        )
        latents[:, :, :1] = first_frame_latents
        timesteps, deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=self.video_plan_steps,
            device=latents.device, dtype=latents.dtype,
        )
        for timestep, delta in zip(timesteps, deltas):
            pre = self.video_expert.pre_dit(
                x=latents, timestep=timestep.expand(latents.shape[0]),
                context=context, context_mask=context_mask, action=None,
                fuse_vae_embedding_in_latents=True, clean_prefix_frames=1,
                temporal_position_ids=temporal_position_ids,
            )
            if attention_mask is None:
                attention_mask = self.video_expert.build_video_to_video_mask(
                    video_seq_len=pre["tokens"].shape[1],
                    video_tokens_per_frame=pre["meta"]["tokens_per_frame"],
                    device=latents.device,
                )
                if prefix_cache is not None:
                    prefix_len = prefix_cache[0]["k"].shape[1]
                    attention_mask = torch.cat([
                        torch.ones((attention_mask.shape[0], prefix_len),
                                   dtype=torch.bool, device=latents.device),
                        attention_mask,
                    ], dim=1)
            velocity, cache = self.mot.forward_video_plan(
                video_pre=pre, video_attention_mask=attention_mask,
                prefix_cache=prefix_cache,
            )
            latents = self.infer_video_scheduler.step(velocity, delta, latents)
            latents[:, :, :1] = first_frame_latents
            callback = getattr(self, "_video_plan_step_callback", None)
            if callback is not None:
                callback()
        return {
            "video_pre": pre,
            "video_seq_len": int(pre["tokens"].shape[1]),
            "video_tokens_per_frame": int(pre["meta"]["tokens_per_frame"]),
            "video_kv_cache": cache,
            "video_plan_latents": latents,
            "video_plan_horizon": self.video_plan_horizon,
            "video_sampled_frames": sampled_frames,
            "video_latent_frames": latent_frames,
            "video_action_stride": self.video_action_stride,
            "video_cache_sigma": float(timesteps[-1].float().item())
                / self.infer_video_scheduler.num_train_timesteps,
        }

    def _validate_training_future_layout(self, *, inputs, sample, action_horizon):
        sampled_frames, latent_frames = self._future_video_layout()
        actual_frames = int(inputs["input_latents"].shape[2])
        if actual_frames != latent_frames:
            raise ValueError(
                f"Training clip has {actual_frames} latent frames, expected {latent_frames} "
                f"({sampled_frames} sampled images) for video_plan_horizon={self.video_plan_horizon}."
            )
        # Shape alone cannot distinguish, e.g., 32 steps at stride 4 from 64 at 8.
        stride = sample.get("action_video_freq_ratio")
        if stride is None or not bool((torch.as_tensor(stride) == self.video_action_stride).all()):
            raise ValueError("Sample action_video_freq_ratio must match video_action_stride.")
        offsets = self._normalize_action_offsets(sample, batch_size=inputs["action"].shape[0])
        if int(offsets.max().item()) + action_horizon > self.video_plan_horizon:
            raise ValueError("Offset action window exceeds video plan coverage; extend the video horizon.")

    def _training_future_cache(
        self, *, inputs, sample, action_horizon, temporal_position_ids,
        history_video_kv_cache, video_attention_mask,
    ):
        if not self.uses_future_video():
            return None
        self._validate_training_future_layout(inputs=inputs, sample=sample, action_horizon=action_horizon)
        # Only the clean first observation and past memory enter this rollout.
        # The noisy GT future remains exclusive to the existing video FM loss.
        plan = self._generate_future_video(
            first_frame_latents=inputs["first_frame_latents"],
            context=inputs["context"], context_mask=inputs["context_mask"],
            temporal_position_ids=temporal_position_ids,
            prefix_cache=history_video_kv_cache,
            attention_mask=video_attention_mask,
            rand_device=str(inputs["first_frame_latents"].device),
        )
        return plan["video_kv_cache"]

    def uses_idm_training(self):
        return getattr(self, "future_training_mode", "generated") == "idm"

    def validate_idm_trainability(self):
        if self.uses_idm_training() and not all(p.requires_grad for p in self.video_expert.parameters()):
            raise ValueError("IDM requires a trainable shared VideoDiT; action-to-video gradients must not be frozen.")

    def _sample_idm_video_condition(self, input_latents, first_frame_latents):
        batch = input_latents.shape[0]
        noisy = torch.rand(batch, device=input_latents.device) < self.idm_video_cond_noise_prob
        sampled = self.train_video_scheduler.sample_training_t(
            batch_size=batch, device=input_latents.device, dtype=torch.float32,
        )
        # Match the timestep actually seen by pre_dit and add_noise in BF16 too.
        timestep = torch.where(noisy, sampled, 0).to(input_latents.dtype)
        condition = self.train_video_scheduler.add_noise(input_latents, torch.randn_like(input_latents), timestep)
        condition[:, :, :1] = first_frame_latents
        sigma = (timestep / self.train_video_scheduler.num_train_timesteps).float()
        return condition, timestep, {
            "idm_cond_noisy_fraction": float(noisy.float().mean()),
            "idm_cond_sigma_mean": float(sigma.mean()),
            "idm_cond_sigma_max": float(sigma.max()),
        }

    def _idm_video_valid_keys(self, inputs, tokens_per_frame):
        latents = inputs["input_latents"]
        pad = inputs["image_is_pad"]
        if pad is None:
            return torch.ones((latents.shape[0], latents.shape[2] * tokens_per_frame),
                              device=latents.device, dtype=torch.bool)
        factor = int(self.vae.temporal_downsample_factor)
        if pad.shape != (latents.shape[0], 1 + factor * (latents.shape[2] - 1)):
            raise ValueError("IDM image padding must match the sampled video / latent layout.")
        if bool(pad[:, 0].any()):
            raise ValueError("IDM requires a valid clean first frame.")
        latent_pad = torch.cat([pad[:, :1], pad[:, 1:].reshape(pad.shape[0], -1, factor).all(-1)], dim=1)
        return (~latent_pad).repeat_interleave(tokens_per_frame, dim=1)

    def _forward_training_video_action(
        self, *, inputs, sample, action_horizon, video_pre, action_pre,
        video_attention_mask, chunk_queries, temporal_position_ids, history_video_kv_cache,
    ):
        if self.uses_idm_training():
            self._validate_training_future_layout(inputs=inputs, sample=sample, action_horizon=action_horizon)
            if torch.is_grad_enabled():
                self.validate_idm_trainability()
            condition, t_cond, metrics = self._sample_idm_video_condition(
                inputs["input_latents"], inputs["first_frame_latents"],
            )
            cond_pre = self.video_expert.pre_dit(
                x=condition, timestep=t_cond, context=inputs["context"],
                context_mask=inputs["context_mask"], action=None,
                fuse_vae_embedding_in_latents=True, clean_prefix_frames=1,
                temporal_position_ids=temporal_position_ids,
                projected_context=video_pre["context"],
            )
            valid = self._idm_video_valid_keys(inputs, int(video_pre["meta"]["tokens_per_frame"]))
            prefix_len = video_attention_mask.shape[-1] - valid.shape[-1]
            all_valid = torch.cat([torch.ones((valid.shape[0], prefix_len), device=valid.device, dtype=torch.bool), valid], dim=1)
            mask = video_attention_mask & all_valid[:, None, :]
            return self.mot.forward_idm_action_with_chunk_updated_kv(
                video_fm_pre=video_pre, video_cond_pre=cond_pre, action_pre=action_pre,
                video_attention_mask=mask, chunk_queries=chunk_queries,
                action_chunk_size=action_pre["meta"].get("tokens_per_chunk", self.action_chunk_size), history_video_kv_cache=history_video_kv_cache,
                video_valid_key_mask=valid, checkpoint_full_layer=self.idm_checkpoint_full_layer,
                checkpoint_layer_stride=self.idm_checkpoint_layer_stride,
            ), metrics
        return self.mot.forward_prior_action_with_chunk_updated_kv(
            video_tokens=video_pre["tokens"], video_freqs=video_pre["freqs"], video_t_mod=video_pre["t_mod"],
            video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
            video_attention_mask=video_attention_mask, action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"], action_t_mod=action_pre["t_mod"],
            action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
            chunk_queries=chunk_queries, video_tokens_per_frame=video_pre["meta"]["tokens_per_frame"],
            action_chunk_size=action_pre["meta"].get("tokens_per_chunk", self.action_chunk_size), history_video_kv_cache=history_video_kv_cache,
            action_video_kv_cache=self._training_future_cache(
                inputs=inputs, sample=sample, action_horizon=action_horizon,
                temporal_position_ids=temporal_position_ids, history_video_kv_cache=history_video_kv_cache,
                video_attention_mask=video_attention_mask,
            ),
        ), {}

    def _idm_loss_metrics(self, total, video, action):
        if not self.uses_idm_training():
            return {}
        return {"val_idm_loss": float(total.detach()), "val_idm_video_loss": float(video.detach()),
                "val_idm_action_loss": float(action.detach())}

    def bind_training_data_recipe(self, train_dataset, val_dataset=None):
        if hasattr(getattr(train_dataset, "processor", None), "statistics"):
            self._posttrain_statistics = train_dataset.processor.statistics
        def offset_config(dataset):
            if dataset is None:
                return None
            fields = ("max_action_offset", "action_horizon", "action_chunk_size", "action_video_freq_ratio")
            config = {key: int(getattr(dataset, key)) for key in fields if hasattr(dataset, key)}
            if hasattr(dataset, "_action_offset_enabled"):
                config["action_offset_enabled"] = bool(dataset._action_offset_enabled)
            if self.action_expert.uses_domain_conditioning:
                if self.action_expert.domain_configs and getattr(dataset, "domain_configs", None) != self.action_expert.domain_configs:
                    raise ValueError("Dataset and model domain banks must have identical order and native schemas.")
                ids = getattr(dataset, "domain_ids", [getattr(dataset, "domain_id", None)])
                for domain in ids:
                    self.action_expert.normalize_domain_id(domain, batch_size=1, device="cpu")
                config["domains"] = getattr(dataset, "domain_data_recipe", {"domain_ids": list(ids)})
            return config
        recipe = {"train": offset_config(train_dataset), "val": offset_config(val_dataset)}
        expected = dict(max_action_offset=self.max_action_offset, action_horizon=self.action_horizon,
                        action_chunk_size=self.action_chunk_size)
        if self.uses_future_video():
            expected.update(video_plan_horizon=self.video_plan_horizon,
                            action_video_freq_ratio=self.video_action_stride)
        for split, dataset in (("train", train_dataset), ("val", val_dataset)):
            for key, value in expected.items():
                if hasattr(dataset, key) and getattr(dataset, key) != value:
                    raise ValueError(f"{split} dataset {key} must match the model for exact training semantics.")
        self._training_data_recipe = recipe

    def training_recipe(self):
        def scheduler_config(scheduler):
            return {"type": type(scheduler).__module__ + "." + type(scheduler).__qualname__,
                    "shift": scheduler.shift, "num_train_timesteps": scheduler.num_train_timesteps}
        recipe = {
            "version": 2, "future_training_mode": getattr(self, "future_training_mode", "generated"),
            "max_action_offset": int(getattr(self, "max_action_offset", 0)),
            "data_offset_config": getattr(self, "_training_data_recipe", None),
            "idm_video_cond_noise_prob": getattr(self, "idm_video_cond_noise_prob", 0.5),
            "idm_checkpoint_full_layer": getattr(self, "idm_checkpoint_full_layer", True),
            "idm_checkpoint_layer_stride": getattr(self, "idm_checkpoint_layer_stride", 1),
            "video_scheduler": scheduler_config(self.train_video_scheduler),
            "action_scheduler": scheduler_config(self.train_action_scheduler),
            "loss_lambda_video": self.loss_lambda_video,
            "loss_lambda_action_prior": self.loss_lambda_action_prior,
            "action_timestep_mode": self.action_train_timestep_mode,
        }
        if self.action_expert.uses_domain_conditioning:
            recipe["soft_prompt_training"] = getattr(self, "_soft_prompt_training_recipe", None)
        return recipe

    def validate_training_recipe(self, payload):
        saved = payload.get("training_recipe")
        if saved is None and not self.uses_idm_training():
            return  # Preserve legacy generated-mode resume.
        current = self.training_recipe()
        # Recompute policy is provenance, not a change in the statistical objective.
        recompute_keys = {"idm_checkpoint_full_layer", "idm_checkpoint_layer_stride"}
        strip = lambda recipe: {k: v for k, v in recipe.items() if k not in recompute_keys}
        if not isinstance(saved, dict) or strip(saved) != strip(current):
            raise ValueError("Training recipe mismatch or missing training_recipe; use weight-only initialization "
                             "instead of optimizer/full-state resume when changing objectives or offset sampling.")

    def _align_future_action_positions(self, action_pre, sample):
        if not self.uses_future_video():
            return
        batch = action_pre["tokens"].shape[0]
        length = action_pre["meta"]["seq_len"]
        offsets = self._normalize_action_offsets(sample, batch_size=batch)
        positions = offsets[:, None] + torch.arange(length, device=offsets.device)[None, :]
        if int(positions.max()) >= self.action_expert.freqs.shape[0]:
            raise ValueError("Offset actions exceed the ActionDiT RoPE table.")
        action_pre["freqs"] = self.action_expert.pack_action_freqs(
            self.action_expert.freqs.to(offsets.device)[positions].unsqueeze(2), action_pre["meta"],
        )

    def _validate_plan_chunk(self, inference_state, chunk_index):
        if not self.uses_future_video():
            return
        horizon = int(inference_state.get("video_plan_horizon", 0))
        offset = int(inference_state.get("action_position_offset", 0))
        end = offset + (int(chunk_index) + 1) * self.action_chunk_size
        if offset < 0 or chunk_index < 0 or end > horizon:
            raise ValueError(f"Action chunk ends at {end}, beyond video coverage {horizon}; refresh video.")
        cache = inference_state["video_kv_cache"]
        expected = int(inference_state["video_seq_len"])
        if expected <= int(inference_state["video_tokens_per_frame"]):
            raise ValueError("Future video cache contains only the current frame.")
        if any(layer["k"].shape[1] != expected or layer["v"].shape[1] != expected for layer in cache):
            raise ValueError("Future video K/V length does not match the published plan.")
