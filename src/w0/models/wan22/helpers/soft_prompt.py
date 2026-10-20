"""Action-only domain prompts; all public lengths remain physical action steps."""

import math
from numbers import Integral

import torch
from torch import nn


class ActionSoftPromptMixin:
    def _init_soft_prompts(self, num_soft_prompts, domains, init_std):
        if isinstance(num_soft_prompts, bool) or not isinstance(num_soft_prompts, Integral) or num_soft_prompts < 0:
            raise ValueError("num_soft_prompts must be a nonnegative integer.")
        self.num_soft_prompts = int(num_soft_prompts)
        if isinstance(domains, str):
            raise ValueError("soft_prompt_domains must be an ordered list of unique names.")
        self.soft_prompt_domains = tuple(domains or ())
        if any(not isinstance(d, str) or not d.strip() for d in self.soft_prompt_domains):
            raise ValueError("soft_prompt_domains must contain nonempty names.")
        if len(set(self.soft_prompt_domains)) != len(self.soft_prompt_domains):
            raise ValueError("soft_prompt_domains must be unique.")
        if self.num_soft_prompts and not self.soft_prompt_domains:
            raise ValueError("soft_prompt_domains is required when prompts are enabled.")
        if not math.isfinite(init_std) or init_std <= 0:
            raise ValueError("soft_prompt_init_std must be finite and positive.")
        self.soft_prompt_init_std = float(init_std)
        # K=0 does not register a parameter or consume RNG.
        if self.num_soft_prompts:
            self.soft_prompt_hub = nn.Embedding(
                len(self.soft_prompt_domains), self.num_soft_prompts * self.hidden_dim,
            )
            nn.init.normal_(self.soft_prompt_hub.weight, std=init_std)

    def soft_prompt_behavior(self):
        return {
            "num_soft_prompts": self.num_soft_prompts,
            "soft_prompt_domains": list(self.soft_prompt_domains) if self.uses_domain_conditioning else [],
            "domain_configs": self.domain_configs,
            "soft_prompt_layout": "per_chunk_prefix",
            "soft_prompt_position_mode": "identity_rope",
            "soft_prompt_timestep_mode": "follow_chunk",
        }

    def normalize_domain_id(self, domain_id, *, batch_size, device, allow_default=False):
        if not self.uses_domain_conditioning:
            return None
        if domain_id is None:
            if allow_default and len(self.soft_prompt_domains) == 1:
                domain_id = 0
            else:
                raise ValueError("domain_id is required for action soft prompts.")
        ids = torch.as_tensor(domain_id, device=device)
        if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("domain_id must contain integers, not floats or booleans.")
        if ids.ndim == 0:
            ids = ids.expand(batch_size)
        if ids.shape != (batch_size,):
            raise ValueError(f"domain_id must be scalar or [B={batch_size}], got {tuple(ids.shape)}.")
        if bool(((ids < 0) | (ids >= len(self.soft_prompt_domains))).any()):
            raise ValueError("domain_id is outside soft_prompt_domains.")
        return ids.to(dtype=torch.long)

    def _pack_prompt_rows(self, x, meta, *, identity=False):
        """Pack [B,H,...] using each chunk's first row, or identity RoPE."""
        b, h = x.shape[:2]
        n, c, k = meta["num_chunks"], meta["chunk_size"], self.num_soft_prompts
        if h != n * c:
            raise ValueError("Action soft prompt row count does not match physical action length.")
        chunks = x.reshape(b, n, c, *x.shape[2:])
        shape = (b, n, k, *x.shape[2:])
        prefix = x.new_ones(shape) if identity else chunks[:, :, :1].expand(shape)
        return torch.cat([prefix, chunks], dim=2).reshape(b, n * (c + k), *x.shape[2:])

    def pack_action_freqs(self, action_freqs, meta):
        if not self.num_soft_prompts:
            return action_freqs
        if action_freqs.ndim == 3:
            return self._pack_prompt_rows(action_freqs[None], meta, identity=True)[0]
        if action_freqs.ndim == 4:
            return self._pack_prompt_rows(action_freqs, meta, identity=True)
        raise ValueError("Action RoPE must have shape [H,1,R] or [B,H,1,R].")

    @staticmethod
    def build_chunk_local_mask(seq_len, tokens_per_chunk, device):
        if seq_len % tokens_per_chunk:
            raise ValueError("Packed action length must be divisible by tokens_per_chunk.")
        chunk_ids = torch.arange(seq_len, device=device) // tokens_per_chunk
        return chunk_ids[:, None] == chunk_ids[None, :]

    def _pack_soft_prompt_state(self, state, domain_id, timestep):
        if not self.num_soft_prompts:
            return state
        meta = state["meta"]
        if meta["clean_seq_len"]:
            raise ValueError("Action soft prompts do not support clean teacher-forcing tokens.")
        b, h, c = meta["batch_size"], meta["seq_len"], meta["chunk_size"]
        if h % c:
            raise ValueError("Action length must be divisible by chunk_size for soft prompts.")
        n, k = h // c, self.num_soft_prompts
        meta.update(num_soft_prompts=k, num_chunks=n, tokens_per_chunk=c + k,
                    packed_seq_len=n * (c + k), total_seq_len=n * (c + k))
        if timestep.ndim == 2:
            times = timestep.reshape(b, n, c)
            if not torch.equal(times, times[:, :, :1].expand_as(times)):
                raise ValueError("Soft prompts require one shared timestep per action chunk.")
        mask = state["context_mask"].reshape(b, n, c, -1)
        if not torch.equal(mask, mask[:, :, :1].expand_as(mask)):
            raise ValueError("Soft prompts require shared context visibility within each chunk.")
        tokens = state["tokens"]
        ids = self.normalize_domain_id(domain_id, batch_size=b, device=tokens.device)
        prompt = self.soft_prompt_hub(ids).reshape(b, 1, k, self.hidden_dim).to(tokens.dtype)
        state["tokens"] = torch.cat([
            prompt.expand(b, n, k, self.hidden_dim), tokens.reshape(b, n, c, self.hidden_dim),
        ], dim=2).reshape(b, n * (c + k), self.hidden_dim)
        for name, seq_ndim in (("t", 3), ("t_mod", 4)):
            if state[name].ndim == seq_ndim:
                state[name] = self._pack_prompt_rows(state[name], meta)
        state["context_mask"] = self._pack_prompt_rows(state["context_mask"], meta)
        state["freqs"] = self.pack_action_freqs(state["freqs"], meta)
        state["self_attn_mask"] = self.build_chunk_local_mask(n * (c + k), c + k, tokens.device)
        return state

    def unpack_action_tokens(self, tokens, meta):
        if not self.num_soft_prompts:
            return tokens
        b, n, c, k = meta["batch_size"], meta["num_chunks"], meta["chunk_size"], self.num_soft_prompts
        if tokens.shape[:2] != (b, meta["packed_seq_len"]):
            raise ValueError("Packed action output does not match soft prompt metadata.")
        return tokens.reshape(b, n, c + k, -1)[:, :, k:].reshape(b, n * c, -1)
