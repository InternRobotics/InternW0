"""Frozen DINOv3 observation encoders, with HF ViT-B/16 as the default.

The legacy ViT-S/16 implementation below was ported from wam_v2/ahav2;
that reference credits ST-WAM (MIT), commit df335b0d.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _get_patches_center_coordinates(
    num_patches_h: int,
    num_patches_w: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device) / num_patches_h
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device) / num_patches_w
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    return coords.flatten(0, 1).mul_(2.0).sub_(1.0)


def _build_rope_embeddings(
    height: int,
    width: int,
    patch_size: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    rope_theta: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build DINOv3 2D RoPE embeddings for patch tokens."""
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", enabled=False):
        coords = _get_patches_center_coordinates(
            num_patches_h,
            num_patches_w,
            dtype=torch.float32,
            device=device,
        )
        inv_freq = 1.0 / rope_theta ** torch.arange(
            0,
            1,
            4 / head_dim,
            dtype=torch.float32,
            device=device,
        )
        angles = 2 * torch.pi * coords[:, :, None] * inv_freq[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        cos = torch.cos(angles).to(dtype=dtype)
        sin = torch.sin(angles).to(dtype=dtype)
    return cos, sin


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to patch tokens while leaving CLS/register prefix tokens untouched."""
    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix_tokens = num_tokens - num_patches

    q_prefix, q_patches = q.split((num_prefix_tokens, num_patches), dim=-2)
    k_prefix, k_patches = k.split((num_prefix_tokens, num_patches), dim=-2)

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_patches = (q_patches * cos) + (_rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (_rotate_half(k_patches) * sin)
    return torch.cat((q_prefix, q_patches), dim=-2), torch.cat((k_prefix, k_patches), dim=-2)


# ---------------------------------------------------------------------------
# Lightweight ViT backbone - structure mirrors the DINOv3 safetensors exactly
# ---------------------------------------------------------------------------
# Key naming in this module matches the checkpoint 1-to-1 so that no key
# remapping is needed at load time.  Verified against the header of
# dinov3_vitl16_pretrain_lvd1689m.safetensors (415 keys, 24 layers).


class _Attention(nn.Module):
    """Multi-head attention with separate Q/K/V projections (DINOv3 style).

    K projection has NO bias; Q, V, and O projections have bias.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        query = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        query, key = _apply_rotary_pos_emb(query, key, *position_embeddings)
        attn_out = F.scaled_dot_product_attention(query, key, value)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_out)


class _MLP(nn.Module):
    """Standard GELU MLP with up_proj / down_proj naming (matching DINOv3 checkpoint)."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.up_proj = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class _Block(nn.Module):
    """Single transformer block matching DINOv3 checkpoint layout."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 layer_scale_init: Optional[float] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _MLP(dim, mlp_ratio)

        if layer_scale_init is not None:
            self.layer_scale1 = nn.ParameterDict({
                "lambda1": nn.Parameter(torch.full((dim,), layer_scale_init))
            })
            self.layer_scale2 = nn.ParameterDict({
                "lambda1": nn.Parameter(torch.full((dim,), layer_scale_init))
            })
        else:
            self.layer_scale1 = None
            self.layer_scale2 = None

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        attn_out = self.attention(self.norm1(x), position_embeddings)
        if self.layer_scale1 is not None:
            x = x + attn_out * self.layer_scale1["lambda1"]
        else:
            x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        if self.layer_scale2 is not None:
            x = x + mlp_out * self.layer_scale2["lambda1"]
        else:
            x = x + mlp_out
        return x


class _Embeddings(nn.Module):
    """Patch embedding + special tokens, matching DINOv3 checkpoint naming."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int,
                 num_register_tokens: int = 4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embeddings = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))


class _DinoViT(nn.Module):
    """Minimal ViT whose state-dict keys mirror the DINOv3 safetensors exactly."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        layer_scale_init: Optional[float] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.num_register_tokens = num_register_tokens
        self.rope_theta = 100.0

        self.embeddings = _Embeddings(in_channels, embed_dim, patch_size, num_register_tokens)

        self.layer = nn.ModuleList([
            _Block(embed_dim, num_heads, mlp_ratio, layer_scale_init=layer_scale_init)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward_features(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]
        _, _, height, width = x.shape

        patch_tokens = self.embeddings.patch_embeddings(x).flatten(2).transpose(1, 2)  # [B, N, D]

        cls_tokens = self.embeddings.cls_token.expand(batch_size, -1, -1)
        if self.num_register_tokens > 0:
            reg_tokens = self.embeddings.register_tokens.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, reg_tokens, patch_tokens], dim=1)
        else:
            x = torch.cat([cls_tokens, patch_tokens], dim=1)

        position_embeddings = _build_rope_embeddings(
            height=height,
            width=width,
            patch_size=self.patch_size,
            head_dim=self.embed_dim // self.layer[0].attention.num_heads,
            dtype=x.dtype,
            device=x.device,
            rope_theta=self.rope_theta,
        )

        for block in self.layer:
            x = block(x, position_embeddings)
        x = self.norm(x)

        cls_out = x[:, 0]
        patch_tokens = x[:, 1 + self.num_register_tokens:]

        return {
            "x_norm_clstoken": cls_out,
            "x_norm_patchtokens": patch_tokens,
        }


def _load_hf_b16(path: Path) -> tuple[nn.Module, str]:
    """Load a complete local HF B/16 asset and fingerprint config plus weights."""
    if not path.is_dir():
        raise FileNotFoundError(
            f"DINOv3 B/16 requires a local HF directory containing config.json "
            f"and safetensors weights: {path}. Set model.dino_config.weights_path."
        )
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    config_path = path / "config.json"
    raw_config = json.loads(config_path.read_text())
    if raw_config.get("model_type") != "dinov3_vit":
        raise ValueError("Expected a Hugging Face DINOv3 ViT config.json.")
    config = DINOv3ViTConfig.from_dict(raw_config)
    expected = {"hidden_size": 768, "num_hidden_layers": 12,
                "num_attention_heads": 12, "patch_size": 16, "num_channels": 3}
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(f"DINOv3 B/16 requires {name}={value}; got {getattr(config, name)}.")

    files = [config_path]
    if (path / "model.safetensors").is_file():
        files.append(path / "model.safetensors")
    else:
        index_path = path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index["weight_map"].values()))
        if not shard_names or any(Path(name).name != name for name in shard_names):
            raise ValueError("DINO safetensors index must reference local shard filenames.")
        files.extend([index_path, *(path / name for name in shard_names)])
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode() + b"\0")
        digest.update(str(file.stat().st_size).encode() + b"\0")
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)

    # HF normally tolerates missing keys by initializing them. An external
    # frozen encoder must never silently acquire random parameters.
    with torch.random.fork_rng(devices=[]):
        backbone, info = DINOv3ViTModel.from_pretrained(
            str(path), config=config, local_files_only=True, use_safetensors=True,
            dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True,
        )
    errors = {key: info[key] for key in
              ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
    if errors:
        raise ValueError(f"Incomplete or incompatible DINOv3 B/16 checkpoint: {errors}")
    return backbone, digest.hexdigest()


class DinoVideoEncoder(nn.Module):
    """Frozen HF DINOv3 ViT-B/16, matching the pretraining feature path.

    The backbone is loaded before optimizer/distributed setup. Native AHA
    weights reference this external frozen encoder by its SHA-256 identity.
    ViT-S/16 remains available by explicit model_name and feature_dim for old
    checkpoints; its preprocessing and identity contract remain unchanged.
    """

    def __init__(self, *, weights_path, model_name="dinov3-vitb16-hf",
                 input_resolution=(384, 320), feature_dim=768, patch_size=16,
                 use_cls_token=False, normalize_features=False,
                 encode_microbatch_size=16):
        super().__init__()
        dimensions = {"dinov3-vitb16-hf": 768, "dinov3-vits16-timm": 384}
        if model_name not in dimensions or feature_dim != dimensions[model_name] or patch_size != 16:
            raise ValueError("DINO requires dinov3-vitb16-hf (768-D) or dinov3-vits16-timm (384-D), patch16.")
        if model_name == "dinov3-vitb16-hf" and (use_cls_token or normalize_features):
            raise ValueError("Pretraining B/16 features exclude CLS/register tokens and additional L2 normalization.")
        self.model_name = model_name
        self.input_resolution = tuple(int(x) for x in input_resolution)
        if len(self.input_resolution) != 2 or any(x <= 0 or x % patch_size for x in self.input_resolution):
            raise ValueError("DINO input_resolution must contain two positive multiples of patch_size.")
        self.feature_dim, self.patch_size = int(feature_dim), int(patch_size)
        self.use_cls_token = bool(use_cls_token)
        self.normalize_features = bool(normalize_features)
        self.encode_microbatch_size = int(encode_microbatch_size)
        self.num_patches = (self.input_resolution[0] // patch_size) * (self.input_resolution[1] // patch_size)
        path = Path(weights_path)
        if self.model_name == "dinov3-vitb16-hf":
            self.backbone, self.weights_sha256 = _load_hf_b16(path)
        else:
            if not path.is_file():
                raise FileNotFoundError(f"DINO checkpoint not found: {path}. Set model.dino_config.weights_path.")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self.weights_sha256 = digest.hexdigest()
            from safetensors.torch import load_file

            self.backbone = _DinoViT(layer_scale_init=1e-5)
            self.backbone.load_state_dict(load_file(str(path), device="cpu"), strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def behavior_config(self):
        contract = {
            "model_name": self.model_name, "weights_sha256": self.weights_sha256,
            "input_resolution": list(self.input_resolution),
            "feature_dim": self.feature_dim, "patch_size": self.patch_size,
            "use_cls_token": self.use_cls_token, "normalize_features": self.normalize_features,
            "preprocessing": "minus_one_one_to_imagenet_bilinear",
        }
        if self.model_name == "dinov3-vitb16-hf":
            contract.update(
                preprocessing="uint8_or_minus_one_one_to_imagenet_float32_no_resize",
                weights_hash_format="hf_config_and_safetensors_v1",
            )
        return contract

    @torch.no_grad()
    def encode_frames(self, frames):
        if frames.ndim != 4 or frames.shape[1] != 3 or frames.shape[0] == 0:
            raise ValueError("DINO frames must be nonempty [B,3,H,W] RGB tensors.")
        if self.model_name == "dinov3-vitb16-hf":
            return self._encode_b16_frames(frames)
        parameter = next(self.backbone.parameters())
        limit = self.encode_microbatch_size if self.encode_microbatch_size > 0 else frames.shape[0]
        outputs = []
        for images in frames.split(limit):
            # Keep ahav2's resize -> dtype/device -> ImageNet normalization order.
            if tuple(images.shape[-2:]) != self.input_resolution:
                images = F.interpolate(images, size=self.input_resolution, mode="bilinear", align_corners=False)
            images = images.to(device=parameter.device, dtype=parameter.dtype)
            mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            features = self.backbone.forward_features(((images + 1) * 0.5 - mean) / std)
            tokens = features["x_norm_patchtokens"]
            if self.use_cls_token:
                tokens = torch.cat([features["x_norm_clstoken"].unsqueeze(1), tokens], dim=1)
            if self.normalize_features:
                tokens = F.normalize(tokens, dim=-1)
            outputs.append(tokens)
        return torch.cat(outputs)

    def _encode_b16_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Preserve pretraining's canvas, FP32 normalization and patch ordering."""
        if tuple(frames.shape[-2:]) != self.input_resolution:
            raise ValueError(f"DINO B/16 expects the prepared canvas {self.input_resolution}; got {tuple(frames.shape[-2:])}.")
        if frames.dtype != torch.uint8 and not frames.is_floating_point():
            raise ValueError("DINO B/16 expects uint8 RGB or floating-point RGB in [-1,1].")
        parameter = next(self.backbone.parameters())
        limit = self.encode_microbatch_size if self.encode_microbatch_size > 0 else frames.shape[0]
        outputs = []
        for images in frames.split(limit):
            pixels = images.to(device=parameter.device, dtype=torch.float32)
            pixels = pixels / 255.0 if frames.dtype == torch.uint8 else (pixels + 1.0) / 2.0
            mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            # HF casts normalized pixels to the patch embedding's dtype.
            features = self.backbone(pixel_values=(pixels - mean) / std).last_hidden_state
            tokens = features[:, 1 + self.backbone.config.num_register_tokens:]
            if tokens.shape[1:] != (self.num_patches, self.feature_dim):
                raise ValueError(f"Unexpected DINO B/16 patch shape: {tuple(tokens.shape)}")
            outputs.append(tokens)
        return torch.cat(outputs)

