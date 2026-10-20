"""Sample-local, temporally consistent photometric augmentation for RGB clips.

Follows OpenWAM's fixed brightness -> contrast -> saturation order, operating
on float tensors instead of quantizing resized frames back to PIL uint8. Hue
is an optional fourth operation (unlike OpenWAM's unused hue field).
"""
from collections.abc import Mapping
import math
from numbers import Real

import numpy as np
import torch
from torchvision.transforms import functional as TF


class VideoColorJitter:
    """Transform [T, 3, H, W] float32 RGB in [-1, 1], sharing factors over T.

    ``exclusion_mask`` optionally marks structural missing-camera pixels, with
    shape [H, W] or [T, H, W]. Those pixels are excluded from contrast means
    and preserved exactly. Do not infer this mask from RGB values.
    """

    _DEFAULTS = dict(enabled=False, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
    _RNG_STREAM = 0x434A4954  # CJIT; independent of offset / episode / text RNG.

    def __init__(self, config=None, *, training=True):
        if config is None or config is False:
            config = {}
        elif config is True:
            config = {"enabled": True}
        if not isinstance(config, Mapping):
            raise ValueError("color_jitter must be a mapping, boolean or null.")
        unknown = set(config) - self._DEFAULTS.keys()
        if unknown:
            raise ValueError(f"Unknown color_jitter settings: {sorted(unknown)}")
        settings = dict(self._DEFAULTS, **config)
        if not isinstance(settings["enabled"], bool):
            raise ValueError("color_jitter.enabled must be a boolean.")
        for name in ("brightness", "contrast", "saturation", "hue"):
            value = settings[name]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"color_jitter.{name} must be a finite nonnegative number.")
            if name == "hue" and value > 0.5:
                raise ValueError("color_jitter.hue must be in [0, 0.5].")
            settings[name] = float(value)
        self.settings = settings
        self.active = bool(training and settings["enabled"] and any(
            settings[name] for name in ("brightness", "contrast", "saturation", "hue")))

    def recipe(self):
        """Only active augmentation changes the historical dataset recipe."""
        if not self.active:
            return {}
        return {"color_jitter": dict(
            self.settings, version=1, implementation="torchvision_float32_rgb_v1",
            order=["brightness", "contrast", "saturation", "hue"],
            sharing="sample_all_frames_and_camera_slots",
            rng="numpy_seed_epoch_index_cjit_v1", exclusion="explicit_missing_camera_mask",
        )}

    def __call__(self, frames, *, seed, epoch, index, exclusion_mask=None):
        # Preserve pixels AND RNG state exactly for old configs and validation.
        if not self.active:
            return frames
        if frames.ndim != 4 or frames.shape[1] != 3 or frames.dtype != torch.float32:
            raise ValueError("ColorJitter expects [T, 3, H, W] float32 RGB in [-1, 1].")
        if not torch.isfinite(frames).all() or frames.min() < -1.000001 or frames.max() > 1.000001:
            raise ValueError("ColorJitter RGB values must be finite and in [-1, 1].")
        mask = None
        if exclusion_mask is not None:
            if exclusion_mask.dtype != torch.bool:
                raise ValueError("ColorJitter exclusion_mask must be boolean.")
            if exclusion_mask.shape == frames.shape[-2:]:
                mask = exclusion_mask[None, None]
            elif exclusion_mask.shape == (len(frames), *frames.shape[-2:]):
                mask = exclusion_mask[:, None]
            else:
                raise ValueError("ColorJitter exclusion_mask must be [H, W] or [T, H, W].")
            mask = mask.to(frames.device)

        rng = np.random.default_rng(np.random.SeedSequence([
            int(seed), int(epoch), int(index), self._RNG_STREAM,
        ]))
        factors = [float(rng.uniform(max(0., 1. - self.settings[name]), 1. + self.settings[name]))
                   for name in ("brightness", "contrast", "saturation")]
        hue = float(rng.uniform(-self.settings["hue"], self.settings["hue"]))
        # Bilinear canvas resizing can overshoot by a few float32 ulps. Hue
        # requires nonnegative RGB; clamp only this active augmentation path.
        rgb = ((frames + 1.) * 0.5).clamp(0, 1)
        if self.settings["brightness"]:
            rgb = TF.adjust_brightness(rgb, factors[0])
        if self.settings["contrast"]:
            if mask is None:
                rgb = TF.adjust_contrast(rgb, factors[1])
            else:
                valid = ~mask
                gray = TF.rgb_to_grayscale(rgb)
                count = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
                mean = (gray * valid).sum(dim=(-2, -1), keepdim=True) / count
                rgb = (factors[1] * rgb + (1. - factors[1]) * mean).clamp(0, 1)
        if self.settings["saturation"]:
            rgb = TF.adjust_saturation(rgb, factors[2])
        if self.settings["hue"]:
            rgb = TF.adjust_hue(rgb, hue)
        result = rgb.clamp(0, 1) * 2. - 1.
        return result if mask is None else torch.where(mask, frames, result)
