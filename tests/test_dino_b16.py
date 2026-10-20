"""Check the real HF B/16 path with local synthetic weights, without downloads."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from w0.models.wan22.dino_encoder import DinoVideoEncoder


@pytest.fixture(scope="module")
def b16_assets(tmp_path_factory):
    torch.set_num_threads(1)
    if os.environ.get("W0_DINO_WEIGHTS"):
        return Path(os.environ["W0_DINO_WEIGHTS"])
    path = tmp_path_factory.mktemp("dino_b16")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        backbone = DINOv3ViTModel(DINOv3ViTConfig(
            hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
            num_attention_heads=12, patch_size=16, num_register_tokens=4,
        ))
    backbone.config.save_pretrained(path)
    save_file(backbone.state_dict(), str(path / "model.safetensors"), metadata={"format": "pt"})
    return path


@pytest.fixture(scope="module")
def encoder(b16_assets):
    return DinoVideoEncoder(
        weights_path=b16_assets, input_resolution=(32, 48), encode_microbatch_size=1,
    )


def test_b16_matches_pretraining_hf_features_and_microbatch_order(encoder):
    images = torch.randint(0, 256, (3, 3, 32, 48), dtype=torch.uint8)
    pixels = images.float() / 255.0
    pixels = (pixels - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / (
        torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    )
    with torch.no_grad():
        expected = encoder.backbone(pixel_values=pixels).last_hidden_state[:, 5:]
    actual = encoder.encode_frames(images)
    assert actual.shape == (3, 6, 768)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        encoder.encode_frames(images.float() / 255.0 * 2.0 - 1.0),
        actual, rtol=2e-5, atol=2e-5,
    )
    encoder.train(True)
    assert not encoder.training and not encoder.backbone.training
    assert not actual.requires_grad
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    contract = encoder.behavior_config()
    assert contract["feature_dim"] == 768
    assert contract["weights_hash_format"] == "hf_config_and_safetensors_v1"
    assert len(contract["weights_sha256"]) == 64


def test_b16_preserves_float32_normalization_with_bfloat16_backbone(encoder):
    images = torch.randint(0, 256, (1, 3, 32, 48), dtype=torch.uint8)
    pixels = images.float() / 255.0
    pixels = (pixels - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / (
        torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    )
    encoder.to(dtype=torch.bfloat16)
    try:
        with torch.no_grad():
            expected = encoder.backbone(pixel_values=pixels).last_hidden_state[:, 5:]
        actual = encoder.encode_frames(images)
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        encoder.to(dtype=torch.float32)


def test_b16_rejects_unprepared_canvas_and_wrong_input_dtype(encoder):
    with pytest.raises(ValueError, match="prepared canvas"):
        encoder.encode_frames(torch.zeros(1, 3, 33, 48))
    with pytest.raises(ValueError, match="uint8"):
        encoder.encode_frames(torch.zeros(1, 3, 32, 48, dtype=torch.int32))


def test_b16_rejects_wrong_model_configuration(tmp_path):
    config = DINOv3ViTConfig(hidden_size=384)
    config.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="hidden_size=768"):
        DinoVideoEncoder(weights_path=tmp_path)
    with pytest.raises(ValueError, match="768-D"):
        DinoVideoEncoder(weights_path=tmp_path, feature_dim=384)


def test_b16_rejects_missing_weights_instead_of_freezing_random_parameters(b16_assets, tmp_path):
    (tmp_path / "config.json").write_text((b16_assets / "config.json").read_text())
    weights = load_file(str(b16_assets / "model.safetensors"))
    del weights["norm.bias"]
    save_file(weights, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="missing_keys.*norm.bias"):
        DinoVideoEncoder(weights_path=tmp_path)


def test_b16_fingerprint_includes_config(b16_assets, encoder, tmp_path):
    config = json.loads((b16_assets / "config.json").read_text())
    config["rope_theta"] = 200.0
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").symlink_to(b16_assets / "model.safetensors")
    changed = DinoVideoEncoder(weights_path=tmp_path, input_resolution=(32, 48))
    assert changed.weights_sha256 != encoder.weights_sha256
