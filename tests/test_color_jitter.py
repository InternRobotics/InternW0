"""Photometric invariants, reader alignment and checkpoint/config contracts."""
import copy
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance
import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from w0.datasets.color_jitter import VideoColorJitter
from w0.datasets.motus_posttrain import MotusPosttrainDataset
from test_motus_posttrain import make_clean_dataset
from test_robotwin_posttrain import tiny_model


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def color_clip(size=16):
    # Nonuniform colors expose independent-frame draws and channel/order errors.
    rgb = torch.linspace(-0.9, 0.8, 3 * size * size).reshape(1, 3, size, size)
    return rgb.repeat(3, 1, 1, 1)


@pytest.mark.parametrize('config,training', [
    (None, True), (False, True), ({}, True), ({'enabled': False}, True),
    ({'enabled': True, 'brightness': 0, 'contrast': 0, 'saturation': 0, 'hue': 0}, True),
    ({'enabled': True, 'hue': .1}, False),
])
def test_disabled_and_validation_are_exact_noops(config, training):
    jitter = VideoColorJitter(config, training=training)
    frames = color_clip()
    # Disabled paths must not even allocate/sample a new generator.
    from unittest.mock import patch
    with patch('numpy.random.default_rng', side_effect=AssertionError('unexpected RNG')):
        assert jitter(frames, seed=42, epoch=0, index=0) is frames
    assert jitter.recipe() == {}


def test_temporal_consistency_reproducibility_and_rng_isolation():
    jitter, frames = VideoColorJitter({'enabled': True, 'hue': .1}), color_clip()
    original = frames.clone()
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    a = jitter(frames, seed=42, epoch=3, index=7)
    random.random()
    np.random.random()
    torch.rand(1)
    b = jitter(frames, seed=42, epoch=3, index=7)
    assert torch.equal(a, b) and torch.equal(a[0], a[1])
    assert torch.equal(frames, original) and not torch.equal(a, frames)
    assert torch.isfinite(a).all() and a.min() >= -1 and a.max() <= 1
    assert a.shape == frames.shape and a.dtype == torch.float32
    for change in ({'epoch': 4}, {'index': 8}, {'seed': 43}):
        assert not torch.equal(a, jitter(frames, **dict(dict(seed=42, epoch=3, index=7), **change)))
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.random.set_rng_state(state[2])
    jitter(frames, seed=42, epoch=3, index=7)
    assert random.getstate() == state[0]
    assert np.array_equal(np.random.get_state()[1], state[1][1])
    assert torch.equal(torch.random.get_rng_state(), state[2])


def test_default_operations_match_openwam_pil_reference_with_float_tolerance(monkeypatch):
    # An independent PIL reference pins OpenWAM's fixed operation order. The
    # float path intentionally avoids PIL's repeated uint8 rounding.
    class Draws:
        def __init__(self):
            self.draws = iter([1.13, .87, 1.16, 0.])
        def uniform(self, *args):
            return next(self.draws)
    monkeypatch.setattr(np.random, 'default_rng', lambda *args: Draws())
    rgb = np.random.RandomState(9).randint(20, 210, (16, 16, 3)).astype(np.uint8)
    frames = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 127.5 - 1
    reference = ImageEnhance.Brightness(Image.fromarray(rgb)).enhance(1.13)
    reference = ImageEnhance.Contrast(reference).enhance(.87)
    reference = ImageEnhance.Color(reference).enhance(1.16)
    expected = torch.from_numpy(np.asarray(reference).copy()).permute(2, 0, 1).float()[None] / 127.5 - 1
    result = VideoColorJitter(True)(frames, seed=0, epoch=0, index=0)
    torch.testing.assert_close(result, expected, atol=4 / 127.5, rtol=0)


def test_hue_is_implemented_and_structural_padding_is_preserved():
    frames = color_clip()
    hue = VideoColorJitter(dict(enabled=True, brightness=0, contrast=0, saturation=0, hue=.2))
    assert not torch.equal(hue(frames, seed=42, epoch=0, index=0), frames)
    # A real black pixel stays inside the contrast statistic; only structural
    # padding is restored, and adding that padding cannot change valid pixels.
    frames[:, :, 0, 0] = -1
    padded = F.pad(frames, (0, 16), value=-1)
    mask = torch.zeros(16, 32, dtype=torch.bool)
    mask[:, 16:] = True
    jitter = VideoColorJitter(dict(enabled=True, brightness=0, saturation=0, contrast=.5))
    plain = jitter(frames, seed=42, epoch=0, index=0)
    result = jitter(padded, seed=42, epoch=0, index=0, exclusion_mask=mask)
    torch.testing.assert_close(result[..., :16], plain, atol=1e-6, rtol=0)
    assert torch.equal(result[..., 16:], padded[..., 16:])
    assert not torch.equal(result[:, :, 0, 0], frames[:, :, 0, 0])
    assert torch.equal(jitter(padded, seed=42, epoch=0, index=0,
                              exclusion_mask=torch.ones_like(mask)), padded)


def test_resize_roundoff_is_clamped_only_on_active_augmentation():
    frames = color_clip()
    frames[:, 0, 0, 0] = 1.000000238418579
    frames[:, 1, 0, 0] = -1.000000238418579
    jitter = VideoColorJitter(dict(enabled=True, brightness=0, contrast=0, saturation=0, hue=.1))
    result = jitter(frames, seed=42, epoch=0, index=0)
    assert result.min() >= -1 and result.max() <= 1 and torch.isfinite(result).all()
    assert VideoColorJitter(False)(frames, seed=42, epoch=0, index=0) is frames
    frames[:, 0, 0, 0] = 1.01
    with pytest.raises(ValueError, match='RGB values'):
        jitter(frames, seed=42, epoch=0, index=0)


@pytest.mark.parametrize('config', [
    'on', {'enable': True}, {'enabled': 'false'}, {'brightness': -0.1},
    {'contrast': float('nan')}, {'saturation': float('inf')}, {'hue': .51},
    {'hue': -.1}, {'brightness': True}, {'brightness': [0.8, 1.2]},
])
def test_invalid_config_is_rejected(config):
    with pytest.raises(ValueError, match='color_jitter'):
        VideoColorJitter(config)


@pytest.mark.parametrize('config_name', [
    'posttrain_robotwin',
])
def test_hydra_overrides_reach_train_but_never_augment_validation(config_name):
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'configs'), version_base=None):
        baseline = compose(config_name=config_name)
        cfg = compose(config_name=config_name, overrides=[
            'data.train.color_jitter.enabled=true', 'data.train.color_jitter.brightness=0.35',
            'data.train.color_jitter.contrast=0.1', 'data.train.color_jitter.saturation=0.4',
            'data.train.color_jitter.hue=0.05',
        ])
        OmegaConf.resolve(cfg)
    assert VideoColorJitter(baseline.data.train.color_jitter).active
    train = VideoColorJitter(cfg.data.train.color_jitter, training=cfg.data.train.is_training_set)
    val = VideoColorJitter(cfg.data.val.color_jitter, training=cfg.data.val.is_training_set)
    assert train.active and not val.active
    assert [train.settings[key] for key in ('brightness', 'contrast', 'saturation', 'hue')] == [.35, .1, .4, .05]


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('offset', [0, 15])
def test_motus_reader_preserves_labels_rng_and_aligns_all_visual_inputs(tmp_path, monkeypatch, training, offset):
    plain, processor, _ = make_clean_dataset(tmp_path, length=201, training=training, max_action_offset=offset)
    augmented = MotusPosttrainDataset(str(tmp_path), processor, is_training_set=training,
                                     max_action_offset=offset, color_jitter={'enabled': True})
    # Repeat an identifiable colored image at all times, including padded tails.
    raw = ((color_clip(8)[0] + 1) * 127.5).to(torch.uint8)
    monkeypatch.setattr('w0.datasets.motus_posttrain.decode_native_frames', lambda path, positions, fps:
                        F.interpolate(raw.float()[None], (360, 320), mode='nearest').to(torch.uint8).repeat(len(positions), 1, 1, 1))
    for index in (0, len(plain) - 1):
        a, b = plain[(index, 3)], augmented[(index, 3)]
        for key in a.keys() - {'video', 'chunk_obs_images'}:
            if torch.is_tensor(a[key]):
                assert torch.equal(a[key], b[key]), key
            else:
                assert a[key] == b[key], key
        assert torch.equal(b['video'][:, 0], b['chunk_obs_images'][0])
        assert torch.equal(b['video'][:, -1], b['chunk_obs_images'][-1])
        assert torch.equal(a['video'], b['video']) is not training
        assert torch.equal(b['video'], augmented[(index, 3)]['video'])
    assert processor.statistics == augmented.processor.statistics
    assert ('color_jitter' in augmented.domain_data_recipe) is training


def test_augmented_input_backward_and_checkpoint_recipe_compatibility(tmp_path, monkeypatch):
    plain, processor, _ = make_clean_dataset(tmp_path, length=81)
    augmented = MotusPosttrainDataset(str(tmp_path), processor, color_jitter=True)
    monkeypatch.setattr('w0.datasets.motus_posttrain.decode_native_frames', lambda path, positions, fps:
                        torch.full((len(positions), 3, 360, 320), 100, dtype=torch.uint8))
    model = tiny_model()
    model.action_expert.domain_configs = plain.domain_configs
    model.action_expert.soft_prompt_domains = [plain.domain_configs[0]['name']]
    model.bind_training_data_recipe(plain)
    old = copy.deepcopy({'training_recipe': model.training_recipe()})
    # Explicitly disabled augmentation retains old recipe and weight loading.
    disabled = MotusPosttrainDataset(str(tmp_path), processor, color_jitter={'enabled': False, 'brightness': .5})
    model.bind_training_data_recipe(disabled)
    model.validate_training_recipe(old)
    model.bind_training_data_recipe(augmented)
    with pytest.raises(ValueError, match='Training recipe mismatch'):
        model.validate_training_recipe(old)
    sample = augmented[(0, 2)]
    sample['video'] = F.interpolate(sample['video'].permute(1, 0, 2, 3), (16, 16)).permute(1, 0, 2, 3)
    sample['chunk_obs_images'] = F.interpolate(sample['chunk_obs_images'], (16, 16))
    sample['context'], sample['context_mask'] = torch.ones(3, 32), torch.ones(3, dtype=torch.bool)
    from w0.trainer import Wan22Trainer
    loss, _ = model.training_loss(Wan22Trainer._to_batched_eval_sample(sample))
    loss.backward()
    assert torch.isfinite(loss)
    grad = model.action_expert.head.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    path = tmp_path / 'augmented.pt'
    model.save_checkpoint(path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model.validate_training_recipe(payload)
    changed = MotusPosttrainDataset(str(tmp_path), processor, color_jitter={'enabled': True, 'brightness': .4})
    model.bind_training_data_recipe(changed)
    with pytest.raises(ValueError, match='Training recipe mismatch'):
        model.validate_training_recipe(payload)
    model.load_checkpoint(path)  # Weight-only initialization may change augmentation.
