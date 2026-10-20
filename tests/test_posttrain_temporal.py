"""Offset configuration, exact sampling alignment and initialization contracts."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from w0.datasets.motus_posttrain import MotusPosttrainDataset
from w0.datasets.robotwin_posttrain import RobotwinPosttrainDataset
from test_motus_posttrain import make_clean_dataset
from test_robotwin_posttrain import tiny_model


@pytest.mark.parametrize('offset,horizon,chunk,video_horizon,stride', [
    (15, 64, 16, 96, 8), (0, 64, 16, 96, 8), (0, 64, 16, 64, 8),
    (7, 32, 8, 64, 4), (32, 64, 16, 96, 8),
])
def test_sampling_uses_configured_timelines(tmp_path, monkeypatch, offset, horizon, chunk, video_horizon, stride):
    import w0.datasets.motus_posttrain as module
    dataset, processor, qpos = make_clean_dataset(
        tmp_path, length=401, max_action_offset=offset, action_horizon=horizon,
        action_chunk_size=chunk, video_plan_horizon=video_horizon, action_video_freq_ratio=stride,
    )
    # Encode each source frame index as a pixel, exposing exact chunk/video selection.
    monkeypatch.setattr(module, 'decode_native_frames', lambda path, positions, fps:
                        torch.tensor(positions).view(-1, 1, 1, 1).expand(-1, 3, 360, 320))
    monkeypatch.setattr(module, 'camera_canvas', lambda images: images[0][..., :1, :1])
    anchor = 3
    for epoch in range(4):
        sample = dataset[(anchor, epoch)]
        selected_offset = int(sample['action_offset'])
        expected_rng = np.random.default_rng(np.random.SeedSequence([42, epoch, anchor]))
        assert selected_offset == (int(expected_rng.integers(offset + 1)) if offset else 0)
        assert sample['action'].shape == sample['proprio'].shape == (horizon + offset, 37)
        assert not sample['action_is_pad'].any()
        torch.testing.assert_close(sample['video'][0, :, 0, 0],
                                   torch.arange(0, video_horizon + 1, stride) * 2 + anchor)
        starts = torch.arange(0, horizon, chunk) + selected_offset
        torch.testing.assert_close(sample['chunk_obs_images'][:, 0, 0, 0], anchor + 2 * starts)
        torch.testing.assert_close(processor.unpack_action(sample['action'])[selected_offset],
                                   qpos[anchor + 2 * (selected_offset + 1)])
        expected, _ = processor.pack(qpos[anchor + 2 * starts], 'proprio')
        torch.testing.assert_close(sample['proprio'][starts], expected)
    assert dataset.domain_data_recipe['max_action_offset'] == offset
    assert dataset.domain_data_recipe['video_plan_horizon'] == video_horizon
    # Temporal ablations reuse the same real-transition normalization statistics.
    assert processor.statistics['recipe']['max_action_offset'] == 15


@pytest.mark.parametrize('dataset_type', [MotusPosttrainDataset, RobotwinPosttrainDataset])
@pytest.mark.parametrize('kwargs,message', [
    ({'max_action_offset': -1}, 'max_action_offset'),
    ({'max_action_offset': 1.5}, 'max_action_offset'),
    ({'max_action_offset': True}, 'max_action_offset'),
    ({'action_horizon': 0}, 'action_horizon'),
    ({'action_chunk_size': 0}, 'action_chunk_size'),
    ({'action_horizon': 63}, 'divisible by action_chunk_size'),
    ({'video_plan_horizon': 64}, 'cover action_horizon'),
    ({'max_action_offset': 0, 'video_plan_horizon': 80}, 'divisible by 4'),
    ({'action_video_freq_ratio': 0}, 'action_video_freq_ratio'),
])
def test_invalid_layout_fails_before_dataset_io(dataset_type, kwargs, message):
    with pytest.raises(ValueError, match=message):
        dataset_type('/nonexistent', None, **kwargs)


@pytest.mark.parametrize('config_name', ['posttrain_robotwin', 'posttrain_libero'])
def test_hydra_model_overrides_reach_both_datasets(config_name):
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'configs'), version_base=None):
        cfg = compose(config_name=config_name, overrides=['model.max_action_offset=0',
                      'model.video_plan_horizon=64', 'model.action_chunk_size=8'])
        OmegaConf.resolve(cfg)
    for dataset in (cfg.data.train, cfg.data.val):
        assert dataset.max_action_offset == 0
        assert dataset.video_plan_horizon == dataset.action_horizon == 64
        assert dataset.action_chunk_size == 8
        assert dataset.action_video_freq_ratio == cfg.model.video_action_stride


def test_disabled_offset_rejects_nonzero_samples_and_resume_objective_changes(tmp_path):
    from types import SimpleNamespace
    model = tiny_model()
    dataset = SimpleNamespace(max_action_offset=15, action_horizon=64, action_chunk_size=16,
                              action_video_freq_ratio=8, video_plan_horizon=96,
                              _action_offset_enabled=True, domain_configs=model.action_expert.domain_configs,
                              domain_ids=[0])
    model.bind_training_data_recipe(dataset, dataset)
    saved = copy.deepcopy({'training_recipe': model.training_recipe()})
    model.max_action_offset = dataset.max_action_offset = 0
    dataset._action_offset_enabled = False
    model.bind_training_data_recipe(dataset, dataset)
    assert model._normalize_action_offsets({'action_offset': 0}, batch_size=1).item() == 0
    with pytest.raises(ValueError, match='exceeds configured max_action_offset'):
        model._normalize_action_offsets({'action_offset': 1}, batch_size=1)
    with pytest.raises(ValueError, match='recipe mismatch'):
        model.validate_training_recipe(saved)
    for key, value in [('action_horizon', 32), ('action_chunk_size', 8),
                       ('video_plan_horizon', 64), ('action_video_freq_ratio', 4)]:
        changed = copy.copy(dataset)
        setattr(changed, key, value)
        with pytest.raises(ValueError, match=key):
            model.bind_training_data_recipe(dataset, changed)
    # Weight-only initialization is compatible when only max offset changes.
    checkpoint = tmp_path / 'model.pt'
    tiny_model().save_checkpoint(checkpoint)
    model.load_checkpoint(checkpoint)
