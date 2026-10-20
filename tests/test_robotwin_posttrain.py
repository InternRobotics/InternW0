import copy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from w0.datasets.robotwin_posttrain import DOMAIN, RECIPE, SLOTS, RobotwinPosttrainProcessor
from test_future_video import make_model
from test_dino_observations import FakeDino


def processor(tmp_path):
    import json
    stats = {'recipe': RECIPE}
    for name in ('action', 'proprio'):
        stats[name] = dict(q01=[-2.] * 14, q99=[2.] * 14, minmax_bounds={'6': [-1., 1.]})
    path = tmp_path / 'stats.json'
    path.write_text(json.dumps(stats))
    return RobotwinPosttrainProcessor(str(path))


def test_native_slot_normalization_roundtrip_and_padding(tmp_path):
    p = processor(tmp_path)
    raw = torch.linspace(-1, 1, 14)[None].repeat(3, 1)
    raw[2] = float('nan')
    valid = torch.tensor([True, True, False])[:, None]
    packed, mask = p.pack(raw, 'action', valid)
    assert packed.shape == mask.shape == (3, 37)
    assert mask.sum().item() == 28
    assert torch.isfinite(packed).all()
    assert not packed[~mask].any()
    torch.testing.assert_close(p.unpack_action(packed)[:2], raw[:2])
    torch.testing.assert_close(packed[0, 16], raw[0, 6])
    assert not mask[:, 6].any() and not mask[:, 23].any()


def tiny_model():
    model = make_model(horizon=96, action_dim=37, proprio_dim=37,
                       action_kwargs={'domain_configs': [DOMAIN], 'num_soft_prompts': 8})
    model.max_action_offset = 15
    model.configure_future_video(video_plan_horizon=96, video_action_stride=8,
                                 future_training_mode='idm', video_plan_steps=1)
    with patch('w0.models.wan22.dino_encoder.DinoVideoEncoder', FakeDino):
        model.configure_chunk_obs_context(observation_encoder='dino', dino_config={'weights_path': 'test'},
                                          action_obs_downsample_factor=1, chunk_kv_editor_num_queries=3)
    return model


def short_sample():
    action_mask = torch.zeros(1, 79, 37, dtype=torch.bool)
    action_mask[:, :9, SLOTS] = True
    prop_mask = action_mask.clone()
    x = dict(video=torch.randn(1, 3, 13, 16, 16), chunk_obs_images=torch.randn(1, 4, 3, 16, 16),
             action=torch.randn(1, 79, 37).masked_fill(~action_mask, 0),
             proprio=torch.randn(1, 79, 37).masked_fill(~prop_mask, 0),
             action_valid_mask=action_mask, proprio_valid_mask=prop_mask,
             action_is_pad=~action_mask.any(-1), proprio_is_pad=~prop_mask.any(-1),
             image_is_pad=torch.tensor([[False, False] + [True] * 11]),
             context=torch.randn(1, 3, 32), context_mask=torch.ones(1, 3, dtype=torch.bool),
             action_video_freq_ratio=torch.tensor([8]), action_offset=torch.tensor([3]), domain_id=torch.tensor([0]))
    return x


def test_short_offset_training_masks_noise_proprio_and_gradients(tmp_path):
    torch.set_num_threads(1)
    model, sample = tiny_model(), short_sample()
    captured = []
    original = model.action_expert.pre_dit
    def capture(**kwargs):
        captured.append(kwargs['action_tokens'].detach().clone())
        return original(**kwargs)
    with patch.object(model.action_expert, 'pre_dit', side_effect=capture):
        loss, _ = model.training_loss(sample)
    assert torch.isfinite(loss)
    expected = sample['action_valid_mask'][:, 3:67]
    assert not captured[-1][~expected].any()
    inputs = model.build_inputs(sample)
    assert inputs['obs_context_mask'][0, :, -1].tolist() == [True, False, False, False]
    loss.backward()
    gradient = model.action_expert.head.weight.grad
    invalid = [i for i in range(37) if i not in SLOTS]
    assert gradient[:, SLOTS].abs().sum() > 0
    assert not gradient[:, invalid].any()
    assert not model.action_expert.action_encoder.weight.grad[..., invalid].any()
    assert not model.proprio_encoder.weight.grad[..., invalid].any()
    assert any(p.grad is not None for p in model.video_expert.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    opt.step()
    path = tmp_path / 'posttrain.pt'
    model._posttrain_statistics = processor(tmp_path).statistics
    model._posttrain_provenance = {'strategy': 'new_domain', 'seed': 42}
    model.save_checkpoint(path)
    restored = tiny_model()
    restored.load_checkpoint(path)
    assert restored.checkpoint_behavior()['schema_version'] == 4
    assert restored._posttrain_statistics == model._posttrain_statistics
    assert restored._posttrain_provenance == model._posttrain_provenance
    for key, value in model.mot.state_dict().items():
        torch.testing.assert_close(restored.mot.state_dict()[key], value, rtol=0, atol=0)
    restored._posttrain_statistics = {}
    with pytest.raises(ValueError, match='normalization/split metadata mismatch'):
        restored.load_checkpoint(path)


def test_full_state_metadata_preserves_statistics_and_initialization(tmp_path):
    from types import SimpleNamespace
    from w0.trainer import Wan22Trainer
    model = tiny_model()
    model._posttrain_statistics = processor(tmp_path).statistics
    model._posttrain_provenance = {'strategy': 'new_domain', 'seed': 42}
    trainer = SimpleNamespace(model=model, global_step=3, epoch=0, batch_in_epoch=3,
                              accelerator=SimpleNamespace(unwrap_model=lambda value: value))
    Wan22Trainer._save_trainer_state(trainer, str(tmp_path))
    restored = tiny_model()
    Wan22Trainer._read_trainer_state_metadata(trainer, tmp_path, model=restored)
    assert restored._posttrain_statistics == model._posttrain_statistics
    assert restored._posttrain_provenance == model._posttrain_provenance
    restored._posttrain_statistics = {}
    with pytest.raises(ValueError, match='normalization/split metadata mismatch'):
        Wan22Trainer._read_trainer_state_metadata(trainer, tmp_path, model=restored)


def test_invalid_action_values_cannot_change_loss():
    torch.set_num_threads(1)
    model, sample = tiny_model(), short_sample()
    modified = copy.deepcopy(sample)
    modified['action'][~modified['action_valid_mask']] = float('nan')
    torch.manual_seed(91)
    a, _ = model.training_loss(sample)
    torch.manual_seed(91)
    b, _ = model.training_loss(modified)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_short_offset_validation_runs_and_excludes_padded_actions(tmp_path):
    from types import SimpleNamespace
    from w0.trainer import Wan22Trainer
    model = tiny_model().eval()
    sample = short_sample()
    sample['prompt'] = ['test']
    unbatched = {key: value[0] for key, value in sample.items()}
    sample = Wan22Trainer._to_batched_eval_sample(unbatched)
    p = processor(tmp_path)
    sample['_eval_dataset'] = SimpleNamespace(denormalize_action=lambda action, domain: p.unpack_action(action))
    metrics = model.evaluate_validation(sample, eval_num_inference_steps=1)
    assert all(torch.isfinite(torch.tensor(metrics[key])) for key in ('val_loss', 'action_l1', 'action_l2'))
    predicted = sample['action'][0, 3:67].clone()
    predicted[6:] = 1000  # All of these output ticks are episode-tail padding.
    assert model._evaluate_action_metrics(sample=sample, pred_action=predicted) == {'action_l1': 0., 'action_l2': 0.}


def test_bfloat16_model_preserves_float32_pixels_until_dino_normalization():
    model, sample = tiny_model(), short_sample()
    model.to(dtype=torch.bfloat16)
    model.torch_dtype = torch.bfloat16
    captured = []
    def features(images):
        captured.append(images.clone())
        return torch.zeros(len(images), 4, 768, dtype=torch.bfloat16)
    with patch.object(model.dino_encoder, 'encode_frames', side_effect=features):
        model._build_offset_obs_context(sample, offsets=sample['action_offset'], action_horizon=64, tiled=False)
        state = model.prefill_video(input_image=sample['video'][:, :, 0], action_horizon=64, domain_id=0,
                                   context=sample['context'], context_mask=sample['context_mask'])
        model.infer_action_chunk(inference_state=state, chunk_index=0, num_inference_steps=1,
                                 chunk_obs_image=sample['chunk_obs_images'][:, 0],
                                 chunk_proprio=sample['proprio'][:, 3])
    assert all(value.dtype == torch.float32 for value in captured)
    torch.testing.assert_close(captured[0], sample['chunk_obs_images'].flatten(0, 1), rtol=0, atol=0)
    torch.testing.assert_close(captured[1], sample['chunk_obs_images'][:, 0], rtol=0, atol=0)


@pytest.mark.parametrize('offset,horizon', [(15, 96), (0, 96), (0, 64)])
@pytest.mark.parametrize('jitter_enabled', [False, True])
def test_short_lerobot_episode_uses_interval_end_commands_and_current_proprio(tmp_path, monkeypatch, offset, horizon, jitter_enabled):
    import json
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import w0.datasets.robotwin_posttrain as data
    root = tmp_path / 'robotwin'
    (root / 'meta').mkdir(parents=True)
    (root / 'data').mkdir()
    info = {'fps': 50, 'chunks_size': 1000, 'data_path': 'data/{episode_index}.parquet',
            'video_path': 'videos/{video_key}/{episode_index}.mp4',
            'features': {k: {'shape': [14], 'names': [data.JOINT_NAMES]} for k in ('action', 'observation.state')}}
    (root / 'meta/info.json').write_text(json.dumps(info))
    (root / 'meta/episodes.jsonl').write_text(json.dumps({'episode_index': 0, 'length': 13, 'tasks': ['test']}) + '\n')
    commands = np.repeat((np.arange(13, dtype=np.float32) / 100)[:, None], 14, axis=1)
    pq.write_table(pa.table({'action': commands.tolist(), 'observation.state': (commands + .5).tolist(),
                            'timestamp': np.arange(13) / 50}), root / 'data/0.parquet')
    p = processor(tmp_path)
    statistics = dict(p.statistics, metadata_sha256=data.read_metadata(root)[2], split_seed=42, val_set_proportion=.01)
    Path(p.stats_path).write_text(json.dumps(statistics))
    p.set_normalizer_from_stats(statistics)
    dataset = data.RobotwinPosttrainDataset(str(root), p, is_training_set=not data.is_validation(0, 42, .01),
                                          max_action_offset=offset, video_plan_horizon=horizon,
                                          color_jitter={'enabled': jitter_enabled})
    selected = []
    def decode(path, positions, fps):
        selected.append(positions.copy())
        return torch.full((len(positions), 3, 4, 4), 100, dtype=torch.uint8)
    monkeypatch.setattr(data, 'decode_native_frames', decode)
    monkeypatch.setattr(dataset, '_text', lambda task: (task, torch.zeros(128, 4096), torch.ones(128, dtype=torch.bool)))
    sample = dataset[0]
    assert sample['action'].shape == sample['proprio'].shape == (64 + offset, 37)
    assert sample['video'].shape[1] == horizon // 8 + 1
    if offset == 0:
        assert sample['action_offset'].item() == 0
        assert not dataset._action_offset_enabled
    phase = int(selected[0][0])
    assert 0 <= phase < 4
    expected_count = (12 - phase) // 4
    assert (~sample['action_is_pad']).sum() == expected_count
    torch.testing.assert_close(p.unpack_action(sample['action'])[0], torch.tensor(commands[phase + 3]))
    expected_state, _ = p.pack(torch.tensor(commands[phase] + .5), 'proprio')
    torch.testing.assert_close(sample['proprio'][0], expected_state)
    assert not sample['action'][~sample['action_valid_mask']].any()
    assert not sample['proprio'][~sample['proprio_valid_mask']].any()
    assert sample['image_is_pad'][1:].all()
    assert selected[0][-1] == 12
    assert sample['chunk_obs_images'].shape == (4, 3, 384, 320)
    assert torch.equal(sample['video'][:, 0], sample['chunk_obs_images'][0])
    if jitter_enabled:
        from w0.datasets.color_jitter import VideoColorJitter
        dataset.color_jitter = VideoColorJitter(False)
        plain = dataset[0]
        assert not torch.equal(plain['video'], sample['video'])
        for key in sample.keys() - {'video', 'chunk_obs_images'}:
            if torch.is_tensor(sample[key]):
                assert torch.equal(plain[key], sample[key]), key
            else:
                assert plain[key] == sample[key], key
