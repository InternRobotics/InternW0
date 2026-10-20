"""Full-epoch RoboTwin sampling at the retained 25 Hz stride."""
import json

import numpy as np
import pytest
import torch

import w0.datasets.robotwin_posttrain as data
from w0.utils.samplers import ResumableEpochSampler


def make_dataset(tmp_path, monkeypatch, **kwargs):
    episodes = [dict(episode_index=i, length=n, instruction=str(i))
                for i, n in enumerate((5, 13, 300))]
    info = dict(video_path="{episode_index}/{video_key}.mp4", chunks_size=1000)
    monkeypatch.setattr(data, "read_metadata", lambda root: (info, episodes, "fixture"))
    monkeypatch.setattr(data, "is_validation", lambda *args: False)
    stats = dict(recipe=data.RECIPE, metadata_sha256="fixture", split_seed=42,
                 val_set_proportion=.01)
    for kind in ("action", "proprio"):
        stats[kind] = dict(q01=[-2.] * 14, q99=[2.] * 14, minmax_bounds={})
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    options = dict(
        max_action_offset=0,
        sampling_stride=2,
        epoch_sampling="all_valid_windows",
        samples_per_epoch=None,
    )
    options.update(kwargs)
    dataset = data.RobotwinPosttrainDataset(
        str(tmp_path), data.RobotwinPosttrainProcessor(str(path)), **options)

    def numeric(episode):
        x = np.repeat((np.arange(episode["length"], dtype=np.float32) / 1000)[:, None], 14, 1)
        return x, x + .5

    monkeypatch.setattr(dataset, "_numeric", numeric)
    monkeypatch.setattr(dataset, "_text", lambda text: (
        text, torch.zeros(1), torch.ones(1, dtype=torch.bool)))
    monkeypatch.setattr(data, "decode_native_frames",
                        lambda path, positions, fps: torch.zeros(len(positions), 3, 2, 2))
    monkeypatch.setattr(data, "camera_canvas", lambda images: images[0])
    return dataset


def test_every_phase_anchor_once_with_real_first_action_and_masked_tail(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path, monkeypatch)
    expected = [(episode, anchor)
                for episode, length in enumerate((5, 13, 300))
                for anchor in range(length - 2)]
    assert len(dataset) == len(expected) == 312
    seen = []
    for index, (episode, anchor) in enumerate(expected):
        sample = dataset[index, 3]
        assert sample["action_offset"].item() == 0
        assert int(sample["prompt"]) == episode
        assert not sample["action_is_pad"][0]
        action = dataset.processor.unpack_action(sample["action"])[0, 0]
        assert action.item() == pytest.approx((anchor + 1) / 1000, abs=3e-7)
        valid = min(64, ((5, 13, 300)[episode] - 1 - anchor) // 2)
        assert (~sample["action_is_pad"]).sum().item() == valid
        assert not sample["action"][~sample["action_valid_mask"]].any()
        assert not sample["proprio"][~sample["proprio_valid_mask"]].any()
        seen.append((int(sample["prompt"]), round(action.item() * 1000) - 1))
    assert seen == expected
    sampler = ResumableEpochSampler(dataset, seed=42, batch_size=16, num_processes=128)
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        assert sorted(i for i, _ in indices) == list(range(len(dataset)))
        assert all(e == epoch for _, e in indices)


def test_epoch_mode_rejects_conflicting_size(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="samples_per_epoch=null"):
        make_dataset(tmp_path, monkeypatch, samples_per_epoch=100)
    with pytest.raises(ValueError, match="epoch_sampling"):
        make_dataset(tmp_path, monkeypatch, epoch_sampling="unknown")
