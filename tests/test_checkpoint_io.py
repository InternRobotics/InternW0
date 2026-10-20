import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from w0.trainer import Wan22Trainer
from w0.utils.checkpoint_io import atomic_checkpoint_path, prune_training_states


def test_failed_weight_write_preserves_existing_checkpoint(tmp_path):
    destination = tmp_path / 'weights.pt'
    torch.save({'step': 10}, destination)
    with pytest.raises(OSError):
        with atomic_checkpoint_path(destination) as temporary:
            Path(temporary).write_bytes(b'partial')
            raise OSError('quota exceeded')
    assert torch.load(destination, weights_only=True)['step'] == 10
    assert list(tmp_path.iterdir()) == [destination]


def test_retention_only_removes_complete_managed_states(tmp_path):
    for step in (10, 20):
        path = tmp_path / f'step_{step:06d}'
        path.mkdir()
        (path / 'trainer_state.json').write_text('{}')
        (path / 'checkpoint_complete.json').write_text(json.dumps({'global_step': step}))
    legacy = tmp_path / 'step_000005'; legacy.mkdir()
    incomplete = tmp_path / 'step_000030.incomplete'; incomplete.mkdir()
    removed = prune_training_states(tmp_path, 1, 20)
    assert removed == [str(tmp_path / 'step_000010')]
    assert legacy.exists() and incomplete.exists() and (tmp_path / 'step_000020').exists()


def trainer(tmp_path):
    obj = object.__new__(Wan22Trainer)
    obj.checkpoint_root = str(tmp_path)
    obj.weights_dir = str(tmp_path / 'weights'); Path(obj.weights_dir).mkdir()
    obj.state_dir = str(tmp_path / 'state'); Path(obj.state_dir).mkdir()
    obj.global_step = 10; obj.epoch = 0; obj.batch_in_epoch = 10
    obj.keep_training_states = 1; obj.checkpoint_min_free_gib = 0
    obj._last_saved_checkpoint = None
    obj.model = SimpleNamespace(save_checkpoint=lambda path, **kwargs: torch.save({'step': kwargs['step']}, path))
    obj.accelerator = SimpleNamespace(is_main_process=True, wait_for_everyone=lambda: None,
        unwrap_model=lambda model: model, save_state=lambda output_dir: torch.save({'optimizer': 'test'}, Path(output_dir) / 'optimizer.pt'))
    return obj


def test_state_publication_retention_and_repeated_final_save(tmp_path):
    obj = trainer(tmp_path)
    first = obj.save_checkpoint()
    assert obj.save_checkpoint() == first
    obj.global_step = 20
    second = obj.save_checkpoint()
    assert not Path(first['state_path']).exists()
    assert Path(first['weights_path']).exists()
    assert (Path(second['state_path']) / 'checkpoint_complete.json').exists()
    assert not list(Path(obj.state_dir).glob('*.incomplete'))


def test_failed_state_save_does_not_publish_or_prune(tmp_path):
    obj = trainer(tmp_path)
    first = obj.save_checkpoint()
    obj.global_step = 20
    def fail(**kwargs): raise OSError('quota exceeded')
    obj.accelerator.save_state = fail
    with pytest.raises(OSError, match='quota'):
        obj.save_checkpoint()
    assert Path(first['state_path']).exists()
    assert not (Path(obj.state_dir) / 'step_000020').exists()
    with pytest.raises(ValueError, match='incomplete'):
        obj.load_training_state(str(Path(obj.state_dir) / 'step_000020.incomplete'))


def test_storage_failure_is_broadcast_to_other_ranks(tmp_path, monkeypatch):
    obj = trainer(tmp_path)
    obj.accelerator.is_main_process = False
    def broadcast(values, from_process): values[1] = 'OSError: quota exceeded'
    monkeypatch.setattr('w0.trainer.broadcast_object_list', broadcast)
    with pytest.raises(RuntimeError, match='quota exceeded'):
        obj._checkpoint_main_io(lambda: pytest.fail('worker must not perform rank-zero I/O'))
