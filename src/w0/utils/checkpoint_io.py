"""Publish complete checkpoints and retain only explicitly managed states."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


@contextmanager
def atomic_checkpoint_path(destination):
    destination = Path(destination)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.incomplete', dir=destination.parent)
    os.close(fd)
    try:
        yield temporary
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prune_training_states(state_root, keep, current_step):
    """Delete only older, complete states produced by this save protocol.

    Weight files and external resume sources are never included.
    """
    if keep == 0:
        return []
    candidates = []
    for path in Path(state_root).iterdir():
        match = re.fullmatch(r'step_(\d+)', path.name)
        if not match or path.is_symlink() or not path.is_dir():
            continue
        step = int(match.group(1))
        marker = path / 'checkpoint_complete.json'
        if step <= current_step and marker.is_file() and (path / 'trainer_state.json').is_file():
            if json.loads(marker.read_text()).get('global_step') == step:
                candidates.append((step, path))
    candidates.sort()
    removed = []
    for _, path in candidates[:-keep]:
        shutil.rmtree(path)
        removed.append(str(path))
    return removed
