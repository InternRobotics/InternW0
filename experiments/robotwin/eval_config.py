"""Resolve one evaluation configuration and carry it unchanged to workers."""
import hashlib
from pathlib import Path

from omegaconf import OmegaConf, open_dict


def checkpoint_tag(path):
    path = Path(path).resolve()
    # Include the checkpoint step and path: two weights from one run must not
    # overwrite one another's task results.
    return f'{path.stem}_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}'


def resolve_training_config(cfg, checkpoint):
    """Posttrain deployment uses the saved training model/data, not defaults."""
    if not cfg.get('posttrain_required', False):
        return cfg
    explicit = cfg.EVALUATION.get('training_config')
    path = Path(explicit).expanduser().resolve() if explicit else None
    trained = cfg
    if path is None:
        # A composed eval_robotwin config already inherits the released
        # posttraining graph. Use it directly when no run-specific YAML is set.
        pass
    elif not path.is_file():
        raise FileNotFoundError('Posttrain evaluation needs the original resolved training config. '
                                'Set EVALUATION.training_config=/path/to/run/config/experiment.yaml.')
    else:
        trained = OmegaConf.load(path)
        OmegaConf.resolve(trained)
    checkpoint = Path(checkpoint).resolve()
    if path is None and checkpoint.parent.name == 'weights' and checkpoint.parent.parent.name == 'checkpoints':
        candidate = checkpoint.parent.parent.parent / 'config.yaml'
        if candidate.is_file():
            path = candidate
    if not trained.get('posttrain_required', False):
        raise ValueError('training_config must describe posttraining.')
    target = trained.data.train._target_
    if target != 'w0.datasets.robotwin_posttrain.RobotwinPosttrainDataset':
        raise ValueError('Fixed-clock deployment currently supports LeRobot RoboTwin posttraining only; '
                         'Motus has a different image/text recipe and needs a separate adapter.')
    stride = trained.data.train.get('sampling_stride', 4)
    if isinstance(stride, bool) or stride not in (2, 4):
        raise ValueError('Expected sampling_stride=2 or 4.')
    with open_dict(cfg):
        for key in ('model', 'data', 'posttrain_stats', 'mixed_precision'):
            cfg[key] = OmegaConf.to_container(trained[key], resolve=True) if OmegaConf.is_config(trained[key]) else trained[key]
        cfg.EVALUATION.training_config = str(path) if path else 'composed:eval_robotwin'
        cfg.EVALUATION.training_config_sha256 = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path else hashlib.sha256(OmegaConf.to_yaml(cfg).encode()).hexdigest()
        )
        # An explicit stats path must exist, and checkpoint validation later
        # requires numerical equality. Never silently fall back to Full stats.
        cfg.EVALUATION.dataset_stats_path = cfg.EVALUATION.get('dataset_stats_path') or trained.posttrain_stats
        cfg.EVALUATION.action_horizon = int(trained.model.action_horizon)
        cfg.EVALUATION.target_fps = 50.0 / stride
        cfg.EVALUATION.sampling_stride = stride
        cfg.EVALUATION.control_mode = cfg.EVALUATION.get('control_mode') or 'fixed_period'
        cfg.EVALUATION.text_cache_dir = str(Path(trained.data.train.dataset_root) / 'text_embeds_cache')
    return cfg
