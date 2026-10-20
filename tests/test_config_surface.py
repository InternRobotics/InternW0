"""The public Hydra surface stays small and aligned with released recipes."""
from pathlib import Path
import json

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = {
    "train.yaml",
    "pretrain.yaml",
    "posttrain_robotwin.yaml",
    "posttrain_libero.yaml",
    "eval_robotwin.yaml",
    "eval_libero.yaml",
    "model/w0.yaml",
    "data/pretrain.yaml",
    "data/robotwin.yaml",
    "data/libero.yaml",
    "data/domain_banks/internw0_domains.json",
}


def test_only_the_documented_config_surface_is_published():
    files = {
        path.relative_to(ROOT / "configs").as_posix()
        for path in (ROOT / "configs").rglob("*")
        if path.is_file()
    }
    assert files == EXPECTED_FILES


def test_all_public_entries_compose_and_resolve():
    names = [
        "pretrain",
        "posttrain_robotwin",
        "posttrain_libero",
        "eval_robotwin",
        "eval_libero",
    ]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        for name in names:
            cfg = compose(config_name=name)
            OmegaConf.resolve(cfg)
            assert cfg.model._target_ == "w0.runtime.create_w0"
            assert cfg.model.observation_encoder == "dino"
            assert cfg.model.future_training_mode == "idm"


def test_robotwin_config_matches_best_clean2random_recipe():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="posttrain_robotwin")
    assert cfg.batch_size == 12
    assert cfg.learning_rate == 3.0e-5
    assert cfg.num_epochs == 15
    assert cfg.model.max_action_offset == 0
    assert cfg.data.train.sampling_stride == 2
    assert cfg.data.train.epoch_sampling == "all_valid_windows"
    assert cfg.data.train.samples_per_epoch is None
    assert cfg.data.train.color_jitter.enabled is True


def test_libero_config_matches_best_goal_long_random_frame_recipe():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="posttrain_libero")
    assert cfg.batch_size == 48
    assert cfg.learning_rate == 5.0e-5
    assert cfg.num_epochs == 10
    assert cfg.model.max_action_offset == 15
    assert cfg.data.train.epoch_sampling == "random_frame"
    assert list(cfg.data.train.include_suites) == ["libero_goal", "libero_10"]
    assert cfg.data.train.samples_per_epoch == 100000


def test_evaluation_defaults_match_released_protocols():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        robotwin = compose(config_name="eval_robotwin")
        libero = compose(config_name="eval_libero")
    assert robotwin.EVALUATION.chunks_per_video_prefill == 4
    assert robotwin.EVALUATION.target_fps == 25.0
    assert robotwin.EVALUATION.control_mode == "fixed_period"
    assert robotwin.EVALUATION.eval_num_episodes == 15
    assert libero.EVALUATION.task_suite_name == "libero_goal"
    assert list(libero.MULTIRUN.task_suite_names) == ["libero_goal", "libero_10"]


def test_robotwin_text_precompute_reads_episode_tasks(tmp_path):
    from scripts.precompute_text_embeds import (
        _collect_dataset_settings, _read_unique_prompts,
    )

    root = tmp_path / "robotwin"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["pick the cup", "lift the cup"]})
        + "\n"
    )
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="posttrain_robotwin", overrides=[
            f"data.train.dataset_root={root}",
        ])
    dirs, caches, _ = _collect_dataset_settings(cfg.data)
    assert dirs == [str(root)]
    assert caches == [root / "text_embeds_cache"]
    assert len(_read_unique_prompts(dirs)) == 2
