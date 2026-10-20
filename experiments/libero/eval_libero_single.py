"""Evaluate a w0 LIBERO posttrain checkpoint in the official LIBERO env."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.libero.w0_policy.deploy_policy import get_model
from experiments.libero.libero_utils import DUMMY_ACTION, MAX_STEPS, get_libero_env
from libero.libero import benchmark

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _max_steps(cfg: DictConfig, suite: str) -> int:
    mapping = OmegaConf.to_container(cfg.EVALUATION.get("max_steps_by_suite") or {}, resolve=True)
    return int(mapping.get(suite, MAX_STEPS[suite]))


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval_libero")
def main(cfg: DictConfig):
    ckpt = cfg.get("ckpt")
    if not ckpt:
        raise ValueError("Pass ckpt=/path/to/finetuned_checkpoint.pt")
    suite_name = str(cfg.EVALUATION.task_suite_name)
    task_id = int(cfg.EVALUATION.task_id)
    num_trials = int(cfg.EVALUATION.num_trials)
    seed = int(cfg.EVALUATION.get("seed", 42))
    settle_steps = int(cfg.EVALUATION.get("settle_steps", 30))
    settle_action = list(cfg.EVALUATION.get("settle_action", DUMMY_ACTION))
    output_dir = Path(str(cfg.EVALUATION.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = get_model({
        "cfg": cfg,
        "ckpt": str(ckpt),
        "dataset_stats_path": cfg.EVALUATION.dataset_stats_path,
        "text_cache_dir": cfg.EVALUATION.get("text_cache_dir"),
        "device": cfg.EVALUATION.get("device", "cuda"),
        "seed": seed,
    })
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    env, instruction = get_libero_env(task, seed=seed)
    init_states = task_suite.get_task_init_states(task_id)
    max_steps = _max_steps(cfg, suite_name)
    successes = []
    for trial in range(num_trials):
        policy.reset()
        env.reset()
        obs = env.set_init_state(init_states[trial % len(init_states)])
        done = False
        for _ in range(settle_steps):
            obs, _, done, _ = env.step(settle_action)
        pbar = tqdm(range(max_steps), desc=f"{suite_name} task{task_id} trial{trial + 1}")
        for _ in pbar:
            if done:
                break
            action = policy.act(obs, instruction)
            obs, _, done, _ = env.step(action.tolist())
        successes.append(bool(done))
        pbar.close()
        print(f"trial={trial} success={int(done)} running={float(np.mean(successes)):.3f}", flush=True)
    summary = {
        "suite": suite_name, "task_id": task_id, "instruction": instruction,
        "num_trials": num_trials, "successes": int(np.sum(successes)),
        "success_rate": float(np.mean(successes)), "ckpt": str(ckpt),
    }
    (output_dir / f"{suite_name}_task{task_id}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
