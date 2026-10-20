"""Build exact train-only q01/q99 statistics for LIBERO native EEF10."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from w0.datasets.libero_eef import EEF10_DIM, ROT6D_DIMS
from w0.datasets.libero_posttrain import (
    RECIPE, discover_roots, is_validation, read_metadata, read_numeric,
)


def prepare(root: Path, output: Path, *, seed: int = 42, validation_fraction: float = 0.01,
            workers: int = 8, extra_roots: list[Path] | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    roots = []
    for path in [root, *(extra_roots or [])]:
        roots.extend(discover_roots(Path(path)))
    jobs = []
    fingerprint = __import__("hashlib").sha256()
    for dataset_root in roots:
        info, episodes, digest, _ = read_metadata(dataset_root)
        fingerprint.update(digest.encode())
        fingerprint.update(Path(dataset_root).name.encode())
        for episode in episodes:
            if not is_validation(episode["suite"], episode["episode_index"], seed, validation_fraction):
                jobs.append((dataset_root, info, episode))
    if not jobs:
        raise ValueError("No training episodes.")

    def read(job):
        dataset_root, info, episode = job
        action, state = read_numeric(dataset_root, info, episode)
        return action, state

    streams = [[], []]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for values in tqdm(pool.map(read, jobs), total=len(jobs), desc="train statistics"):
            for storage, value in zip(streams, values):
                storage.append(value)
    result = dict(
        recipe=RECIPE, metadata_sha256=fingerprint.hexdigest(), split_seed=seed,
        val_set_proportion=validation_fraction, train_episodes=len(jobs),
        validation_episodes=None,
        position_selection="union_all_native_ticks_no_padding",
        quantile_method="exact_numpy_linear",
        rot6d_identity_dims=list(ROT6D_DIMS),
        dataset_roots=[str(path) for path in roots],
    )
    all_episodes = 0
    for dataset_root in roots:
        _, episodes, _, _ = read_metadata(dataset_root)
        all_episodes += len(episodes)
    result["validation_episodes"] = all_episodes - len(jobs)
    for kind, values in zip(("action", "proprio"), streams):
        joined = np.concatenate(values)
        q01, q99 = np.quantile(joined.astype(np.float64), [0.01, 0.99], axis=0, method="linear")
        for dim in ROT6D_DIMS:
            q01[dim], q99[dim] = -1.0, 1.0
        fallback = {}
        for i in range(EEF10_DIM):
            if i in ROT6D_DIMS:
                continue
            if q01[i] == q99[i] and joined[:, i].min() != joined[:, i].max():
                fallback[str(i)] = [float(joined[:, i].min()), float(joined[:, i].max())]
        result[kind] = dict(
            q01=q01.tolist(), q99=q99.tolist(), minmax_bounds=fallback,
            valid_count=[len(joined)] * EEF10_DIM,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.link(temporary, output)
    temporary.unlink()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path(os.environ.get("W0_LIBERO_DATA", "data/libero")),
    )
    parser.add_argument("--extra-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare(
        args.dataset_root, args.output, seed=args.seed,
        validation_fraction=args.validation_fraction, workers=args.workers,
        extra_roots=args.extra_root,
    )
