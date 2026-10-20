"""Build exact train-only q01/q99 statistics for RoboTwin LeRobot exports."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from w0.datasets.robotwin_posttrain import (
    RECIPE, is_validation, read_metadata, read_numeric,
)


def prepare(root: Path, output: Path, *, seed: int = 42,
            validation_fraction: float = 0.01, workers: int = 8) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    info, episodes, fingerprint = read_metadata(root)
    train = [e for e in episodes
             if not is_validation(e["episode_index"], seed, validation_fraction)]
    if not train:
        raise ValueError("No training episodes.")

    def read(episode):
        action, state = read_numeric(root, info, episode)
        return action[3:-1], state[:-4]

    streams = [[], []]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        values_iter = tqdm(pool.map(read, train), total=len(train),
                           desc="train statistics")
        for values in values_iter:
            for storage, value in zip(streams, values):
                storage.append(value)
    result = dict(
        recipe=RECIPE,
        metadata_sha256=fingerprint,
        split_seed=seed,
        val_set_proportion=validation_fraction,
        train_episodes=len(train),
        validation_episodes=len(episodes) - len(train),
        position_selection="union_all_reachable_phase_tick_no_padding",
        quantile_method="exact_numpy_linear",
    )
    for kind, values in zip(("action", "proprio"), streams):
        joined = np.concatenate(values)
        q01, q99 = np.quantile(
            joined.astype(np.float64), [0.01, 0.99], axis=0, method="linear",
        )
        fallback = {}
        for i in range(14):
            if q01[i] == q99[i] and joined[:, i].min() != joined[:, i].max():
                fallback[str(i)] = [float(joined[:, i].min()),
                                    float(joined[:, i].max())]
        result[kind] = dict(q01=q01.tolist(), q99=q99.tolist(),
                            minmax_bounds=fallback,
                            valid_count=[len(joined)] * 14)
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
        default=Path(os.environ.get("W0_ROBOTWIN_DATA", "data/robotwin")),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare(args.dataset_root, args.output, seed=args.seed,
            validation_fraction=args.validation_fraction, workers=args.workers)
