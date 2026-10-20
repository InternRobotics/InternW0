"""Pool native action/proprio moments from training trajectories only."""

import argparse
import json
from pathlib import Path

import numpy as np

from w0.datasets.domain_registry import read_registry, registry_fingerprint, episode_is_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sources-per-domain", type=int)
    parser.add_argument("--max-episodes-per-source", type=int, help="Optional bounded preparation, recorded in output")
    args = parser.parse_args()
    registry = read_registry(args.registry)
    configs = {d["name"]: d for d in registry["domain_configs"]}
    moments, counts, episodes_used = {}, {}, {}
    for source in registry["sources"]:
        name = source["domain"]
        if args.max_sources_per_domain is not None and counts.get(name, 0) >= args.max_sources_per_domain:
            continue
        counts[name] = counts.get(name, 0) + 1
        selected = 0
        with (Path(source["root"]) / "meta/episodes_stats.jsonl").open() as episode_file:
            for line in episode_file:
                ep = json.loads(line)
                if episode_is_validation(source, ep["episode_index"], args.val_fraction, args.seed):
                    continue
                for kind, fields in (("action", "action_fields"), ("proprio", "proprio_fields")):
                    stats = [ep["stats"][f["key"]] for f in configs[name][fields]]
                    n = [int(np.asarray(s["count"]).reshape(-1)[0]) for s in stats]
                    if len(set(n)) != 1 or n[0] <= 0:
                        raise ValueError(f"Inconsistent feature counts: {source['root']}")
                    mean = np.concatenate([np.asarray(s["mean"]).reshape(-1) for s in stats])
                    std = np.concatenate([np.asarray(s["std"]).reshape(-1) for s in stats])
                    key = (name, kind)
                    count, total, second = moments.get(key, (0, np.zeros_like(mean), np.zeros_like(mean)))
                    moments[key] = (count + n[0], total + n[0] * mean, second + n[0] * (std**2 + mean**2))
                selected += 1
                episodes_used[name] = episodes_used.get(name, 0) + 1
                if args.max_episodes_per_source and selected >= args.max_episodes_per_source:
                    break
    domains = {}
    for name in configs:
        domains[name] = {}
        for kind in ("action", "proprio"):
            count, total, second = moments[name, kind]
            mean = total / count
            domains[name][kind] = {"mean": mean.tolist(), "std": np.sqrt(np.maximum(second / count - mean**2, 0)).tolist(), "count": count}
    result = {"version": 1, "registry_fingerprint": registry_fingerprint(registry), "domains": domains,
              "split_seed": args.seed, "val_set_proportion": args.val_fraction,
              "episodes_used": episodes_used, "max_sources_per_domain": args.max_sources_per_domain,
              "max_episodes_per_source": args.max_episodes_per_source}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(path), "episodes_used": episodes_used}))


if __name__ == "__main__":
    main()
