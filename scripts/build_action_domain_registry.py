"""Inspect metadata and build the dataset x embodiment registry; never modify data."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from w0.datasets.domain_registry import source_schema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--info-path-list", help="Optional newline-delimited paths from rg --files -g info.json")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--schema-overrides", help="JSON mapping dataset/embodiment to action_keys, proprio_keys and video_keys")
    args = parser.parse_args()
    roots = [Path(p).resolve() for p in args.roots]
    paths = ([Path(p) for p in Path(args.info_path_list).read_text().splitlines()] if args.info_path_list else
             [p for root in roots for p in root.rglob("meta/info.json")])
    overrides = json.loads(Path(args.schema_overrides).read_text()) if args.schema_overrides else {}
    jobs = [(p, next(root for root in roots if p.is_relative_to(root)), overrides) for p in sorted(paths)]
    configs, sources = {}, []
    with ThreadPoolExecutor(args.workers) as pool:
        for config, source in pool.map(lambda job: source_schema(*job), jobs):
            name = config["name"]
            if name in configs and configs[name] != config:
                raise ValueError(f"Conflicting native schemas within {name}: {source['root']}; split or align explicitly.")
            configs[name] = config
            sources.append(source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"version": 1, "domain_configs": [configs[k] for k in sorted(configs)],
                                 "sources": sources}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"domains": len(configs), "sources": len(sources), "output": str(output)}))


if __name__ == "__main__":
    main()
