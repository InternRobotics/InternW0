"""Explicit dataset domain identity and reproducible normalization provenance."""

import hashlib
from numbers import Integral
from pathlib import Path


def validate_domain_id(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("dataset domain_id must be a nonnegative integer.")
    return int(value)


def domain_data_recipe(dataset_dirs, domain_ids, stats_path):
    stats = Path(stats_path) if stats_path else None
    return {
        "sources": [{"path": str(Path(p).resolve()), "domain_id": d}
                    for p, d in zip(dataset_dirs, domain_ids)],
        "stats_sha256": hashlib.sha256(stats.read_bytes()).hexdigest() if stats and stats.is_file() else None,
    }
