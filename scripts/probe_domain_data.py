"""Read and validate one real training sample from every registered domain."""

import argparse
import json

import torch

from w0.datasets.domain_video_dataset import DomainVideoDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--text-encoder-id", default="truewan22")
    parser.add_argument("--max-sources-per-domain", type=int, default=1)
    args = parser.parse_args()
    dataset = DomainVideoDataset(args.registry, args.stats, args.text_cache,
                                 text_encoder_id=args.text_encoder_id,
                                 max_sources_per_domain=args.max_sources_per_domain)
    for i, config in enumerate(dataset.domain_configs):
        index = int((i + 0.5) * len(dataset) / len(dataset.domain_configs))
        data = dataset[index]
        assert int(data["domain_id"]) == i
        assert data["video"].shape == (3, 9, 384, 320)
        assert data["action"].shape == (64, dataset.action_dim)
        assert data["proprio"].shape == (65, dataset.proprio_dim)
        assert data["context"].shape == (128, 4096)
        for key in ("video", "action", "proprio", "context"):
            assert torch.isfinite(data[key]).all(), key
        assert torch.count_nonzero(data["action"][:, config["action_dim"]:]) == 0
        assert torch.count_nonzero(data["proprio"][:, config["proprio_dim"]:]) == 0
        print(json.dumps({"domain": config["name"], "action_dim": config["action_dim"],
                          "proprio_dim": config["proprio_dim"], "result": "PASS"}), flush=True)


if __name__ == "__main__":
    main()
