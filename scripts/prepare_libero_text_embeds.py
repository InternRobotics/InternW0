"""Precompute UMT5 caches for LIBERO episode instructions."""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path

import torch

from w0.datasets.libero_posttrain import collect_prompts


def _load_precompute_module():
    path = Path(__file__).resolve().parent / "precompute_text_embeds.py"
    spec = importlib.util.spec_from_file_location("w0_precompute_text_embeds", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("W0_LIBERO_DATA", "data/libero"),
    )
    parser.add_argument("--extra-root", action="append", default=[])
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("W0_LIBERO_TEXT_CACHE", "assets/libero/text_embeds_cache"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    precompute = _load_precompute_module()
    prompts = collect_prompts(args.dataset_root, args.extra_root or None)
    if not prompts:
        raise ValueError("No LIBERO instructions found.")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_model_dir = os.environ.get("W0_WAN_MODEL_DIR")
    model_id = local_model_dir or precompute.DEFAULT_MODEL_ID
    tokenizer_model_id = (
        local_model_dir
        if local_model_dir
        else precompute.DEFAULT_TOKENIZER_MODEL_ID
    )
    _, text_config, _, tokenizer_config = precompute._resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=not bool(local_model_dir),
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    text_encoder = precompute._load_registered_model(
        text_config.path, "wan_video_text_encoder",
        torch_dtype=torch.bfloat16, device=device,
    ).eval()
    tokenizer = precompute.HuggingfaceTokenizer(
        name=tokenizer_config.path, seq_len=precompute.DEFAULT_CONTEXT_LEN, clean="whitespace",
    )
    enc_id = precompute._model_id_to_enc_id(precompute.DEFAULT_MODEL_ID)
    written = skipped = 0
    with torch.no_grad():
        for start in range(0, len(prompts), precompute.DEFAULT_BATCH_SIZE):
            batch = prompts[start:start + precompute.DEFAULT_BATCH_SIZE]
            ids, mask = tokenizer(batch, return_mask=True, add_special_tokens=True)
            context = text_encoder(ids.to(device), mask.to(device=device, dtype=torch.bool))
            for i, prompt in enumerate(batch):
                path = cache_dir / (
                    f"{hashlib.sha256(prompt.encode()).hexdigest()}"
                    f".t5_len{precompute.DEFAULT_CONTEXT_LEN}.{enc_id}.pt"
                )
                if path.exists() and not args.overwrite:
                    skipped += 1
                    continue
                payload = {
                    "context": context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                    "mask": mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                }
                precompute._atomic_torch_save(payload, path)
                written += 1
    print({"prompts": len(prompts), "written": written, "skipped": skipped, "cache_dir": str(cache_dir)})


if __name__ == "__main__":
    main()
