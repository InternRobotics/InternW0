"""Run the real trainer briefly and verify prompt updates, without saving large weights.

Pass normal scripts/train.py Hydra arguments, including a bounded max_steps.
Tiny-model tests separately cover native and distributed checkpoint round trips.
"""

import json
from pathlib import Path
import runpy
from unittest.mock import patch

import torch

from w0.trainer import Wan22Trainer


original_train = Wan22Trainer.train


def checked_train(trainer):
    model = trainer.accelerator.unwrap_model(trainer.model)
    hub = model.action_expert.soft_prompt_hub
    trainer._probe_prompt_initial = hub.weight.detach().clone()
    trainer._probe_prompt_grad_norms = []
    def record(grad):
        assert torch.isfinite(grad).all(), "Non-finite prompt gradient."
        trainer._probe_prompt_grad_norms.append(float(grad.float().norm()))
    hook = hub.weight.register_hook(record)
    try:
        return original_train(trainer)
    finally:
        hook.remove()


def report(trainer):
    model = trainer.accelerator.unwrap_model(trainer.model)
    current = model.action_expert.soft_prompt_hub.weight.detach()
    changed = int((current != trainer._probe_prompt_initial).sum())
    assert changed > 0, "Prompt parameters did not change."
    assert any(v > 0 for v in trainer._probe_prompt_grad_norms), "No nonzero prompt gradient."
    info = {
        "rank": trainer.accelerator.process_index, "step": trainer.global_step,
        "world_size": trainer.accelerator.num_processes, "batch_size": trainer.batch_size,
        "changed_prompt_values": changed, "prompt_parameters": current.numel(),
        "prompt_gradient_norms": trainer._probe_prompt_grad_norms,
        "learning_rates": {g.get("name", "default"): g["lr"] for g in trainer.optimizer.param_groups},
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "result": "PASS",
    }
    output = Path(trainer.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"soft_prompt_probe_rank{info['rank']}.json").write_text(json.dumps(info, indent=2))
    print("SOFT_PROMPT_PROBE " + json.dumps(info), flush=True)
    trainer.accelerator.wait_for_everyone()
    return {"weights_path": "not-saved-probe", "state_path": "not-saved-probe"}


if __name__ == "__main__":
    with patch.object(Wan22Trainer, "train", checked_train), patch.object(Wan22Trainer, "save_checkpoint", report):
        runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
