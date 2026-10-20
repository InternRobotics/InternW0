"""Two-GPU BF16 prompt synchronization and full-state resume smoke test.

PYTHONPATH=src:tests torchrun --standalone --nproc_per_node=2 \
    tests/run_soft_prompt_distributed.py --backend ddp  # or zero2
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from w0.trainer import group_action_prompt_parameters
from test_action_domain_bank import bank_model
from test_future_video import sample


class LossModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        return self.model.training_loss(data)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ddp", "zero2"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    directory = [tempfile.mkdtemp(prefix="aha_prompt_resume_") if rank == 0 else None]
    torch.distributed.broadcast_object_list(directory, src=0)
    try:
        model = bank_model().to(device=device, dtype=torch.bfloat16)
        model.device, model.torch_dtype = device, torch.bfloat16
        wrapper = LossModule(model)
        groups = group_action_prompt_parameters(
            [p for p in model.parameters() if p.requires_grad], model.action_expert.soft_prompt_hub.parameters(),
            lr=5e-5, lr_scale=0.1, max_numel=10000,
        )
        optimizer = torch.optim.AdamW(groups, weight_decay=0)
        if args.backend == "zero2":
            import deepspeed
            wrapper, optimizer, _, _ = deepspeed.initialize(
                model=wrapper, optimizer=optimizer, config={
                    "train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 1,
                    "bf16": {"enabled": True}, "zero_allow_untested_optimizer": True,
                    "zero_optimization": {"stage": 2, "reduce_bucket_size": 100000,
                                          "allgather_bucket_size": 100000},
                },
            )
        else:
            wrapper = DistributedDataParallel(wrapper, device_ids=[device.index], static_graph=True)
        initial = model.action_expert.soft_prompt_hub.weight.detach().clone()
        losses = []
        for step in range(3):
            torch.manual_seed(123 + step * 10 + rank)
            data = dict(sample(), domain_id=torch.tensor([rank % 2]))
            loss = wrapper(data)
            if args.backend == "zero2":
                wrapper.backward(loss)
                wrapper.step()
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            assert torch.isfinite(loss)
            losses.append(float(loss.detach()))
        current = model.action_expert.soft_prompt_hub.weight.detach().clone()
        assert not torch.equal(current, initial), "Prompt bank did not update."
        gathered = [torch.empty_like(current) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, current)
        for other in gathered:
            torch.testing.assert_close(current, other, atol=0, rtol=0)
        for bank in (model.action_expert.action_encoder, model.action_expert.head, model.proprio_encoder):
            rows = [torch.empty_like(bank.weight) for _ in gathered]
            torch.distributed.all_gather(rows, bank.weight.detach().contiguous())
            for row in rows:
                torch.testing.assert_close(bank.weight, row, atol=0, rtol=0)
        if args.backend == "zero2":
            wrapper.save_checkpoint(directory[0], tag="step3", client_state={"step": 3})
            with torch.no_grad():
                model.action_expert.soft_prompt_hub.weight.add_(1)
            _, state = wrapper.load_checkpoint(directory[0], tag="step3")
            assert state["step"] == 3
        else:
            path = Path(directory[0]) / "state.pt"
            if rank == 0:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
            torch.distributed.barrier()
            saved = torch.load(path, map_location=device, weights_only=False)
            with torch.no_grad():
                model.action_expert.soft_prompt_hub.weight.add_(1)
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
        torch.testing.assert_close(current, model.action_expert.soft_prompt_hub.weight, atol=0, rtol=0)
        print(json.dumps({"backend": args.backend, "rank": rank, "result": "PASS",
                          "steps": 3, "changed_prompt_values": int((current != initial).sum()),
                          "losses": losses, "full_state_reload": True}), flush=True)
    finally:
        torch.distributed.barrier()
        if rank == 0:
            shutil.rmtree(directory[0])
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
