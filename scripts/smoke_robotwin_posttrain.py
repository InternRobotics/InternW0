"""Run one real pretrained-model AdamW step, without saving or launching a long run."""
import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import torch

from w0.trainer import Wan22Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/robotwin_posttrain_smoke.json'))
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'configs'), version_base=None):
        cfg = compose(config_name='posttrain_robotwin')
    dataset = instantiate(cfg.data.train)
    raw = dataset[0]
    batch = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else [v] for k, v in raw.items()}
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device='cuda:0')
    model.bind_training_data_recipe(dataset)
    Wan22Trainer._apply_dit_only_train_mode(model)
    model.load_checkpoint(args.checkpoint)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, foreach=False)
    loss, metrics = model.training_loss(batch)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite loss.')
    loss.backward()
    gradients = {}
    for name, module in [('encoder', model.action_expert.action_encoder), ('head', model.action_expert.head),
                         ('prompt', model.action_expert.soft_prompt_hub), ('proprio', model.proprio_encoder)]:
        gradient = module.weight.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f'Missing/nonfinite {name} gradient.')
        gradients[name] = float(gradient.float().abs().sum())
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError('Nonfinite model gradient.')
    optimizer.step()
    torch.cuda.synchronize()
    result = dict(loss=float(loss), metrics=metrics, gradient_l1=gradients,
                  real_action_steps=int((~raw['action_is_pad']).sum()), offset=int(raw['action_offset']),
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
