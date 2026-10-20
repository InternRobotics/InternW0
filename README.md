<div align="center">

<img src="assets/w0-title.svg" alt="InternW0" width="430">

### A Foundational Physical World Model for Efficient Real-World Interactions

[![Project Page](https://img.shields.io/badge/Project-Page-2f80ed.svg)](https://internrobotics.github.io/internw0)
[![arXiv](https://img.shields.io/badge/arXiv-2609.27656-b31b1b.svg)](https://arxiv.org/abs/2609.27656)
[![Python](https://img.shields.io/badge/Python-3.10-3776ab.svg)](#environment)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-ee4c2c.svg)](#environment)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoints-InternW0-ffcc00.svg)](https://huggingface.co/collections/InternRobotics/nternw0)

</div>

## 📚 Contents

- [🗂️ Repository Layout](#repository-layout)
- [⚙️ Environment](#environment)
- [📦 Model Assets & Checkpoints](#model-assets--checkpoints)
- [🧩 Data Preparation](#data-preparation)
- [🚀 Training](#training)
- [🧪 RoboTwin Evaluation](#robotwin-evaluation)
- [🧪 LIBERO Evaluation](#libero-evaluation)
- [📌 Citation](#citation)
- [🙏 Acknowledgements](#acknowledgements)
- [📄 License](#license)

**InternW0** is a physical world-action model with a slow Wan2.2 VideoDiT
planner and a fast chunked ActionDiT executor. The video expert produces
reusable future context; observation-conditioned routing adapts its layerwise
K/V to each new chunk observation, so the model can predict slowly and act
quickly.

This repository contains:

- multi-domain soft-prompt/domain-bank pretraining;
- DINOv3 observation encoding and IDM video/action training;
- RoboTwin and LIBERO posttraining readers;
- RoboTwin and LIBERO closed-loop evaluation adapters;
- focused implementation-contract and distributed tests.

Pretraining and posttraining use the same native `.pt` checkpoint contract.
Force/tactile, real-robot, and laboratory assets described in the paper are not
included here.


<a id="repository-layout"></a>

## 🗂️ Repository Layout

```text
InternW0/
├── configs/
│   ├── data/                 Dataset and domain-registry configurations
│   ├── model/                W0 DINO/IDM model configuration
│   ├── pretrain.yaml         Multi-domain pretraining entry point
│   ├── posttrain_*.yaml      RoboTwin/LIBERO posttraining entry points
│   ├── eval_*.yaml           RoboTwin/LIBERO evaluation entry points
│   └── train.yaml            Base Hydra training graph
├── scripts/
│   ├── train.py              Hydra training entry point
│   ├── train_zero1.sh        ZeRO-1 launcher
│   ├── train_zero2.sh        ZeRO-2 launcher
│   ├── prepare_*             Statistics, data, and checkpoint preparation
│   └── run_libero_eval.sh    Two-process LIBERO evaluation launcher
├── experiments/
│   ├── robotwin/             RoboTwin policy adapter and evaluation entries
│   └── libero/               LIBERO policy server, client, and adapter
├── src/w0/
│   ├── datasets/             Native readers, normalization, and temporal masks
│   ├── models/wan22/         Wan, W0, ActionDiT, DINO, OVCR, and banks
│   ├── runtime.py            Model/data construction and training entry
│   └── trainer.py            Distributed trainer and checkpoint lifecycle
├── tests/                    Contract and distributed regression tests
├── docs/                     Reproduction and architecture documentation
└── assets/                   README figures
```

<a id="environment"></a>

## ⚙️ Environment

Python 3.10 is required.

```bash
conda create -n w0 python=3.10 -y
conda activate w0
pip install -U pip setuptools wheel
pip install torch==2.7.1 torchvision==0.22.1 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
conda install -c conda-forge ffmpeg -y
```

The Python package is imported as `w0`.

RoboTwin and LIBERO are external simulation dependencies and are not installed
by this package.

<a id="model-assets-and-checkpoints"></a>

## 📦 Model Assets & Checkpoints

InternW0 does not bundle model weights, datasets, or trained checkpoints. Use
explicit local paths or the environment variables below.

| Asset | Purpose | Environment variable |
| --- | --- | --- |
| Wan2.2 TI2V-5B model directory | VideoDiT, VAE, tokenizer, and text encoder | `W0_WAN_MODEL_DIR` |
| DINOv3 ViT-B/16 | Future/IDM observation encoder | `W0_DINO_WEIGHTS` |
| InternW0 checkpoints | Base, RoboTwin-clean, and LIBERO `.pt` weights | `W0_BASE_CHECKPOINT`, `W0_ROBOTWIN_CHECKPOINT`, or `W0_LIBERO_CHECKPOINT` |
| RoboTwin dataset | Joint-space posttraining and evaluation | `W0_ROBOTWIN_DATA` |
| LIBERO dataset | Native 20 Hz EEF10 posttraining | `W0_LIBERO_DATA` |
| LIBERO checkout | Closed-loop LIBERO environment | `LIBERO_PATH` |
| RoboTwin checkout | Closed-loop RoboTwin environment | `EVALUATION.robotwin_root` |

A convenient local layout is:

```text
assets/
  wan22/
  dinov3-vitb16-pretrain-lvd1689m/
checkpoints/
  base.pt
  robotwin-clean.pt
  libero.pt
data/
  robotwin/
  libero/
external/
  RoboTwin/
  LIBERO/
outputs/
```

### Hugging Face checkpoints

All released InternW0 checkpoints are available in the
[InternW0 Hugging Face collection](https://huggingface.co/collections/InternRobotics/nternw0).

| Asset | Hugging Face URL |
| --- | --- |
| [InternW0 base](https://huggingface.co/InternRobotics/InternW0-Base) | `InternRobotics/InternW0-Base` |
| [InternW0 RoboTwin-clean](https://huggingface.co/InternRobotics/InternW0-RoboTwin-clean) | `InternRobotics/InternW0-RoboTwin-clean` |
| [InternW0 LIBERO](https://huggingface.co/InternRobotics/InternW0-Libero) | `InternRobotics/InternW0-Libero` |

Download the canonical `.pt` files rather than the auto-generated
`pytorch_model.bin` copies. For example:

```bash
hf download InternRobotics/InternW0-Base \
  InternW0-Base.pt InternW0-Base.transfer.json normalization.json \
  --local-dir checkpoints/base

hf download InternRobotics/InternW0-RoboTwin-clean \
  InternW0-RoboTwin-clean.pt normalization.json \
  checkpoint_metadata.json posttrain_config.yaml \
  --local-dir checkpoints/robotwin-clean

hf download InternRobotics/InternW0-Libero \
  InternW0-Libero.pt normalization.json \
  checkpoint_metadata.json posttrain_config.yaml \
  --local-dir checkpoints/libero
```

Pass the resulting paths through the environment variables below. Do not commit
model files to this repository.

<a id="data-preparation"></a>

## 🧩 Data Preparation

### RoboTwin

First precompute UMT5 text embeddings from the LeRobot task metadata, then
build train-only q01/q99 command/statistics:

```bash
W0_ROBOTWIN_DATA="$W0_ROBOTWIN_DATA" \
python scripts/precompute_text_embeds.py \
  --config-name posttrain_robotwin

python scripts/prepare_robotwin_stats.py \
  --dataset-root "$W0_ROBOTWIN_DATA" \
  --output assets/robotwin/normalization.json
```

The retained best-practice recipe uses the clean export, 25 Hz sampling
(`sampling_stride=2`), all valid windows, ColorJitter, and offset 0 for
Clean2Random. Actions and proprio are scattered into 37D slots:

```text
[0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 22, 33]
```

### LIBERO

First precompute UMT5 text embeddings, then build train-only statistics:

```bash
python scripts/prepare_libero_text_embeds.py \
  --dataset-root "$W0_LIBERO_DATA" \
  --cache-dir assets/libero/text_embeds_cache

python scripts/prepare_libero_posttrain.py \
  --dataset-root "$W0_LIBERO_DATA" \
  --output assets/libero/normalization.json
```

LIBERO remains at native 20 Hz. Its EEF10 representation occupies:

```text
[7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
```

### Multi-domain pretraining registry

The reference registry is:

```text
configs/data/domain_banks/internw0_domains.json
```

It preserves domain identity, source ordering, native fields, and camera slots.
For a local copy of the source datasets, regenerate it:

```bash
python scripts/build_action_domain_registry.py \
  --roots external/InternData-A1 external/RoboMIND2 \
  --output configs/data/domain_banks/internw0_domains.json
```

Regenerate domain statistics after changing the registry because its fingerprint
is part of the data contract:

```bash
python scripts/prepare_domain_statistics.py \
  --registry configs/data/domain_banks/internw0_domains.json \
  --output assets/domain_data/pretrain_stats.json
```

<a id="training"></a>

## 🚀 Training

InternW0 uses Hydra. `init_checkpoint` loads model weights only and starts a new
optimizer; `resume` restores a complete compatible training state. The two
modes are mutually exclusive.

### Multi-domain soft-prompt pretraining

```bash
export W0_DOMAIN_STATS=assets/domain_data/pretrain_stats.json
export W0_DOMAIN_TEXT_CACHE=assets/domain_data/text_embeds

bash scripts/train_zero2.sh 8 \
  --config-name pretrain \
  output_dir=outputs/pretrain
```

The task uses K=8 prompts, 11 reference domains, DINOv3 B/16 observations, IDM
training, H=64, C=16, and no action history.

### RoboTwin posttraining

Best Clean2Random recipe:

```bash
export W0_ROBOTWIN_DATA=data/robotwin
export W0_ROBOTWIN_STATS=assets/robotwin/normalization.json
export W0_ROBOTWIN_CHECKPOINT=checkpoints/robotwin-clean.pt
export W0_WAN_MODEL_DIR=assets/wan22
export W0_DINO_WEIGHTS=assets/dinov3-vitb16-pretrain-lvd1689m

bash scripts/train_zero1.sh 8 \
  --config-name posttrain_robotwin
```

The config already uses offset 0. Other offsets can be studied with explicit
Hydra overrides.

### LIBERO posttraining

```bash
export W0_LIBERO_DATA=data/libero
export W0_LIBERO_STATS=assets/libero/normalization.json
export W0_LIBERO_CHECKPOINT=checkpoints/libero.pt
export W0_LIBERO_TEXT_CACHE=assets/libero/text_embeds_cache
export W0_WAN_MODEL_DIR=assets/wan22
export W0_DINO_WEIGHTS=assets/dinov3-vitb16-pretrain-lvd1689m

bash scripts/train_zero1.sh 8 \
  --config-name posttrain_libero
```

<a id="robotwin-evaluation"></a>

## 🧪 RoboTwin Evaluation

Use an external RoboTwin checkout:

```bash
python experiments/robotwin/eval_robotwin_single.py \
  EVALUATION.robotwin_root=external/RoboTwin \
  EVALUATION.task_name=<task> \
  EVALUATION.ckpt=<checkpoint.pt>
```

The adapter applies the same camera canvas, action slots, normalization, and
fixed control clock used by the posttraining reader. See
[Evaluation](docs/EVALUATION.md) for the full protocol.

#### Paper result

| Setting | Clean | Randomized | Average |
| --- | ---: | ---: | ---: |
| Full | 93.20 | 93.04 | 93.12 |
| Clean2Random | 83.20 | 68.00 | 75.60 |

Full jointly uses clean and randomized demonstrations. Clean2Random trains only
on clean demonstrations and evaluates transfer to unseen randomized scenes.
These paper-reported numbers are not rerun as part of code-release validation.

<a id="libero-evaluation"></a>

## 🧪 LIBERO Evaluation

The launcher starts a W0 policy server and a separate LIBERO environment client:

```bash
export CKPT=checkpoints/libero.pt
export W0_WAN_MODEL_DIR=assets/wan22
export W0_LIBERO_STATS=assets/libero/normalization.json
export W0_LIBERO_TEXT_CACHE=assets/libero/text_embeds_cache
export LIBERO_PATH=external/LIBERO

SUITE=libero_goal TASK_ID=0 NUM_TRIALS=50 \
bash scripts/run_libero_eval.sh
```

The default protocol executes a complete 16-action chunk unless
`EXECUTE_N_ACTIONS` is explicitly overridden. Changing that value changes the
closed-loop protocol and is not equivalent to training a chunk-8 model.

#### Paper result

InternW0 covers all 40 tasks in the four official LIBERO suites.

| Spatial | Object | Goal | Long | Average |
| ---: | ---: | ---: | ---: | ---: |
| 99.4 | 99.4 | 98.6 | 97.0 | 98.6 |

These paper-reported numbers are not rerun as part of code-release validation.


<a id="citation"></a>

## 📌 Citation

If you find InternW0 useful, please cite:

```bibtex
@article{internw0,
  title = {InternW0: A Foundational Physical World Model for Efficient Real-World Interactions},
  author = {Jisong Cai and Yao Mu and Ganlin Yang and Zhe Cao and Zhangzheng Tu and Xing Gao and Kailin Li and Xinyu Zhan and Lixin Yang and Yangkun Zhu and Haoxiang Ma and Ming Zhou and Qiaojun Yu and Yufei Xue and Liqun He and Yifei Yao and Yifan Zhu and Long Ling and Bingqi Jiang and Haoyu Guo and Xueyue Zhu and Bowen Zhou and Bin Zhao and Tianfan Xue and Chunhua Shen and Weinan Zhang},
  journal = {arXiv preprint arXiv:2609.27656},
  year = {2026}
}
```

<a id="acknowledgements"></a>

## 🙏 Acknowledgements

InternW0 builds on ideas and open-source infrastructure from Wan2.2, DINOv3,
RoboTwin, LIBERO, LeRobot. We thank
their authors
and contributors for releasing useful models, data tools, and evaluation
environments.

<a id="license"></a>

## 📄 License

This repository is released under the [MIT License](./LICENSE), except for
third-party code and assets that retain their original licenses.
