# Assets and data contracts

## Directory convention

The examples in this release use the following local layout:

```text
assets/                                   # Optional; paths may also use absolute locations
  wan22/                                 # Wan2.2 model directory
  dinov3-vitb16-pretrain-lvd1689m/       # HF DINOv3 B/16 checkpoint
  domain_data/
    pretrain_stats.json
    text_embeds/
  robotwin/
    normalization.json
  libero/
    normalization.json
    text_embeds_cache/
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

Any layout can be used by setting the corresponding environment variable or
Hydra override.

## Environment variables

| Variable | Meaning | Default in configs |
|---|---|---|
| `W0_WAN_MODEL_DIR` | Local Wan2.2 model directory | `assets/wan22` |
| `W0_DINO_WEIGHTS` | DINOv3 B/16 directory | `assets/dinov3-vitb16-pretrain-lvd1689m` |
| `W0_OUTPUT_DIR` | Root for training outputs | `outputs` |
| `W0_BASE_CHECKPOINT` | Released base `.pt` checkpoint | `checkpoints/base.pt` |
| `W0_ROBOTWIN_CHECKPOINT` | Released RoboTwin-clean `.pt` checkpoint | `checkpoints/robotwin-clean.pt` |
| `W0_LIBERO_CHECKPOINT` | Released LIBERO `.pt` checkpoint | `checkpoints/libero.pt` |
| `W0_ROBOTWIN_DATA` | RoboTwin dataset root | `data/robotwin` |
| `W0_ROBOTWIN_STATS` | RoboTwin normalization file | `assets/robotwin/normalization.json` |
| `W0_LIBERO_DATA` | LIBERO dataset root | `data/libero` |
| `W0_LIBERO_STATS` | LIBERO normalization file | `assets/libero/normalization.json` |
| `W0_LIBERO_TEXT_CACHE` | LIBERO UMT5 cache | `assets/libero/text_embeds_cache` |
| `LIBERO_PATH` | External LIBERO checkout | `external/LIBERO` |

## Released checkpoints

The public release includes exactly three native `.pt` checkpoints in the
[InternW0 collection](https://huggingface.co/collections/InternRobotics/nternw0):

| Checkpoint | Hugging Face repository |
| --- | --- |
| Base | [InternRobotics/InternW0-Base](https://huggingface.co/InternRobotics/InternW0-Base) |
| RoboTwin-clean | [InternRobotics/InternW0-RoboTwin-clean](https://huggingface.co/InternRobotics/InternW0-RoboTwin-clean) |
| LIBERO | [InternRobotics/InternW0-Libero](https://huggingface.co/InternRobotics/InternW0-Libero) |

All are native `.pt` files and can be loaded directly by this repository. Use
the repository's canonical named `.pt` file rather than `pytorch_model.bin`.

## Wan model resolution

Set `W0_WAN_MODEL_DIR` to a complete local Wan2.2 release containing:

```text
diffusion_pytorch_model*.safetensors
diffusion_pytorch_model.safetensors.index.json
Wan2.2_VAE.pth
google/umt5-xxl/
```

The RoboTwin/LIBERO retained tasks set `model.redirect_common_files=false` so a
local Wan directory supplies its original VAE rather than fetching the
alternative shared VAE file.

## Multi-domain registry

The reference registry is:

```text
configs/data/domain_banks/internw0_domains.json
```

It preserves the 11 `dataset/embodiment` entries and source ordering used to
define domain IDs. Source roots are represented as portable paths under
`external/InternData-A1` and `external/RoboMIND2`. They are intentionally not
machine-specific absolute paths.

For a new copy of the source datasets, regenerate the registry:

```bash
python scripts/build_action_domain_registry.py \
  --roots external/InternData-A1 external/RoboMIND2 \
  --output configs/data/domain_banks/internw0_domains.json
```

Statistics must then be regenerated for that registry because the registry
fingerprint is part of the data contract.

## Normalization rules

- RoboTwin uses train-only q01/q99 statistics over the reachable stride-4
  reference command/state positions. The best 25 Hz stride-2 recipe reuses these
  released statistics without refitting them.
- LIBERO uses train-only q01/q99 statistics for native EEF10 values. Rotation
  6D identity dimensions are fixed to `[-1, 1]`.
- Validation episodes never contribute normalization statistics.
- Short episode tails are padded and masked rather than discarded.

## Text embeddings

Instructions are encoded with the Wan UMT5 text encoder and cached as `.pt`
files. RoboTwin reads task lists from `meta/episodes.jsonl`; LIBERO uses its
dedicated preparation script. Cache filenames record the encoder contract and
must be regenerated when the text model or tokenizer changes.
