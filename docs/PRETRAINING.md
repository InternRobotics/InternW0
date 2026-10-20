# Multi-domain pretraining

Multi-domain pretraining trains one shared world/action model while routing each
sample through domain-specific action interfaces:

```text
sample.domain_id
  -> action soft prompt row
  -> action encoder row
  -> action decoder/head row
  -> proprio encoder row
```

VideoDiT remains shared and does not receive a domain ID.

## 1. Build the domain registry

```bash
python scripts/build_action_domain_registry.py \
  --roots external/InternData-A1 external/RoboMIND2 \
  --output configs/data/domain_banks/internw0_domains.json
```

The registry records:

- ordered domain names;
- native action and proprio fields;
- source FPS and camera keys;
- action semantics;
- trajectory roots.

Domain IDs are positional and must not be reordered without explicit bank
migration.

## 2. Prepare normalization statistics

```bash
export W0_DOMAIN_STATS=assets/domain_data/pretrain_stats.json

python scripts/prepare_domain_statistics.py \
  --registry configs/data/domain_banks/internw0_domains.json \
  --output "$W0_DOMAIN_STATS"
```

Only the training split contributes statistics. The output records the registry
fingerprint and split settings.

## 3. Prepare text embeddings

```bash
export W0_DOMAIN_TEXT_CACHE=assets/domain_data/text_embeds

python scripts/precompute_text_embeds.py \
  --config-name pretrain
```

## 4. Train

Single-node example:

```bash
bash scripts/train_zero2.sh 8 \
  --config-name pretrain \
  output_dir=outputs/pretrain
```

Multi-node example:

```bash
NNODES=8 NODE_RANK=0 MASTER_ADDR=<rank0-ip> MASTER_PORT=29500 \
DEEPSPEED_HOSTFILE=hostfile \
bash scripts/train_zero2.sh 8 --config-name pretrain
```

The task uses:

- `action_horizon=64`;
- `action_chunk_size=16`;
- K=8 action soft prompts;
- DINOv3 B/16 observations;
- IDM video/action training;
- no action history;
- equal domain sampling by default.
