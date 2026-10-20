# Checkpoint compatibility

InternW0 uses native `.pt` checkpoints. The public release plan contains three
checkpoints:

```text
base
robotwin-clean
libero
```

Each checkpoint can be passed directly to `init_checkpoint` or an evaluation
entry point. The repository does not bundle model weights.

## Embedded behavior contract

Every native checkpoint saves a `behavior_config` containing at least:

- soft-prompt length and ordered domains;
- domain schemas and valid slots;
- action horizon and chunk size;
- history setting;
- action/video mode and stride;
- observation encoder identity;
- DINO encoder identity when applicable;
- video/action temporal contracts.

Loading performs a semantic comparison before weight loading. Shape equality
alone is not considered sufficient compatibility.

## Training and resuming

`init_checkpoint` loads model weights only, creates a new optimizer, and starts
at step 0. `resume` restores a complete compatible training state, including
model weights, optimizer, scheduler, progress, and distributed state where
available. The two modes are mutually exclusive.

Posttraining uses the released target-specific `.pt` checkpoint:

```text
W0_ROBOTWIN_CHECKPOINT=checkpoints/robotwin-clean.pt
W0_LIBERO_CHECKPOINT=checkpoints/libero.pt
```

Checkpoints produced by this repository are also native `.pt` files and can be
used directly for later inference or compatible resumed training.
