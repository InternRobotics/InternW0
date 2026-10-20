import json
from pathlib import Path

import numpy as np
import torch

from w0.datasets.libero_eef import (
    EEF10_DIM, SLOTS, axis_angle_to_rot6d, eef10_to_osc7, osc7_to_eef10, rot6d_to_axis_angle,
)
from w0.datasets.libero_posttrain import (
    DOMAIN, RECIPE, LiberoPosttrainProcessor, _representation, file_local_offsets,
    suite_allowed,
)


def processor(tmp_path: Path) -> LiberoPosttrainProcessor:
    stats = {"recipe": RECIPE}
    for name in ("action", "proprio"):
        q01 = [-2.0] * EEF10_DIM
        q99 = [2.0] * EEF10_DIM
        for dim in (3, 4, 5, 6, 7, 8):
            q01[dim], q99[dim] = -1.0, 1.0
        stats[name] = dict(q01=q01, q99=q99, minmax_bounds={})
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    return LiberoPosttrainProcessor(str(path))


def test_suite_allowlist_keeps_goal_and_long_only():
    goal = "libero_goal_no_noops_lerobot"
    long = "libero_10_no_noops_lerobot"
    spatial = "libero_spatial_no_noops_lerobot"
    obj = "libero_object_no_noops_lerobot"
    allow = ["libero_goal", "libero_10"]
    assert suite_allowed(goal, allow) and suite_allowed(long, ["goal", "long"])
    assert not suite_allowed(spatial, allow) and not suite_allowed(obj, allow)
    assert suite_allowed(spatial, None)


def test_left_arm_eef_slots_are_7_to_16():
    assert SLOTS == list(range(7, 17))
    assert DOMAIN["action_valid_indices"] == SLOTS
    assert DOMAIN["name"] == "libero/franka"


def test_osc7_eef10_roundtrip_maps_binary_gripper_to_env():
    open7 = np.array([0.1, -0.2, 0.05, 0.0, 0.0, 0.3, 1.0], dtype=np.float32)
    closed7 = np.array([0.1, -0.2, 0.05, 0.0, 0.0, 0.3, 0.0], dtype=np.float32)
    opened = osc7_to_eef10(open7)
    closed = osc7_to_eef10(closed7)
    np.testing.assert_allclose(opened[:3], open7[:3])
    np.testing.assert_allclose(opened[9], 1.0)
    np.testing.assert_allclose(closed[9], -1.0)
    env_open = eef10_to_osc7(opened, clip=False)
    env_closed = eef10_to_osc7(closed, clip=False)
    np.testing.assert_allclose(env_open[:3], open7[:3], atol=1e-5)
    np.testing.assert_allclose(env_open[3:6], open7[3:6], atol=2e-5)
    np.testing.assert_allclose(env_open[6], -1.0, atol=1e-5)
    np.testing.assert_allclose(env_closed[6], 1.0, atol=1e-5)


def test_identity_rot6d_survives_axis_angle_zero():
    rot6d = axis_angle_to_rot6d(np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(rot6d, np.array([1, 0, 0, 0, 1, 0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(rot6d_to_axis_angle(rot6d), np.zeros(3), atol=1e-6)


def test_native_slot_normalization_keeps_rot6d_and_masks_empty_slots(tmp_path):
    p = processor(tmp_path)
    raw = torch.zeros(3, 10)
    raw[:, :3] = torch.tensor([[-1.0, 0.0, 1.0], [0.5, -0.5, 0.0], [0.0, 0.0, 0.0]])
    raw[:, 3:9] = torch.tensor([0.998, 0.02, -0.04, 0.01, 0.999, 0.02])
    raw[:, 9] = torch.tensor([-0.5, 0.25, 0.0])
    raw[2] = float("nan")
    valid = torch.tensor([True, True, False])[:, None]
    packed, mask = p.pack(raw, "action", valid)
    assert packed.shape == mask.shape == (3, 37)
    assert mask.sum().item() == 20
    assert not packed[~mask].any()
    assert not mask[:, :7].any() and not mask[:, 17:].any()
    torch.testing.assert_close(packed[0, 7:10], torch.tensor([-0.5, 0.0, 0.5]))
    torch.testing.assert_close(packed[0, 10:16], raw[0, 3:9])
    restored = p.unpack_action(packed)
    torch.testing.assert_close(restored[:2], raw[:2])


def test_short_offset_training_masks_unused_slots(tmp_path):
    from unittest.mock import patch
    from test_dino_observations import FakeDino
    from test_future_video import make_model

    torch.set_num_threads(1)
    model = make_model(horizon=96, action_dim=37, proprio_dim=37,
                       action_kwargs={"domain_configs": [DOMAIN], "num_soft_prompts": 8})
    model.max_action_offset = 15
    model.configure_future_video(video_plan_horizon=96, video_action_stride=8,
                                 future_training_mode="idm", video_plan_steps=1)
    with patch("w0.models.wan22.dino_encoder.DinoVideoEncoder", FakeDino):
        model.configure_chunk_obs_context(
            observation_encoder="dino", dino_config={"weights_path": "test"},
            action_obs_downsample_factor=1, chunk_kv_editor_num_queries=3)
    action_mask = torch.zeros(1, 79, 37, dtype=torch.bool)
    action_mask[:, :9, SLOTS] = True
    sample = dict(
        video=torch.randn(1, 3, 13, 16, 16), chunk_obs_images=torch.randn(1, 4, 3, 16, 16),
        action=torch.randn(1, 79, 37).masked_fill(~action_mask, 0),
        proprio=torch.randn(1, 79, 37).masked_fill(~action_mask, 0),
        action_valid_mask=action_mask, proprio_valid_mask=action_mask,
        action_is_pad=~action_mask.any(-1), proprio_is_pad=~action_mask.any(-1),
        image_is_pad=torch.tensor([[False, False] + [True] * 11]),
        context=torch.randn(1, 3, 32), context_mask=torch.ones(1, 3, dtype=torch.bool),
        action_video_freq_ratio=torch.tensor([8]), action_offset=torch.tensor([3]),
        domain_id=torch.tensor([0]),
    )
    loss, _ = model.training_loss(sample)
    assert torch.isfinite(loss)
    loss.backward()
    invalid = [i for i in range(37) if i not in SLOTS]
    assert model.action_expert.head.weight.grad[:, SLOTS].abs().sum() > 0
    assert not model.action_expert.head.weight.grad[:, invalid].any()
    assert not model.action_expert.action_encoder.weight.grad[..., invalid].any()
    assert not model.proprio_encoder.weight.grad[..., invalid].any()


def test_file_local_offsets_pack_two_episodes_in_one_shard():
    offsets = file_local_offsets(
        chunk=[0, 0, 1], file_index=[0, 0, 0], length=[4, 6, 3], from_index=[0, 4, 10])
    np.testing.assert_array_equal(offsets, [0, 4, 0])


def test_ten_d_without_conversion_json_is_rejected(tmp_path):
    info = {"features": {"action": {"shape": [10]}}}
    try:
        _representation(tmp_path, info)
    except ValueError as exc:
        assert "conversion.json" in str(exc)
    else:
        raise AssertionError("10-D LIBERO without conversion.json must be rejected")


def test_w0_camera_keys_match_observation_images():
    from w0.datasets.libero_posttrain import _camera_keys
    keys = _camera_keys({"features": {
        "observation.images.image": {},
        "observation.images.wrist_image": {},
        "action": {"shape": [7]},
    }})
    assert keys == ("observation.images.image", "observation.images.wrist_image")
