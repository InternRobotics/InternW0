"""LIBERO native EEF10 helpers for the w0 37D left-arm slots.

The 37D schema stores left-arm EEF as xyz (7:10), column-major rot6d (10:16)
and a scalar gripper (16). LIBERO's converted on-disk contract is the matching
10-D vector:

    [delta_xyz3, rot6d(Exp(delta_axis_angle3)), gripper_open_scale]

Gripper open-scale is ``-1 = closed, +1 = open``. W0/OpenVLA 7-D files
store binary ``{0,1}`` with ``0=close, 1=open``; convert with ``2g-1``.
Robosuite runtime uses the opposite sign (``-1=open, +1=close``), so
evaluation negates the open-scale channel only.
"""
from __future__ import annotations

import numpy as np

EEF10_DIM = 10
ACTION7_DIM = 7
ROT6D_DIMS = (3, 4, 5, 6, 7, 8)
GRIPPER_DIM = 9
GRIPPER_WIDTH_OPEN = 0.08
GRIPPER_CONVENTION = "minus1_closed_plus1_open"
GRIPPER_SOURCE_OSC7 = "binary01_open"
OUTPUT_REPRESENTATION = "native_delta_eef10"
EEF10_NAMES = [
    "eef_x", "eef_y", "eef_z",
    "rot6d_col0_x", "rot6d_col0_y", "rot6d_col0_z",
    "rot6d_col1_x", "rot6d_col1_y", "rot6d_col1_z",
    "gripper_open_scale",
]
# Left-arm EEF block in the 37D canonical layout.
SLOTS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    value = np.asarray(axis_angle, dtype=np.float64)
    if value.shape[-1] != 3:
        raise ValueError(f"axis-angle must end with 3, got {value.shape}")
    angle = np.linalg.norm(value, axis=-1, keepdims=True)
    small = angle[..., 0] < 1e-8
    axis = np.where(angle > 1e-8, value / np.maximum(angle, 1e-8), 0.0)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    cosine, sine = np.cos(angle[..., 0]), np.sin(angle[..., 0])
    one = 1.0 - cosine
    matrix = np.empty(value.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = cosine + x * x * one
    matrix[..., 0, 1] = x * y * one - z * sine
    matrix[..., 0, 2] = x * z * one + y * sine
    matrix[..., 1, 0] = y * x * one + z * sine
    matrix[..., 1, 1] = cosine + y * y * one
    matrix[..., 1, 2] = y * z * one - x * sine
    matrix[..., 2, 0] = z * x * one - y * sine
    matrix[..., 2, 1] = z * y * one + x * sine
    matrix[..., 2, 2] = cosine + z * z * one
    matrix[small] = np.eye(3)
    return matrix


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Column-major rot6d: first two columns of R, matching the 37D schema."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1).astype(np.float32)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    value = np.asarray(rot6d, dtype=np.float64).reshape(-1)
    if value.shape != (6,):
        raise ValueError(f"rot6d must have 6 elements, got {value.shape}")
    col0 = value[:3]
    col0 = col0 / max(float(np.linalg.norm(col0)), 1e-8)
    col1 = value[3:6] - float(np.dot(col0, value[3:6])) * col0
    col1 = col1 / max(float(np.linalg.norm(col1)), 1e-8)
    col2 = np.cross(col0, col1)
    return np.stack([col0, col1, col2], axis=1)


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    r = np.asarray(matrix, dtype=np.float64)
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]], dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    return (axis * angle).astype(np.float32)


def axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
    return matrix_to_rot6d(axis_angle_to_matrix(axis_angle))


def rot6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    return matrix_to_axis_angle(rot6d_to_matrix(rot6d))


def quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.shape != (4,):
        raise ValueError(f"quaternion must have 4 elements, got {q.shape}")
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    x, y, z, w = q
    col0 = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    col1 = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
    return np.concatenate([col0, col1]).astype(np.float32)


def gripper_qpos_to_open_scale(qpos: np.ndarray) -> float:
    fingers = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if fingers.size == 1:
        width = float(fingers[0])
    elif fingers.size >= 2:
        width = float(fingers[0] - fingers[1])
    else:
        raise ValueError("gripper qpos is empty")
    return float(np.clip(2.0 * width / GRIPPER_WIDTH_OPEN - 1.0, -1.0, 1.0))


def open_scale_to_env_gripper(value: float) -> float:
    return float(np.clip(-float(value), -1.0, 1.0))


def osc7_to_eef10(action7: np.ndarray, *, gripper_source: str = GRIPPER_SOURCE_OSC7) -> np.ndarray:
    """Convert LIBERO 7-D OSC to native EEF10.

    ``gripper_source="binary01_open"`` is the W0 parquet contract (0=close, 1=open).
    ``gripper_source="env_close"`` is robosuite runtime (±1, +1=close).
    """
    value = np.asarray(action7, dtype=np.float32).reshape(-1)
    if value.shape != (ACTION7_DIM,):
        raise ValueError(f"expected 7-D OSC action, got {value.shape}")
    if gripper_source == "binary01_open":
        gripper = float(np.clip(2.0 * value[6] - 1.0, -1.0, 1.0))
    elif gripper_source == "env_close":
        gripper = float(-value[6])
    else:
        raise ValueError(f"Unknown gripper_source: {gripper_source}")
    return np.concatenate([value[:3], axis_angle_to_rot6d(value[3:6]), [gripper]]).astype(np.float32)


def eef10_to_osc7(action10: np.ndarray, *, clip: bool = True) -> np.ndarray:
    """Convert native EEF10 to LIBERO runtime 7-D OSC."""
    value = np.asarray(action10, dtype=np.float32).reshape(-1)
    if value.shape != (EEF10_DIM,):
        raise ValueError(f"expected EEF10 action, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("EEF10 action contains NaN or infinity")
    position = value[:3].copy()
    rotation = rot6d_to_axis_angle(value[3:9])
    gripper = np.array([open_scale_to_env_gripper(value[9])], dtype=np.float32)
    if clip:
        position = np.clip(position, -1.0, 1.0)
        rotation = np.clip(rotation, -1.0, 1.0)
        gripper = np.clip(gripper, -1.0, 1.0)
    return np.concatenate([position, rotation, gripper]).astype(np.float32)


def obs_to_eef10(eef_pos, eef_quat_xyzw, gripper_qpos) -> np.ndarray:
    pos = np.asarray(eef_pos, dtype=np.float32).reshape(-1)
    quat = np.asarray(eef_quat_xyzw, dtype=np.float32).reshape(-1)
    if pos.shape != (3,) or quat.shape != (4,):
        raise ValueError(f"expected eef_pos 3 and quat 4, got {pos.shape} and {quat.shape}")
    grip = np.array([gripper_qpos_to_open_scale(gripper_qpos)], dtype=np.float32)
    return np.concatenate([pos, quat_xyzw_to_rot6d(quat), grip]).astype(np.float32)


def state8_to_eef10(state8: np.ndarray) -> np.ndarray:
    """W0-style 8-D state: xyz3 + axis-angle3 + two finger joints."""
    value = np.asarray(state8, dtype=np.float32).reshape(-1)
    if value.shape != (8,):
        raise ValueError(f"expected 8-D state, got {value.shape}")
    grip = np.array([gripper_qpos_to_open_scale(value[6:8])], dtype=np.float32)
    return np.concatenate([value[:3], axis_angle_to_rot6d(value[3:6]), grip]).astype(np.float32)
