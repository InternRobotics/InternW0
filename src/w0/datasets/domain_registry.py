"""Stable dataset/embodiment identity, native action schemas and episode splits."""

import hashlib
import json
import math
from pathlib import Path


def read_registry(path):
    registry = json.loads(Path(path).read_text())
    if registry.get("version") != 1 or not registry.get("domain_configs"):
        raise ValueError("Invalid domain registry version or empty bank.")
    from w0.models.wan22.helpers.domain_bank import validate_domain_configs
    configs = registry["domain_configs"]
    validate_domain_configs(configs, max(d["action_dim"] for d in configs))
    for config in configs:
        for kind, dim_key in (("action_fields", "action_dim"), ("proprio_fields", "proprio_dim")):
            fields = config[kind]
            if not fields or any(not isinstance(f["key"], str) or not f["key"] or not isinstance(f["dim"], int) or f["dim"] <= 0 for f in fields):
                raise ValueError(f"Invalid native fields in {config['name']}")
            if sum(f["dim"] for f in fields) != config[dim_key] or len({f["key"] for f in fields}) != len(fields):
                raise ValueError(f"Native field dimensions/keys disagree in {config['name']}")
    names = {d["name"] for d in configs}
    if any(s["domain"] not in names for s in registry.get("sources", [])):
        raise ValueError("Registry source refers to an unknown domain.")
    for source in registry.get("sources", []):
        if not math.isfinite(source["fps"]) or source["fps"] <= 0 or source["chunks_size"] < 1:
            raise ValueError("Invalid source FPS or chunk size.")
        if len(source["videos"]) != 3 or not source["videos"][0]:
            raise ValueError("Sources require three camera slots and a main camera.")
    return registry


def registry_fingerprint(registry):
    return hashlib.sha256(json.dumps(registry, sort_keys=True).encode()).hexdigest()


def episode_is_validation(source, episode_index, fraction=0.05, seed=42):
    # RoboMIND's 1rgb/3rgb exports can share an original trajectory. Group them.
    root = str(source["root"]).replace("franka_1rgb", "franka").replace("franka_3rgb", "franka")
    root = root.replace("benchmark1_0_compressed", "benchmark").replace("benchmark1_1_compressed", "benchmark")
    value = int(hashlib.sha256(f"{seed}:{root}:{episode_index}".encode()).hexdigest()[:16], 16)
    return value / 2**64 < fraction


def source_schema(info_path, dataset_root, schema_overrides=None):
    info_path, dataset_root = Path(info_path), Path(dataset_root)
    parts = info_path.relative_to(dataset_root).parts
    if len(parts) < 5:
        raise ValueError(f"Expected task_family/embodiment/task/meta/info.json: {info_path}")
    dataset, embodiment = dataset_root.name, parts[1]
    info = json.loads(info_path.read_text())
    features = info["features"]
    override = (schema_overrides or {}).get(f"{dataset}/{embodiment}")
    if override:
        action_keys, state_keys = override["action_keys"], override["proprio_keys"]
        video = override["video_keys"]
        if not action_keys or not state_keys or len(video) != 3:
            raise ValueError("Schema overrides require action/proprio keys and three camera slots.")
    elif dataset == "InternData-A1":
        action_keys = (["actions.left_joint.position", "actions.left_gripper.position",
                        "actions.right_joint.position", "actions.right_gripper.position"]
                       if "actions.left_joint.position" in features else
                       ["actions.joint.position", "actions.gripper.position"])
        state_keys = [k.replace("actions.", "states.", 1) for k in action_keys]
        video = ["images.rgb.head", "images.rgb.hand_left", "images.rgb.hand_right"]
        if "images.rgb.hand" in features:
            video[1] = "images.rgb.hand"
    elif dataset == "RoboMIND2":
        action_keys = (["actions.joint_position_left", "actions.joint_position_right"]
                       if "actions.joint_position_left" in features else ["actions.joint_position"])
        if embodiment == "tienkung_xsens_1rgb":
            # Despite the column name, these 12 values are finger joints, not an EEF pose.
            names = str(features["actions.end_effector"].get("names", ""))
            if "finger" not in names:
                raise ValueError(f"Unrecognized xsens hand schema: {info_path}")
            action_keys.append("actions.end_effector")
        state_keys = [k.replace("actions.", "observation.states.", 1) for k in action_keys]
        video = ["observation.images.camera_top", "observation.images.camera_left", "observation.images.camera_right"]
        if embodiment == "franka_fr3_dual":
            video[0] = "observation.images.camera_front"
        if embodiment == "franka_1rgb" and "observation.images.camera_top" not in features:
            video = ["observation.images.camera_right", None, None]
        if embodiment == "agilex_3rgb":
            video = ["observation.images.camera_front", "observation.images.camera_left_wrist", "observation.images.camera_right_wrist"]
    else:
        raise ValueError(f"No native schema rule for {dataset}; register its explicit field mapping.")
    def fields(keys):
        result = []
        for key in keys:
            feature = features[key]
            if len(feature["shape"]) != 1:
                raise ValueError(f"Expected vector feature {key} at {info_path}")
            result.append({"key": key, "dim": feature["shape"][0], "names": feature.get("names")})
        return result
    actions, states = fields(action_keys), fields(state_keys)
    config = {"name": f"{dataset}/{embodiment}", "dataset": dataset, "embodiment": embodiment,
              "action_dim": sum(f["dim"] for f in actions), "proprio_dim": sum(f["dim"] for f in states),
              "action_fields": actions, "proprio_fields": states, "action_semantics": "native_joint_command"}
    source = {"domain": config["name"], "root": str(info_path.parent.parent), "fps": info["fps"],
              "videos": [v if v in features else None for v in video],
              "data_path": info["data_path"], "video_path": info["video_path"],
              "chunks_size": info.get("chunks_size", 1000), "reported_robot_type": info.get("robot_type")}
    if source["videos"][0] is None:
        raise ValueError(f"Missing main view at {info_path}")
    return config, source
