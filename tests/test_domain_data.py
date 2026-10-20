"""Native commands, temporal alignment, camera padding and reproducible windows."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from w0.datasets.domain_registry import registry_fingerprint, episode_is_validation
from w0.datasets.domain_video_dataset import DomainVideoDataset, decode_native_frames
from w0.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


class DomainDataTests(unittest.TestCase):
    def test_native_actions_timestamps_and_epoch_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = dict(domain="test/arm", root=str(root), fps=10, chunks_size=1000,
                          data_path="episode.parquet", video_path="video.mp4", videos=["main", None, None])
            config = dict(name="test/arm", dataset="test", embodiment="arm", action_dim=2, proprio_dim=3,
                          action_fields=[dict(key="command", dim=2)], proprio_fields=[dict(key="state", dim=3)])
            registry = dict(version=1, domain_configs=[config], sources=[source])
            (root / "registry.json").write_text(json.dumps(registry))
            ep = next(i for i in range(100) if not episode_is_validation(source, i))
            (root / "meta").mkdir()
            (root / "meta/episodes.jsonl").write_text(json.dumps(dict(episode_index=ep, length=40, tasks=["move"])))
            pq.write_table(pa.table({"command": [[10. + i, 20. + 2 * i] for i in range(40)],
                                     "state": [[-100. - i, 99., 42.] for i in range(40)]}), root / "episode.parquet")
            with av.open(str(root / "video.mp4"), "w") as container:
                stream = container.add_stream("libx264", rate=10)
                stream.width = stream.height = 32
                stream.pix_fmt = "yuv420p"
                for i in range(40):
                    frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 4, dtype=np.uint8), format="rgb24")
                    for packet in stream.encode(frame): container.mux(packet)
                for packet in stream.encode(): container.mux(packet)
            frames = decode_native_frames(root / "video.mp4", torch.tensor([7, 15, 15, 30]), 10)
            torch.testing.assert_close(frames[1], frames[2])
            torch.testing.assert_close(frames.float().mean((1, 2, 3)), torch.tensor([28., 60., 60., 120.]), atol=3, rtol=0)
            stats = dict(registry_fingerprint=registry_fingerprint(registry), split_seed=42, val_set_proportion=.05,
                         domains={"test/arm": {"action": dict(mean=[0.] * 2, std=[1.] * 2),
                                                "proprio": dict(mean=[0.] * 3, std=[1.] * 3)}})
            (root / "stats.json").write_text(json.dumps(stats))
            hashed = hashlib.sha256(DEFAULT_PROMPT.format(task="move").encode()).hexdigest()
            torch.save(dict(context=torch.ones(128, 32), mask=torch.ones(128, dtype=torch.bool)),
                       root / f"{hashed}.t5_len128.test.pt")
            dataset = DomainVideoDataset(root / "registry.json", root / "stats.json", root,
                                         text_encoder_id="test", video_size=(48, 32), samples_per_epoch=50)
            a, b = dataset[0], dataset[0]
            torch.testing.assert_close(a["action"], b["action"], atol=0, rtol=0)
            torch.testing.assert_close(a["proprio"][:-1, 0], -90 - a["action"][:, 0])
            torch.testing.assert_close(a["action"][1:, 0] - a["action"][:-1, 0], torch.full((63,), .2), atol=5e-6, rtol=0)
            self.assertTrue((a["video"][:, :, 32:] == -1).all())
            self.assertFalse(a["action_is_pad"].any())
            self.assertTrue(any(not torch.equal(dataset[(i, 0)]["action"], dataset[(i, 1)]["action"]) for i in range(3)))
            stats["split_seed"] = 99
            (root / "stats.json").write_text(json.dumps(stats))
            with self.assertRaisesRegex(ValueError, "split"):
                DomainVideoDataset(root / "registry.json", root / "stats.json", root)


if __name__ == "__main__":
    unittest.main()
