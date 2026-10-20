#!/usr/bin/env python3
"""HTTP policy server for LIBERO closed-loop eval (w0 process)."""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

from experiments.libero.w0_policy.deploy_policy import get_model
from experiments.libero.libero_utils import decode_image_b64

DEFAULT_CKPT = os.environ.get("W0_LIBERO_CHECKPOINT", "")


def _obs(payload: dict) -> dict:
    return {
        "agentview_image": decode_image_b64(payload["agentview_image"]),
        "robot0_eye_in_hand_image": decode_image_b64(payload["robot0_eye_in_hand_image"]),
        "robot0_eef_pos": np.asarray(payload["robot0_eef_pos"], dtype=np.float32),
        "robot0_eef_quat": np.asarray(payload["robot0_eef_quat"], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray(payload["robot0_gripper_qpos"], dtype=np.float32),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, required=not bool(DEFAULT_CKPT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute-n-actions", type=int, default=None,
                        help="Execute only the first N predicted actions, then replan.")
    parser.add_argument("--action-chunk-size", type=int, default=None,
                        help="Model chunk contract; default reads behavior_config from the checkpoint.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fixed diffusion seed; omitted uses the config seed (42).")
    parser.add_argument("--stochastic", action="store_true",
                        help="Disable the config's deterministic diffusion seed (ablation only).")
    args = parser.parse_args()
    model_args = {
        "ckpt": args.ckpt, "device": args.device,
        "execute_n_actions": args.execute_n_actions,
        "action_chunk_size": args.action_chunk_size,
    }
    if args.stochastic:
        model_args["seed"] = None
    elif args.seed is not None:
        model_args["seed"] = args.seed
    policy = get_model(model_args)
    print(f"LIBERO_POLICY_READY ckpt={args.ckpt} http://{args.host}:{args.port}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"ok": True})
                return
            self._json(404, {"ok": False})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/reset":
                policy.reset()
                self._json(200, {"ok": True})
                return
            if self.path == "/act":
                action = policy.act(_obs(payload), str(payload["instruction"]))
                self._json(200, {"action": np.asarray(action, dtype=np.float32).tolist()})
                return
            self._json(404, {"ok": False})

        def log_message(self, format, *args_):
            return

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
