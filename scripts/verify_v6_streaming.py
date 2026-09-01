"""验证固定前瞻模型的分块与整段推理一致性。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from config import CFG
from dualpath_v3 import DualPathV3
from phase_aware_v6 import PhaseAwareCausalV6
from train_phase_v6 import stream_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "aec_realtime.pt")
    parser.add_argument("--lookahead-frames", type=int, default=4)
    parser.add_argument("--chunk-frames", type=int, default=4)
    parser.add_argument("--incremental-refiner", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "streaming_verify.json")
    args = parser.parse_args()

    base = DualPathV3(n_freq=CFG.n_freq, hidden=96, num_layers=2,
                      mask_mag_max=3.0)
    base.load_state_dict(torch.load(
        ROOT / "checkpoints" / "base_gru.pt", map_location="cpu",
        weights_only=False)["sd"])
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PhaseAwareCausalV6(
        base, channels=64, lookahead_frames=args.lookahead_frames,
        scene_gate_enabled=bool(saved.get("scene_gate_enabled", False)),
        scene_hidden=int(saved.get("scene_hidden", 48)),
        scene_threshold=float(saved.get("scene_threshold", 0.985)),
        scene_low_threshold=float(saved.get("scene_low_threshold", 0.4)),
        scene_confirm_frames=int(saved.get("scene_confirm_frames", 8)),
        scene_off_threshold=float(saved.get("scene_off_threshold", 0.01)),
        scene_release_frames=int(saved.get("scene_release_frames", 16)),
        scene_soft_gate=bool(saved.get("scene_soft_gate", False)),
        scene_soft_power=float(saved.get("scene_soft_power", 1.0)),
        scene_soft_floor=float(saved.get("scene_soft_floor", 0.0))).eval()
    model.load_state_dict(saved["state_dict"])

    torch.manual_seed(260828)
    E = torch.randn(1, 73, CFG.n_freq, dtype=torch.complex64)
    X = torch.randn_like(E)
    Y = torch.randn_like(E)
    with torch.no_grad():
        full = model(E, X, Y, E + Y)
        streamed, stream_gate, _ = stream_model(
            model, E, X, Y, args.chunk_frames,
            incremental_refiner=args.incremental_refiner)
    full_mask = torch.complex(full[0], full[1])
    result = {
        "checkpoint_step": int(saved["step"]),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest().upper(),
        "frames": E.shape[1],
        "output_frames": streamed.shape[1],
        "lookahead_frames": args.lookahead_frames,
        "chunk_frames": args.chunk_frames,
        "history_frames": model.history_frames,
        "incremental_refiner": args.incremental_refiner,
        "scene_gate_enabled": model.scene_gate_enabled,
        "scene_soft_gate": model.scene_soft_gate,
        "scene_soft_floor": model.scene_soft_floor,
        "max_mask_abs_error": float((streamed - full_mask).abs().max()),
        "max_gate_abs_error": float((stream_gate - full[3]).abs().max()),
        "finite": bool(torch.isfinite(streamed.real).all() and
                       torch.isfinite(streamed.imag).all()),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
