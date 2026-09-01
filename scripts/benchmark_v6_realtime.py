"""Benchmark the fixed-lookahead V6 in the real block-by-block AEC pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from aec.runtime import AECPipeline
from config import CFG
from dualpath_v3 import DualPathV3
from phase_aware_v6 import PhaseAwareCausalV6


def load_model(checkpoint, lookahead, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = DualPathV3(n_freq=CFG.n_freq, hidden=96, num_layers=2,
                      mask_mag_max=3.0)
    base.load_state_dict(torch.load(
        ROOT / "checkpoints" / "base_gru.pt", map_location="cpu",
        weights_only=False)["sd"])
    model = PhaseAwareCausalV6(base, channels=64, fs=CFG.fs, hop=CFG.hop,
                               lookahead_frames=lookahead,
                               scene_gate_enabled=bool(saved.get("scene_gate_enabled", False)),
                               scene_hidden=int(saved.get("scene_hidden", 48)),
                               scene_threshold=float(saved.get("scene_threshold", 0.985)),
                               scene_low_threshold=float(saved.get("scene_low_threshold", 0.4)),
                               scene_confirm_frames=int(saved.get("scene_confirm_frames", 8)),
                               scene_off_threshold=float(saved.get("scene_off_threshold", 0.01)),
                               scene_release_frames=int(saved.get("scene_release_frames", 16)),
                               scene_soft_gate=bool(saved.get("scene_soft_gate", False)),
                               scene_soft_power=float(saved.get("scene_soft_power", 1.0)),
                               scene_soft_floor=float(saved.get("scene_soft_floor", 0.0)))
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "aec_realtime.pt")
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--lookahead-frames", type=int, default=4)
    parser.add_argument("--chunk-frames", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=None,
                        help="CPU intra-op threads (set before model construction)")
    parser.add_argument("--torch-interop-threads", type=int, default=None,
                        help="CPU inter-op threads (set once before model construction)")
    parser.add_argument("--warmup-seconds", type=float, default=0.0)
    parser.add_argument("--incremental-refiner", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "runtime_benchmark.json")
    args = parser.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads is not None:
        torch.set_num_interop_threads(args.torch_interop_threads)

    bank = np.load(args.bank_dir / "bank.npy", mmap_mode="r")
    row = np.asarray(bank[200], np.float32) / 32767.0
    repeats = max(1, int(np.ceil(args.seconds * CFG.fs / row.shape[1])))
    err = np.tile(row[0], repeats)[:int(args.seconds * CFG.fs)]
    ref = np.tile(row[1], repeats)[:len(err)]
    echo = np.tile(row[2], repeats)[:len(err)]
    mic = err + echo

    model = load_model(args.checkpoint, args.lookahead_frames, args.device)
    pipeline = AECPipeline(
        model=model, fs=CFG.fs, block=CFG.hop, n_fft=CFG.n_fft, hop=CFG.hop,
        num_partitions=CFG.mdf_partitions, mu=CFG.mdf_mu,
        dtd_mu_floor=CFG.dtd_mu_floor, device=args.device,
        stream_chunk_frames=args.chunk_frames,
        incremental_refiner=args.incremental_refiner,
    )
    if args.warmup_seconds > 0:
        warm_samples = max(CFG.hop, int(args.warmup_seconds * CFG.fs))
        warm_samples -= warm_samples % CFG.hop
        pipeline.process_stream(ref[:warm_samples], mic[:warm_samples])
    if args.device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = pipeline.process_stream(ref, mic)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    audio_seconds = len(ref) / CFG.fs
    result = {
        "device": args.device,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "audio_seconds": audio_seconds,
        "wall_seconds": elapsed,
        "rtf": elapsed / audio_seconds,
        "lookahead_frames": args.lookahead_frames,
        "stream_chunk_frames": args.chunk_frames,
        "scene_soft_gate": model.scene_soft_gate,
        "scene_soft_floor": model.scene_soft_floor,
        "incremental_refiner": args.incremental_refiner,
        "latency_samples": pipeline.latency_samples,
        "latency_ms": pipeline.latency_samples / CFG.fs * 1000.0,
        "finite": bool(np.isfinite(output).all()),
        "output_rms": float(np.sqrt(np.mean(output ** 2))),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
