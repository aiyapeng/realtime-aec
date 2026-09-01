"""Deployment entry point for the V8 exact-cache CPU real-time AEC path."""
from __future__ import annotations

from pathlib import Path
import sys
import warnings

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from aec.runtime import AECPipeline
from config import CFG
from dualpath_v3 import DualPathV3
from phase_aware_v6 import PhaseAwareCausalV6


def configure_cpu(threads: int = 4, interop_threads: int = 1):
    """Configure the validated CPU thread topology before model inference."""
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(int(interop_threads))
    except RuntimeError:
        if torch.get_num_interop_threads() != int(interop_threads):
            warnings.warn(
                "PyTorch inter-op threads were already initialized; start a fresh "
                "process and call configure_cpu() before inference for validated RTF.",
                RuntimeWarning,
            )
    return {
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }


def load_model(
    checkpoint: Path = ROOT / "checkpoints" / "aec_realtime.pt",
    base_checkpoint: Path = ROOT / "checkpoints" / "base_gru.pt",
):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = DualPathV3(n_freq=CFG.n_freq, hidden=96, num_layers=2,
                      mask_mag_max=3.0)
    base.load_state_dict(torch.load(
        base_checkpoint, map_location="cpu", weights_only=False)["sd"])
    model = PhaseAwareCausalV6(
        base, channels=64, fs=CFG.fs, hop=CFG.hop,
        lookahead_frames=int(saved.get("lookahead_frames", 4)),
        scene_gate_enabled=bool(saved.get("scene_gate_enabled", False)),
        scene_hidden=int(saved.get("scene_hidden", 48)),
        scene_threshold=float(saved.get("scene_threshold", 0.985)),
        scene_low_threshold=float(saved.get("scene_low_threshold", 0.4)),
        scene_confirm_frames=int(saved.get("scene_confirm_frames", 8)),
        scene_off_threshold=float(saved.get("scene_off_threshold", 0.01)),
        scene_release_frames=int(saved.get("scene_release_frames", 16)),
        scene_soft_gate=bool(saved.get("scene_soft_gate", False)),
        scene_soft_power=float(saved.get("scene_soft_power", 1.0)),
        scene_soft_floor=float(saved.get("scene_soft_floor", 0.0)),
    )
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def create_pipeline(model=None, *, threads: int = 4,
                    interop_threads: int = 1):
    """Create the validated 16-kHz, 72-ms, chunk=4 CPU pipeline."""
    configure_cpu(threads, interop_threads)
    if model is None:
        model = load_model()
    return AECPipeline(
        model=model, fs=CFG.fs, block=CFG.hop, n_fft=CFG.n_fft,
        hop=CFG.hop, num_partitions=CFG.mdf_partitions, mu=CFG.mdf_mu,
        dtd_mu_floor=CFG.dtd_mu_floor, device="cpu", stream_chunk_frames=4,
        incremental_refiner=True,
    )


__all__ = ["configure_cpu", "load_model", "create_pipeline"]
