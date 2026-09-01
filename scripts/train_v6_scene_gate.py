"""Train only the streaming ST/NST/DT scene gate for frozen V6 acoustics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from config import CFG
from dualpath_v3 import DualPathV3
from phase_aware_v6 import PhaseAwareCausalV6
from train_phase_v4 import stft, window


def new_base(path: Path):
    base = DualPathV3(n_freq=CFG.n_freq, hidden=96, num_layers=2,
                      mask_mag_max=3.0)
    base.load_state_dict(torch.load(path, map_location="cpu",
                                    weights_only=False)["sd"])
    return base


def load_model(args, device):
    source = torch.load(args.acoustic_checkpoint, map_location="cpu",
                        weights_only=False)
    model = PhaseAwareCausalV6(
        new_base(args.base_checkpoint), channels=64, fs=CFG.fs, hop=CFG.hop,
        lookahead_frames=args.lookahead_frames, scene_gate_enabled=True,
        scene_hidden=args.scene_hidden, scene_threshold=args.scene_threshold)
    expected = {"scene_norm.weight", "scene_norm.bias",
                "scene_gru.weight_ih_l0", "scene_gru.weight_hh_l0",
                "scene_gru.bias_ih_l0", "scene_gru.bias_hh_l0",
                "scene_gru.weight_ih_l1", "scene_gru.weight_hh_l1",
                "scene_gru.bias_ih_l1", "scene_gru.bias_hh_l1",
                "scene_output.weight", "scene_output.bias"}
    source_state = source["state_dict"]
    if args.reinit_scene:
        source_state = {key: value for key, value in source_state.items()
                        if not key.startswith("scene_")}
    incompatible = model.load_state_dict(source_state, strict=False)
    expected_missing = expected if args.reinit_scene else set()
    if (set(incompatible.missing_keys) != expected_missing or
            incompatible.unexpected_keys):
        raise RuntimeError(f"unexpected checkpoint mismatch: {incompatible}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.scene_norm, model.scene_gru, model.scene_output):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return model.to(device), int(source.get("step", 0))


def shifted_accuracy(logits, targets, lookahead):
    frames = logits.shape[1]
    indices = (torch.arange(frames, device=logits.device) + lookahead).clamp(
        max=frames - 1)
    predicted = logits.index_select(1, indices).sigmoid() >= 0.5
    return (predicted == targets.bool()).float().mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic-checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "aec_realtime.pt")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "scene_gate.pt")
    parser.add_argument("--base-checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "base_gru.pt")
    parser.add_argument("--train-bank-dir", type=Path, required=True)
    parser.add_argument("--extra-bank-dir", type=Path, default=None,
                        help="optional hard-example bank, sampled 50/50 with the main bank")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lookahead-frames", type=int, default=4)
    parser.add_argument("--scene-hidden", type=int, default=48)
    parser.add_argument("--scene-threshold", type=float, default=0.985)
    parser.add_argument("--reinit-scene", action="store_true",
                        help="discard any source scene head and train a fresh LA-specific gate")
    parser.add_argument("--hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=260829)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, acoustic_step = load_model(args, device)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=1e-4)
    bank = np.load(args.train_bank_dir / "bank.npy", mmap_mode="r")
    labels = np.load(args.train_bank_dir / "scenario.npy")
    extra_bank = (np.load(args.extra_bank_dir / "bank.npy", mmap_mode="r")
                  if args.extra_bank_dir else None)
    extra_labels = (np.load(args.extra_bank_dir / "scenario.npy")
                    if args.extra_bank_dir else None)
    win = window(device)
    crop = int(args.segment_seconds * CFG.fs)
    started = time.time()
    for step in range(args.steps):
        rng = np.random.default_rng(args.seed + step * 104729)
        # Equal class sampling avoids a gate biased toward the two negative classes.
        per_class = max(1, args.batch_size // 3)
        selected = []
        for label in (0, 1, 2):
            main_count = per_class if extra_bank is None else per_class // 2
            extra_count = per_class - main_count
            selected.extend((0, int(index)) for index in rng.choice(
                np.flatnonzero(labels == label), size=main_count, replace=True))
            if extra_count:
                selected.extend((1, int(index)) for index in rng.choice(
                    np.flatnonzero(extra_labels == label), size=extra_count,
                    replace=True))
        while len(selected) < args.batch_size:
            selected.append((0, int(rng.integers(len(bank)))))
        rng.shuffle(selected)
        rows = []
        batch_labels = []
        for source, idx in selected:
            source_bank = bank if source == 0 else extra_bank
            source_labels = labels if source == 0 else extra_labels
            segment = np.asarray(source_bank[int(idx)], np.float32) / 32767.0
            begin = int(rng.integers(0, segment.shape[1] - crop + 1))
            # A common gain makes the scene decision insensitive to absolute level.
            rows.append(segment[:, begin:begin + crop] * rng.uniform(0.35, 1.8))
            batch_labels.append(int(source_labels[int(idx)]) == 2)
        batch = torch.from_numpy(np.stack(rows).astype(np.float32)).to(device)
        with torch.no_grad():
            E, X, Y = [stft(batch[:, channel], win) for channel in range(3)]
            _, _, features, _ = model._base_and_features(E, X, Y, E + Y)
        logits, _ = model._scene_logits(features.detach())
        targets = torch.tensor(batch_labels, device=device, dtype=logits.dtype)
        targets = targets[:, None, None].expand_as(logits)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        # Explicitly train the first decision available after bounded lookahead.
        early_end = min(logits.shape[1], args.lookahead_frames + 16)
        early_loss = F.binary_cross_entropy_with_logits(
            logits[:, args.lookahead_frames:early_end],
            targets[:, args.lookahead_frames:early_end])
        negative_rows = ~torch.tensor(batch_labels, device=device, dtype=torch.bool)
        negative_logits = logits[negative_rows, :, 0]
        hard_count = min(24, negative_logits.shape[1])
        hard_negative = F.softplus(
            negative_logits.topk(hard_count, dim=1).values).mean()
        loss = loss + 0.5 * early_loss + args.hard_negative_weight * hard_negative
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0)
        optimizer.step()
        if step % 50 == 0:
            accuracy = shifted_accuracy(logits.detach(), targets,
                                        args.lookahead_frames)
            print(json.dumps({"step": step, "loss": float(loss.detach()),
                              "hard_negative": float(hard_negative.detach()),
                              "accuracy": float(accuracy),
                              "seconds": round(time.time() - started, 1)}), flush=True)
        if (step + 1) % args.save_every == 0:
            torch.save({"state_dict": model.state_dict(), "step": acoustic_step,
                        "scene_step": step + 1, "scene_gate_enabled": True,
                        "scene_hidden": args.scene_hidden,
                        "scene_threshold": args.scene_threshold,
                        "scene_low_threshold": 0.4,
                        "scene_confirm_frames": 8,
                        "scene_off_threshold": 0.01,
                        "scene_release_frames": 16,
                        "lookahead_frames": args.lookahead_frames,
                        "acoustic_checkpoint": str(args.acoustic_checkpoint),
                        "seed": args.seed}, args.checkpoint)
    torch.save({"state_dict": model.state_dict(), "step": acoustic_step,
                "scene_step": args.steps, "scene_gate_enabled": True,
                "scene_hidden": args.scene_hidden,
                "scene_threshold": args.scene_threshold,
                "scene_low_threshold": 0.4,
                "scene_confirm_frames": 8,
                "scene_off_threshold": 0.01,
                "scene_release_frames": 16,
                "lookahead_frames": args.lookahead_frames,
                "acoustic_checkpoint": str(args.acoustic_checkpoint),
                "seed": args.seed}, args.checkpoint)


if __name__ == "__main__":
    main()
