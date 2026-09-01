"""Train/evaluate the phase-aware V4 DT refiner on locked banks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from config import CFG
from dualpath_v3 import DualPathV3
from phase_aware_v4 import PhaseAwareRefinerV4, clipped_oracle_mask
from scene_loss import _compress, near_dominance_target
from optimize_dt_postfilter import CachedAECMOS


def window(device):
    return torch.from_numpy(
        np.sqrt(np.hanning(CFG.n_fft + 1)[:-1]).astype(np.float32)
    ).to(device)


def stft(x, win):
    return torch.stft(
        x, CFG.n_fft, CFG.hop, CFG.n_fft, win,
        center=True, return_complex=True,
    ).transpose(1, 2)


def istft(x, win, length):
    return torch.istft(
        x.transpose(1, 2), CFG.n_fft, CFG.hop, CFG.n_fft, win,
        center=True, length=length,
    )


def load_model(base_checkpoint: Path, device, checkpoint: Path | None = None):
    base = DualPathV3(n_freq=CFG.n_freq, hidden=96, num_layers=2, mask_mag_max=3.0)
    base_data = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    base.load_state_dict(base_data["sd"])
    model = PhaseAwareRefinerV4(base, hidden=192, num_layers=2)
    step = 0
    optimizer_state = None
    if checkpoint and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"])
        step = int(saved["step"])
        optimizer_state = saved.get("optimizer")
    return model.to(device), step, optimizer_state


def loss_v4(mask, p_ne, E, S):
    target = clipped_oracle_mask(E, S)
    residual = E - S
    # Direct complex-mask regression is weighted by bins carrying either desired
    # speech or residual echo; silent bins cannot dominate the objective.
    weight = (S.abs() + residual.abs() + 1e-4).pow(0.3).detach()
    mask_mse = (weight * (mask - target).abs().square()).sum() / (weight.sum() + 1e-8)

    enhanced = E * mask
    cm_e, cc_e = _compress(enhanced, 0.3)
    cm_s, cc_s = _compress(S, 0.3)
    recon = 0.5 * (cm_e - cm_s).square().mean() + 0.5 * (
        cc_e - cc_s
    ).abs().square().mean()

    target_phase = torch.angle(target)
    phase_weight = S.abs().pow(0.6).detach()
    phase = (
        phase_weight * (1.0 - torch.cos(torch.angle(mask) - target_phase))
    ).sum() / (phase_weight.sum() + 1e-8)

    dominance = near_dominance_target(E, S).detach()
    evidence = F.binary_cross_entropy(
        p_ne.clamp(1e-6, 1 - 1e-6), dominance
    )
    energy = S.abs().pow(0.6).detach()
    calibration = (
        energy * (p_ne - dominance).square()
    ).sum() / (energy.sum() + 1e-8)
    total = 2.0 * mask_mse + 2.0 * recon + 0.75 * phase + 0.25 * evidence + calibration
    return total, {
        "mask": float(mask_mse.detach()), "recon": float(recon.detach()),
        "phase": float(phase.detach()), "evidence": float(evidence.detach()),
        "cal": float(calibration.detach()),
    }


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    model, start, optimizer_state = load_model(
        args.base_checkpoint, device, args.checkpoint if args.resume else None
    )
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    bank = np.load(ROOT / "data" / "bank_train" / "bank.npy", mmap_mode="r")
    labels = np.load(ROOT / "data" / "bank_train" / "scenario.npy")
    dt_indices = np.flatnonzero(labels == 2)
    win = window(device)
    crop = int(args.segment_seconds * CFG.fs)
    started = time.time()
    for step in range(start, args.steps):
        rng = np.random.default_rng(args.seed + step * 17)
        chosen = rng.choice(dt_indices, size=args.batch_size, replace=True)
        rows = []
        for idx in chosen:
            segment = np.asarray(bank[int(idx)], np.float32) / 32767.0
            begin = int(rng.integers(0, segment.shape[1] - crop + 1))
            gain = float(rng.uniform(0.65, 1.35))
            rows.append(segment[:, begin:begin + crop] * gain)
        batch = torch.from_numpy(np.stack(rows)).to(device)
        err, ref, echo_hat, near = [batch[:, i] for i in range(4)]
        E, X, Y, S = [stft(x, win) for x in (err, ref, echo_hat, near)]
        D = E + Y
        mr, mi, p_ne, _, _ = model(E, X, Y, D, force_dt=True)
        mask = torch.complex(mr.float(), mi.float())
        loss, parts = loss_v4(mask, p_ne, E, S)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0
        )
        optimizer.step()
        if step % 100 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach()), **parts,
                              "seconds": round(time.time() - started, 1)}), flush=True)
        if (step + 1) % args.save_every == 0:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step + 1, "seed": args.seed,
                        "base_checkpoint": str(args.base_checkpoint)}, args.checkpoint)
    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": args.steps, "seed": args.seed,
                "base_checkpoint": str(args.base_checkpoint)}, args.checkpoint)
    print("saved", args.checkpoint, flush=True)


def evaluate(args):
    device = torch.device(args.device)
    model, step, _ = load_model(args.base_checkpoint, device, args.checkpoint)
    model.eval()
    bank = np.load(ROOT / "data" / "bank_dev" / "bank.npy", mmap_mode="r")
    labels = np.load(ROOT / "data" / "bank_dev" / "scenario.npy")
    win = window(device)
    scorer = CachedAECMOS(
        ROOT / "third_party" / "aecmos" / "Run_1663915512_Stage_0.onnx"
    )
    rows = {s: [] for s in ("st", "nst", "dt")}
    names = ("st", "nst", "dt")
    with torch.no_grad():
        for idx in range(bank.shape[0]):
            segment = np.asarray(bank[idx], np.float32) / 32767.0
            batch = torch.from_numpy(segment).unsqueeze(0).to(device)
            err, ref, echo_hat = [batch[:, i] for i in range(3)]
            E, X, Y = [stft(x, win) for x in (err, ref, echo_hat)]
            mr, mi, _, _, _ = model(E, X, Y, E + Y)
            enhanced = istft(E * torch.complex(mr, mi), win, err.shape[1])
            enhanced = enhanced[0].cpu().numpy()
            scenario = names[int(labels[idx])]
            fixed = scorer.fixed_features(scenario, segment[1], segment[0] + segment[2])
            rows[scenario].append(scorer.score(fixed, enhanced))
    scores = {
        scenario: {"echo": round(float(np.mean(value, axis=0)[0]), 4),
                   "deg": round(float(np.mean(value, axis=0)[1]), 4)}
        for scenario, value in rows.items()
    }
    gates = {"st_echo": scores["st"]["echo"] >= 4.5,
             "nst_deg": scores["nst"]["deg"] >= 3.8,
             "dt_echo": scores["dt"]["echo"] >= 4.0,
             "dt_deg": scores["dt"]["deg"] >= 3.8}
    result = {"step": step, "scores": scores, "gates": gates}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "eval"))
    parser.add_argument("--base-checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "base_gru.pt")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "phase_refiner_v4.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "phase_v4_score.json")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--segment-seconds", type=float, default=1.2)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
