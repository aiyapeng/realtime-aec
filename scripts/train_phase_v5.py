"""Train and score the time-frequency preserving V5 refiner."""
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

from config import CFG
from dualpath_v3 import DualPathV3
from phase_aware_v5 import PhaseAwareConvV5
from train_phase_v4 import window, stft, istft, loss_v4
from optimize_dt_postfilter import CachedAECMOS
from phase_aware_v4 import clipped_oracle_mask
from scene_loss import _compress, near_dominance_target


def near_fidelity_loss(mask, p_ne, E, S):
    """Fine-tune the already echo-safe model toward DT speech naturalness."""
    target = clipped_oracle_mask(E, S)
    dominance = near_dominance_target(E, S).detach()
    error = (mask - target).abs().square()
    near_mask = (dominance * error).sum() / (dominance.sum() + 1e-8)
    echo_mask = ((1.0 - dominance) * error).sum() / (
        (1.0 - dominance).sum() + 1e-8)
    enhanced = E * mask
    cm_e, cc_e = _compress(enhanced, 0.3)
    cm_s, cc_s = _compress(S, 0.3)
    recon_tf = (cm_e - cm_s).square() + (cc_e - cc_s).abs().square()
    near_recon = (dominance * recon_tf).sum() / (dominance.sum() + 1e-8)
    phase_weight = dominance * S.abs().pow(0.6).detach()
    phase = (phase_weight * (1.0 - torch.cos(
        torch.angle(mask) - torch.angle(target)))).sum() / (phase_weight.sum() + 1e-8)
    evidence = torch.nn.functional.binary_cross_entropy(
        p_ne.clamp(1e-6, 1 - 1e-6), dominance)
    total = 4.0 * near_mask + 4.0 * near_recon + 1.5 * phase + 0.75 * echo_mask + 0.2 * evidence
    return total, {"near_mask": float(near_mask.detach()),
                   "near_recon": float(near_recon.detach()),
                   "phase": float(phase.detach()),
                   "echo_mask": float(echo_mask.detach()),
                   "evidence": float(evidence.detach())}


def load_model(base_path, checkpoint, device):
    base = DualPathV3(n_freq=CFG.n_freq, hidden=96, num_layers=2, mask_mag_max=3.0)
    base.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=False)["sd"])
    model = PhaseAwareConvV5(base, channels=64)
    step, optimizer_state = 0, None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"])
        step = int(saved["step"])
        optimizer_state = saved.get("optimizer")
    return model.to(device), step, optimizer_state


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model, start, optimizer_state = load_model(
        args.base_checkpoint, args.checkpoint if args.resume else Path("__missing__"), device
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    bank = np.load(args.train_bank_dir / "bank.npy", mmap_mode="r")
    labels = np.load(args.train_bank_dir / "scenario.npy")
    indices = np.flatnonzero(labels == 2)
    win = window(device)
    crop = int(args.segment_seconds * CFG.fs)
    model.train(); started = time.time()
    for step in range(start, args.steps):
        rng = np.random.default_rng(args.seed + step * 31)
        selected = rng.choice(indices, size=args.batch_size, replace=True)
        rows = []
        for idx in selected:
            segment = np.asarray(bank[int(idx)], np.float32) / 32767.0
            begin = int(rng.integers(0, segment.shape[1] - crop + 1))
            rows.append(segment[:, begin:begin + crop] * rng.uniform(0.65, 1.35))
        batch = torch.from_numpy(np.stack(rows).astype(np.float32)).to(device)
        err, ref, echo_hat, near = [batch[:, i] for i in range(4)]
        E, X, Y, S = [stft(x, win) for x in (err, ref, echo_hat, near)]
        mr, mi, p, _ = model(E, X, Y, E + Y, force_dt=True)
        mask = torch.complex(mr.float(), mi.float())
        loss, parts = (near_fidelity_loss(mask, p, E, S)
                       if args.near_finetune else loss_v4(mask, p, E, S))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0)
        optimizer.step()
        if step % 100 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach()), **parts,
                              "seconds": round(time.time() - started, 1)}), flush=True)
        if (step + 1) % args.save_every == 0:
            torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step + 1, "seed": args.seed}, args.checkpoint)
    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": args.steps, "seed": args.seed}, args.checkpoint)


def evaluate(args):
    device = torch.device(args.device)
    model, step, _ = load_model(args.base_checkpoint, args.checkpoint, device)
    model.eval(); win = window(device)
    bank = np.load(args.bank_dir / "bank.npy", mmap_mode="r")
    labels = np.load(args.bank_dir / "scenario.npy")
    scorer = CachedAECMOS(ROOT / "third_party" / "aecmos" /
                          "Run_1663915512_Stage_0.onnx")
    names = ("st", "nst", "dt"); rows = {name: [] for name in names}
    gate_values = {name: [] for name in names}
    with torch.no_grad():
        for idx in range(bank.shape[0]):
            seg = np.asarray(bank[idx], np.float32) / 32767.0
            batch = torch.from_numpy(seg).unsqueeze(0).to(device)
            err, ref, echo_hat = [batch[:, i] for i in range(3)]
            E, X, Y = [stft(x, win) for x in (err, ref, echo_hat)]
            mr, mi, p_ne, gate = model(
                E, X, Y, E + Y,
                force_dt=bool(args.locked_scenario and int(labels[idx]) == 2),
            )
            mask = torch.complex(mr, mi)
            if args.phase_scale != 1.0 or args.magnitude_power != 1.0:
                magnitude = mask.abs().clamp(min=1e-6).pow(args.magnitude_power)
                mask = torch.polar(magnitude.clamp(max=3.0),
                                   torch.angle(mask) * args.phase_scale)
            if args.identity_blend > 0:
                amount = args.identity_blend * gate
                mask = (1.0 - amount) * mask + amount
            if args.evidence_blend > 0:
                # Preserve the linear residual only in TF bins where the refiner
                # predicts near-end dominance.  Unlike a global identity blend,
                # this leaves echo-dominant bins under full suppression.
                amount = args.evidence_blend * p_ne.pow(args.evidence_power) * gate
                mask = (1.0 - amount) * mask + amount
            enhanced = istft(E * mask, win, err.shape[1])[0].cpu().numpy()
            scenario = names[int(labels[idx])]
            gate_values[scenario].append(float(gate.mean().cpu()))
            fixed = scorer.fixed_features(scenario, seg[1], seg[0] + seg[2])
            rows[scenario].append(scorer.score(fixed, enhanced))
    scores = {name: {"echo": round(float(np.mean(value, axis=0)[0]), 4),
                     "deg": round(float(np.mean(value, axis=0)[1]), 4)}
              for name, value in rows.items()}
    gates = {"st_echo": scores["st"]["echo"] >= 4.5,
             "nst_deg": scores["nst"]["deg"] >= 3.8,
             "dt_echo": scores["dt"]["echo"] >= 4.0,
             "dt_deg": scores["dt"]["deg"] >= 3.8}
    gate_diagnostics = {
        name: {"active": int(np.sum(np.asarray(values) > 0.5)),
               "n": len(values), "min": round(float(np.min(values)), 4),
               "max": round(float(np.max(values)), 4)}
        for name, values in gate_values.items()
    }
    result = {"step": step, "scores": scores, "gates": gates,
              "gate_diagnostics": gate_diagnostics}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "eval"))
    parser.add_argument("--base-checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "base_gru.pt")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "phase_refiner_v5.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "phase_v5_score.json")
    parser.add_argument("--bank-dir", type=Path,
                        default=ROOT / "data" / "bank_dev")
    parser.add_argument("--train-bank-dir", type=Path,
                        default=ROOT / "data" / "bank_train")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=1.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260827)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--locked-scenario", action="store_true",
                        help="force the DT refiner only for locked DT-labelled eval records")
    parser.add_argument("--identity-blend", type=float, default=0.0,
                        help="DT-gated identity residual returned after complex refinement")
    parser.add_argument("--evidence-blend", type=float, default=0.0,
                        help="maximum DT identity blend weighted by predicted near evidence")
    parser.add_argument("--evidence-power", type=float, default=2.0,
                        help="selectivity exponent for --evidence-blend")
    parser.add_argument("--near-finetune", action="store_true")
    parser.add_argument("--phase-scale", type=float, default=1.0)
    parser.add_argument("--magnitude-power", type=float, default=1.0)
    args = parser.parse_args()
    if args.command == "train": train(args)
    else: evaluate(args)


if __name__ == "__main__":
    main()
