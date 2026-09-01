"""Initialize, distill, train and score the causal streaming V6 refiner."""
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from config import CFG
from dualpath_v3 import DualPathV3
from phase_aware_v5 import PhaseAwareConvV5
from train_phase_v5 import near_fidelity_loss
from phase_aware_v6 import PhaseAwareCausalV6
from train_phase_v4 import window, stft, istft, loss_v4
from optimize_dt_postfilter import CachedAECMOS


def new_base(path: Path):
    base = DualPathV3(n_freq=CFG.n_freq, hidden=96, num_layers=2,
                      mask_mag_max=3.0)
    base.load_state_dict(torch.load(path, map_location="cpu",
                                    weights_only=False)["sd"])
    return base


def load_v6(args, device):
    saved = (torch.load(args.checkpoint, map_location="cpu", weights_only=False)
             if args.checkpoint.exists() else None)
    model = PhaseAwareCausalV6(new_base(args.base_checkpoint), channels=64,
                               fs=CFG.fs, hop=CFG.hop,
                               lookahead_frames=args.lookahead_frames,
                               scene_gate_enabled=bool(
                                   saved and saved.get("scene_gate_enabled", False)),
                               scene_hidden=int(saved.get("scene_hidden", 48))
                               if saved else 48,
                               scene_threshold=float(saved.get("scene_threshold", 0.5))
                               if saved else 0.985,
                               scene_low_threshold=float(saved.get(
                                   "scene_low_threshold", 0.4)) if saved else 0.4,
                               scene_confirm_frames=int(saved.get(
                                   "scene_confirm_frames", 8)) if saved else 8,
                               scene_off_threshold=float(saved.get(
                                   "scene_off_threshold", 0.01)) if saved else 0.01,
                               scene_release_frames=int(saved.get(
                                   "scene_release_frames", 16)) if saved else 16,
                               scene_soft_gate=bool(saved.get(
                                   "scene_soft_gate", False)) if saved else False,
                               scene_soft_power=float(saved.get(
                                   "scene_soft_power", 1.0)) if saved else 1.0,
                               scene_soft_floor=float(saved.get(
                                   "scene_soft_floor", 0.0)) if saved else 0.0)
    step, optimizer_state = 0, None
    if saved is not None:
        model.load_state_dict(saved["state_dict"])
        step = int(saved["step"])
        optimizer_state = saved.get("optimizer")
    else:
        source = torch.load(args.init_v5, map_location="cpu", weights_only=False)
        # Module names and tensor shapes intentionally match V5.  Conv kernels and
        # affine norm parameters transfer exactly; execution semantics become causal.
        model.load_state_dict(source["state_dict"], strict=True)
    return model.to(device), step, optimizer_state


def load_teacher(args, device):
    teacher = PhaseAwareConvV5(new_base(args.base_checkpoint), channels=64)
    teacher.load_state_dict(torch.load(args.init_v5, map_location="cpu",
                                       weights_only=False)["state_dict"])
    return teacher.to(device).eval()


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, start, optimizer_state = load_v6(args, device)
    teacher = load_teacher(args, device) if args.distill_weight > 0 else None
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    bank = np.load(args.train_bank_dir / "bank.npy", mmap_mode="r")
    labels = np.load(args.train_bank_dir / "scenario.npy")
    indices = (np.flatnonzero(labels == 2) if args.train_scenarios == "dt" else
               np.flatnonzero(labels != 1))
    win = window(device)
    crop = int(args.segment_seconds * CFG.fs)
    started = time.time()
    for step in range(start, args.steps):
        rng = np.random.default_rng(args.seed + step * 37)
        selected = rng.choice(indices, size=args.batch_size, replace=True)
        rows = []
        for idx in selected:
            segment = np.asarray(bank[int(idx)], np.float32) / 32767.0
            begin = int(rng.integers(0, segment.shape[1] - crop + 1))
            rows.append(segment[:, begin:begin + crop] * rng.uniform(0.65, 1.35))
        batch = torch.from_numpy(np.stack(rows).astype(np.float32)).to(device)
        err, ref, echo_hat, near = [batch[:, i] for i in range(4)]
        E, X, Y, S = [stft(value, win) for value in (err, ref, echo_hat, near)]
        mr, mi, p_ne, _ = model(E, X, Y, E + Y, force_dt=True)
        mask = torch.complex(mr.float(), mi.float())
        loss, parts = loss_v4(mask, p_ne, E, S)
        if args.near_aux_weight > 0:
            near_loss, near_parts = near_fidelity_loss(mask, p_ne, E, S)
            loss = loss + args.near_aux_weight * near_loss
            parts.update({f"near_{key}": value for key, value in near_parts.items()})
        if teacher is not None:
            with torch.no_grad():
                tr, ti, _, _ = teacher(
                    E, X, Y, E + Y,
                    force_dt=args.train_scenarios == "dt")
                teacher_mask = torch.complex(tr.float(), ti.float())
            weight = (S.abs() + (E - S).abs() + 1e-4).pow(0.3).detach()
            distill = (weight * (mask - teacher_mask).abs().square()).sum() / (
                weight.sum() + 1e-8)
            loss = loss + args.distill_weight * distill
            parts["distill"] = float(distill.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0)
        optimizer.step()
        if step % 100 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach()), **parts,
                              "seconds": round(time.time() - started, 1)}), flush=True)
        if (step + 1) % args.save_every == 0:
            torch.save({"state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "step": step + 1,
                        "seed": args.seed, "causal": True,
                        "history_frames": model.history_frames,
                        "lookahead_frames": model.lookahead_frames,
                        "scene_gate_enabled": model.scene_gate_enabled,
                        "scene_hidden": model.scene_hidden,
                        "scene_threshold": model.scene_threshold,
                        "scene_low_threshold": model.scene_low_threshold,
                        "scene_confirm_frames": model.scene_confirm_frames,
                        "scene_off_threshold": model.scene_off_threshold,
                        "scene_release_frames": model.scene_release_frames}, args.checkpoint)
    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": args.steps, "seed": args.seed, "causal": True,
                "history_frames": model.history_frames,
                "lookahead_frames": model.lookahead_frames,
                "scene_gate_enabled": model.scene_gate_enabled,
                "scene_hidden": model.scene_hidden,
                "scene_threshold": model.scene_threshold,
                "scene_low_threshold": model.scene_low_threshold,
                "scene_confirm_frames": model.scene_confirm_frames,
                "scene_off_threshold": model.scene_off_threshold,
                "scene_release_frames": model.scene_release_frames}, args.checkpoint)


def stream_model(model, E, X, Y, chunk_frames, force_dt=False,
                 incremental_refiner=False):
    init_lookahead = (model.init_incremental_lookahead_state
                      if incremental_refiner else model.init_lookahead_state)
    state = (init_lookahead(batch=E.shape[0], device=E.device)
             if model.lookahead_frames else
             model.init_stream_state(batch=E.shape[0], device=E.device))
    outputs = []
    gates = []
    evidence = []
    for begin in range(0, E.shape[1], chunk_frames):
        end = min(begin + chunk_frames, E.shape[1])
        method = model.lookahead_push if model.lookahead_frames else model.stream_step
        mr, mi, p_chunk, gate, state = method(
            E[:, begin:end], X[:, begin:end], Y[:, begin:end],
            E[:, begin:end] + Y[:, begin:end], state, force_dt=force_dt)
        outputs.append(torch.complex(mr, mi))
        gates.append(gate)
        evidence.append(p_chunk)
    if model.lookahead_frames:
        empty = E[:, :0]
        mr, mi, p_chunk, gate, state = model.lookahead_push(
            empty, empty, empty, empty, state, final=True, force_dt=force_dt)
        outputs.append(torch.complex(mr, mi))
        gates.append(gate)
        evidence.append(p_chunk)
    return (torch.cat(outputs, dim=1), torch.cat(gates, dim=1),
            torch.cat(evidence, dim=1))


def evaluate(args):
    device = torch.device(args.device)
    model, step, _ = load_v6(args, device)
    model.eval()
    win = window(device)
    bank = np.load(args.bank_dir / "bank.npy", mmap_mode="r")
    labels = np.load(args.bank_dir / "scenario.npy")
    scorer = CachedAECMOS(ROOT / "third_party" / "aecmos" /
                          "Run_1663915512_Stage_0.onnx",
                          fast_mel=args.fast_aecmos)
    names = ("st", "nst", "dt")
    rows = {name: [] for name in names}
    gate_values = {name: [] for name in names}
    with torch.no_grad():
        for idx in range(bank.shape[0]):
            seg = np.asarray(bank[idx], np.float32) / 32767.0
            batch = torch.from_numpy(seg).unsqueeze(0).to(device)
            err, ref, echo_hat = [batch[:, i] for i in range(3)]
            E, X, Y = [stft(value, win) for value in (err, ref, echo_hat)]
            label = int(labels[idx])
            force_dt = bool((args.locked_scenario and label == 2) or
                            (args.force_far and label in (0, 2)))
            if args.chunk_frames > 0:
                mask, gate, p_ne = stream_model(
                    model, E, X, Y, args.chunk_frames, force_dt=force_dt,
                    incremental_refiner=args.incremental_refiner)
            else:
                mr, mi, p_ne, gate = model(E, X, Y, E + Y, force_dt=force_dt)
                mask = torch.complex(mr, mi)
            if args.evidence_blend > 0:
                amount = args.evidence_blend * p_ne.pow(args.evidence_power) * gate
                mask = (1.0 - amount) * mask + amount
            enhanced = istft(E * mask, win, err.shape[1])[0].cpu().numpy()
            scenario = names[int(labels[idx])]
            gate_values[scenario].append(float(gate.mean().cpu()))
            fixed = scorer.fixed_features(scenario, seg[1], seg[0] + seg[2])
            rows[scenario].append(scorer.score(fixed, enhanced))
            if args.eval_progress > 0 and (idx + 1) % args.eval_progress == 0:
                print(json.dumps({"evaluated": idx + 1,
                                  "total": int(bank.shape[0])}), flush=True)
    scores = {name: {"echo": round(float(np.mean(value, axis=0)[0]), 4),
                     "deg": round(float(np.mean(value, axis=0)[1]), 4)}
              for name, value in rows.items()}
    gates = {"st_echo": scores["st"]["echo"] >= 4.5,
             "nst_deg": scores["nst"]["deg"] >= 3.8,
             "dt_echo": scores["dt"]["echo"] >= 4.0,
             "dt_deg": scores["dt"]["deg"] >= 3.8}
    diagnostics = {
        name: {"mean_active_fraction": round(float(np.mean(values)), 4),
               "clips_active": int(np.sum(np.asarray(values) > 0.01)),
               "n": len(values)} for name, values in gate_values.items()
    }
    result = {"v6_step": step, "causal": True,
              "chunk_frames": args.chunk_frames,
              "incremental_refiner": args.incremental_refiner,
              "scores": scores,
              "gates": gates, "gate_diagnostics": diagnostics,
              "aecmos_feature_path": ("fast_librosa_equivalent"
                                        if args.fast_aecmos else "official_librosa")}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "eval"))
    parser.add_argument("--base-checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "base_gru.pt")
    parser.add_argument("--init-v5", type=Path,
                        default=ROOT / "checkpoints" / "phase_refiner_v5.pt")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "phase_refiner.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "phase_refiner_score.json")
    parser.add_argument("--bank-dir", type=Path, default=ROOT / "data" / "bank_dev")
    parser.add_argument("--train-bank-dir", type=Path,
                        default=ROOT / "data" / "bank_train")
    parser.add_argument("--train-scenarios", choices=("dt", "far"), default="dt",
                        help="dt trains only double-talk; far jointly trains ST and DT")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=1.5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--near-aux-weight", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260828)
    parser.add_argument("--chunk-frames", type=int, default=16,
                        help="0 uses one full causal call; positive values exercise state caches")
    parser.add_argument("--incremental-refiner", action="store_true",
                        help="use exact layer-local convolution caches for lookahead CPU runtime")
    parser.add_argument("--lookahead-frames", type=int, choices=(0, 4, 8, 16), default=0)
    parser.add_argument("--locked-scenario", action="store_true",
                        help="diagnostic only: force V6 on for labelled DT records")
    parser.add_argument("--force-far", action="store_true",
                        help="diagnostic only: force V6 on for labelled ST and DT records")
    parser.add_argument("--evidence-blend", type=float, default=0.0)
    parser.add_argument("--evidence-power", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fast-aecmos", action="store_true")
    parser.add_argument("--eval-progress", type=int, default=10)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
