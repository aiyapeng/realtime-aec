"""评估并调整双讲场景后滤波参数。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import librosa
import numpy as np
import onnxruntime as ort
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from config import CFG
from dualpath_v3 import DualPathV3
from scene_loss import near_dominance_target


def _hz_to_mel_slaney(frequencies: np.ndarray) -> np.ndarray:
    """Librosa-compatible Slaney Hz-to-mel conversion without lazy imports."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    f_sp = 200.0 / 3.0
    mels = frequencies / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = frequencies >= min_log_hz
    mels[log_region] = (
        min_log_mel + np.log(frequencies[log_region] / min_log_hz) / logstep
    )
    return mels


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    """Librosa-compatible Slaney mel-to-Hz conversion."""
    mels = np.asarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3.0
    frequencies = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    frequencies[log_region] = (
        min_log_hz * np.exp(logstep * (mels[log_region] - min_log_mel))
    )
    return frequencies


def _librosa_compatible_mel_basis(
    sr: int = 16000, n_fft: int = 513, n_mels: int = 160
) -> np.ndarray:
    """Reproduce librosa.filters.mel defaults (Slaney scale and norm)."""
    fft_frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    min_mel, max_mel = _hz_to_mel_slaney(np.asarray([0.0, sr / 2.0]))
    mel_frequencies = _mel_to_hz_slaney(
        np.linspace(min_mel, max_mel, n_mels + 2)
    )
    ramps = np.subtract.outer(mel_frequencies, fft_frequencies)
    fdiff = np.diff(mel_frequencies)
    weights = np.zeros((n_mels, len(fft_frequencies)), dtype=np.float64)
    lower = -ramps[:-2] / fdiff[:-1, np.newaxis]
    upper = ramps[2:] / fdiff[1:, np.newaxis]
    weights[:] = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_frequencies[2:n_mels + 2] -
                       mel_frequencies[:n_mels]))[:, np.newaxis]
    return weights.astype(np.float32)


class CachedAECMOS:
    """Numerically identical feature path, with one reused ONNX session."""

    def __init__(self, model_path: Path, fast_mel: bool = False):
        self.model_path = str(model_path)
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.hidden = np.zeros((4, 1, 64), np.float32)
        self.fast_mel = bool(fast_mel)
        self.mel_basis = (_librosa_compatible_mel_basis()
                          if self.fast_mel else None)
        self.hann = (np.hanning(514)[:-1].astype(np.float32)
                     if self.fast_mel else None)

    def mel(self, x: np.ndarray) -> np.ndarray:
        if self.fast_mel:
            return self._fast_mel(x)
        spec = librosa.feature.melspectrogram(
            y=x,
            sr=16000,
            n_fft=513,
            hop_length=256,
            n_mels=160,
        )
        return ((librosa.power_to_db(spec, ref=np.max) + 40) / 40).T

    def _fast_mel(self, x: np.ndarray) -> np.ndarray:
        """Numerical equivalent of the AECMOS librosa feature path."""
        x = np.asarray(x, dtype=np.float32)
        padded = np.pad(x, (256, 256), mode="constant")
        frames = np.lib.stride_tricks.sliding_window_view(padded, 513)[::256]
        windowed = frames * self.hann[np.newaxis, :]
        spectrum = np.fft.rfft(windowed, n=513, axis=1).astype(np.complex64)
        power = (np.abs(spectrum) ** 2).astype(np.float32)
        spec = self.mel_basis @ power.T
        amin = np.float32(1e-10)
        log_spec = 10.0 * np.log10(np.maximum(amin, spec))
        log_spec -= 10.0 * np.log10(max(float(amin), float(np.max(spec))))
        log_spec = np.maximum(log_spec, float(np.max(log_spec)) - 80.0)
        return ((log_spec + 40.0) / 40.0).T

    @staticmethod
    def mark(x: np.ndarray, first: float) -> np.ndarray:
        return np.concatenate(
            (x, np.ones((20, x.shape[1])) * first, np.zeros((20, x.shape[1]))),
            axis=0,
        )

    def fixed_features(
        self, scenario: str, lpb: np.ndarray, mic: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ne_st, fe_st = ((1, 0) if scenario == "nst" else
                        (0, 1) if scenario == "st" else (0, 0))
        return (
            self.mark(self.mel(lpb), 1 - ne_st),
            self.mark(self.mel(mic), 1 - fe_st),
        )

    def score(
        self,
        fixed: tuple[np.ndarray, np.ndarray],
        enhanced: np.ndarray,
    ) -> tuple[float, float]:
        enh = self.mark(self.mel(enhanced), 1)
        feats = np.expand_dims(np.stack((fixed[0], fixed[1], enh)), 0).astype(
            np.float32
        )
        result = self.session.run(
            [], {self.input_name: feats, "h0": self.hidden}
        )[0]
        return float(result[0]), float(result[1])


def load_dev(checkpoint: Path):
    bank = np.load(ROOT / "data" / "bank_dev" / "bank.npy", mmap_mode="r")
    labels = np.load(ROOT / "data" / "bank_dev" / "scenario.npy")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = DualPathV3(
        n_freq=CFG.n_freq, hidden=96, num_layers=2, mask_mag_max=3.0
    )
    model.load_state_dict(ck["sd"])
    model.eval()
    window = torch.from_numpy(
        np.sqrt(np.hanning(CFG.n_fft + 1)[:-1]).astype(np.float32)
    )

    def stft(x):
        return torch.stft(
            x, CFG.n_fft, CFG.hop, CFG.n_fft, window,
            center=True, return_complex=True,
        ).transpose(1, 2)

    records = []
    with torch.no_grad():
        for idx in range(bank.shape[0]):
            seg = np.asarray(bank[idx], np.float32) / 32767.0
            err, ref, echo_hat, near = [
                torch.from_numpy(seg[j]).unsqueeze(0) for j in range(4)
            ]
            e_spec, x_spec, y_spec, s_spec = (
                stft(err), stft(ref), stft(echo_hat), stft(near)
            )
            mr, mi, p_ne, _ = model(
                e_spec.abs(), x_spec.abs(), y_spec.abs(), (e_spec + y_spec).abs()
            )
            dominance = near_dominance_target(e_spec, s_spec)
            records.append(
                {
                    "idx": idx,
                    "scenario": ("st", "nst", "dt")[int(labels[idx])],
                    "segment": seg,
                    "E": e_spec[0],
                    "mask": torch.complex(mr[0], mi[0]),
                    "p": p_ne[0],
                    "d": dominance[0],
                    "S": s_spec[0],
                    "window": window,
                }
            )
    return records


def diagnostics(records):
    out = {}
    for scenario in ("st", "nst", "dt"):
        rows = [r for r in records if r["scenario"] == scenario]
        p = torch.cat([r["p"].flatten() for r in rows])
        d = torch.cat([r["d"].flatten() for r in rows])
        mag = torch.cat([r["mask"].abs().flatten() for r in rows])
        energy = torch.cat([r["S"].abs().square().flatten() for r in rows])
        energy_sum = energy.sum() + 1e-12
        corr = float(torch.corrcoef(torch.stack((p, d)))[0, 1])
        out[scenario] = {
            "p_mean": float(p.mean()),
            "d_mean": float(d.mean()),
            "p_energy_weighted": float((p * energy).sum() / energy_sum),
            "d_energy_weighted": float((d * energy).sum() / energy_sum),
            "p_d_corr": corr,
            "mask_mag_mean": float(mag.mean()),
            "mask_below_0_5": float((mag < 0.5).float().mean()),
        }
    return out


def istft(spec: torch.Tensor, window: torch.Tensor, length: int) -> np.ndarray:
    return torch.istft(
        spec.transpose(0, 1), CFG.n_fft, CFG.hop, CFG.n_fft, window,
        center=True, length=length,
    ).numpy()


def evaluate_candidate(records, scorer, threshold, temperature, preserve, low_gain,
                       gate_key="p"):
    values = []
    for record in records:
        if record["scenario"] != "dt":
            continue
        q = preserve * torch.sigmoid((record[gate_key] - threshold) / temperature)
        effective = q + (1.0 - q) * low_gain * record["mask"]
        enhanced = istft(
            record["E"] * effective,
            record["window"],
            record["segment"].shape[1],
        )
        seg = record["segment"]
        fixed = record["fixed"]
        values.append(scorer.score(fixed, enhanced))
    mean = np.mean(np.asarray(values), axis=0)
    return float(mean[0]), float(mean[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "checkpoints" / "base_gru.pt"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "postfilter_sweep.json")
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--oracle-gate-probe", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    records = load_dev(args.checkpoint)
    diag = diagnostics(records)
    print(json.dumps(diag, indent=2), flush=True)
    if args.diagnose_only:
        return

    scorer = CachedAECMOS(
        ROOT / "third_party" / "aecmos" / "Run_1663915512_Stage_0.onnx"
    )
    for record in records:
        if record["scenario"] == "dt":
            seg = record["segment"]
            record["fixed"] = scorer.fixed_features("dt", seg[1], seg[0] + seg[2])

    if args.oracle_gate_probe:
        probes = []
        for gate_key in ("p", "d"):
            for threshold in (0.25, 0.5, 0.75):
                for temperature in (0.03, 0.1, 0.25):
                    echo, deg = evaluate_candidate(
                        records, scorer, threshold, temperature, 1.0, 0.25,
                        gate_key=gate_key,
                    )
                    probes.append({"gate": gate_key, "threshold": threshold,
                                   "temperature": temperature,
                                   "dt_echo": round(echo, 4),
                                   "dt_deg": round(deg, 4)})
        # Exact oracle complex mask is included as the upper-bound control.
        oracle = []
        oracle_mag = []
        oracle_wiener = []
        for record in records:
            if record["scenario"] != "dt":
                continue
            mask = record["S"] / (record["E"] + 1e-8)
            mask = torch.polar(mask.abs().clamp(max=3.0), torch.angle(mask))
            enhanced = istft(record["E"] * mask, record["window"],
                             record["segment"].shape[1])
            oracle.append(scorer.score(record["fixed"], enhanced))
            mag_only = mask.abs().clamp(max=3.0)
            enhanced_mag = istft(record["E"] * mag_only, record["window"],
                                 record["segment"].shape[1])
            oracle_mag.append(scorer.score(record["fixed"], enhanced_mag))
            residual = record["E"] - record["S"]
            wiener = record["S"].abs().square() / (
                record["S"].abs().square() + residual.abs().square() + 1e-8)
            enhanced_wiener = istft(record["E"] * wiener, record["window"],
                                    record["segment"].shape[1])
            oracle_wiener.append(scorer.score(record["fixed"], enhanced_wiener))
        oracle = np.mean(np.asarray(oracle), axis=0)
        oracle_mag = np.mean(np.asarray(oracle_mag), axis=0)
        oracle_wiener = np.mean(np.asarray(oracle_wiener), axis=0)
        probes.append({"gate": "complex_oracle", "dt_echo": round(float(oracle[0]), 4),
                       "dt_deg": round(float(oracle[1]), 4)})
        probes.append({"gate": "magnitude_oracle", "dt_echo": round(float(oracle_mag[0]), 4),
                       "dt_deg": round(float(oracle_mag[1]), 4)})
        probes.append({"gate": "wiener_oracle", "dt_echo": round(float(oracle_wiener[0]), 4),
                       "dt_deg": round(float(oracle_wiener[1]), 4)})
        print(json.dumps(probes, indent=2), flush=True)
        return

    # Coarse grid, intentionally small enough for rapid iteration.
    candidates = []
    for threshold in (0.25, 0.4, 0.55, 0.7):
        for temperature in (0.05, 0.12, 0.25):
            for preserve in (0.35, 0.65, 1.0):
                for low_gain in (0.55, 0.75, 1.0):
                    echo, deg = evaluate_candidate(
                        records, scorer, threshold, temperature, preserve, low_gain
                    )
                    row = {
                        "threshold": threshold,
                        "temperature": temperature,
                        "preserve": preserve,
                        "low_gain": low_gain,
                        "dt_echo": round(echo, 4),
                        "dt_deg": round(deg, 4),
                    }
                    candidates.append(row)
                    print(json.dumps(row), flush=True)
    candidates.sort(
        key=lambda r: (min(r["dt_echo"] - 4.0, r["dt_deg"] - 3.8),
                       r["dt_echo"] + r["dt_deg"]),
        reverse=True,
    )
    result = {"diagnostics": diag, "best": candidates[:15], "all": candidates}
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("BEST", json.dumps(candidates[:5], indent=2), flush=True)


if __name__ == "__main__":
    main()
