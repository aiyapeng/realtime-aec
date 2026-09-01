"""Run the bundled synthetic double-talk example through the streaming AEC."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from scipy import signal


DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent
sys.path.insert(0, str(ROOT))

from scripts.v8_cpu_runtime import create_pipeline


SAMPLE_RATE = 16000


def load_mono(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path.name}: expected {SAMPLE_RATE} Hz, got {sample_rate}")
    return audio.mean(axis=1)


def repeat_to_length(audio: np.ndarray, length: int) -> np.ndarray:
    repeats = int(np.ceil(length / len(audio)))
    return np.tile(audio, repeats)[:length]


def normalize_peak(audio: np.ndarray, peak: float) -> np.ndarray:
    return audio * (peak / max(float(np.max(np.abs(audio))), 1e-8))


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-8)
    target = scale * reference
    residual = estimate - target
    return float(10 * np.log10((np.sum(target**2) + 1e-8) /
                               (np.sum(residual**2) + 1e-8)))


def main() -> None:
    duration_seconds = 7.0
    sample_count = int(duration_seconds * SAMPLE_RATE)
    near_start = int(2.0 * SAMPLE_RATE)

    far_source = load_mono(DEMO_DIR / "far_end.wav")
    near_source = load_mono(DEMO_DIR / "near_end_reference.wav")
    far_end = normalize_peak(repeat_to_length(far_source, sample_count), 0.55)
    near_end = np.zeros(sample_count, dtype=np.float32)
    near_length = min(len(near_source), sample_count - near_start)
    near_end[near_start:near_start + near_length] = normalize_peak(
        near_source[:near_length], 0.38
    )

    echo_path = np.zeros(1024, dtype=np.float32)
    echo_path[256] = 0.62
    echo_path[416] = 0.24
    echo_path[704] = -0.12
    linear_echo = signal.lfilter(echo_path, [1.0], far_end).astype(np.float32)
    echo = linear_echo + 0.025 * np.tanh(3.0 * linear_echo)
    rng = np.random.default_rng(20260901)
    microphone = near_end + echo + rng.normal(0.0, 8e-4, sample_count)
    microphone = np.clip(microphone, -0.98, 0.98).astype(np.float32)

    pipeline = create_pipeline()
    raw_output = pipeline.process_stream(far_end, microphone).astype(np.float32)
    latency = int(pipeline.latency_samples)
    output = np.pad(raw_output[latency:], (0, latency))[:sample_count]

    sf.write(DEMO_DIR / "far_end_reference.wav", far_end, SAMPLE_RATE, subtype="PCM_16")
    sf.write(DEMO_DIR / "near_end_timeline.wav", near_end, SAMPLE_RATE, subtype="PCM_16")
    sf.write(DEMO_DIR / "microphone_input.wav", microphone, SAMPLE_RATE, subtype="PCM_16")
    sf.write(DEMO_DIR / "aec_output.wav", output, SAMPLE_RATE, subtype="PCM_16")

    far_slice = slice(int(0.8 * SAMPLE_RATE), near_start)
    double_talk_slice = slice(near_start, near_start + near_length)
    erle = 10 * np.log10(
        (np.mean(microphone[far_slice] ** 2) + 1e-8)
        / (np.mean(output[far_slice] ** 2) + 1e-8)
    )
    metrics = {
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": duration_seconds,
        "pipeline_latency_ms": latency / SAMPLE_RATE * 1000.0,
        "synthetic_far_end_erle_db": float(erle),
        "double_talk_output_si_sdr_db": si_sdr(
            output[double_talk_slice], near_end[double_talk_slice]
        ),
        "finite_output": bool(np.isfinite(output).all()),
    }
    (DEMO_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
