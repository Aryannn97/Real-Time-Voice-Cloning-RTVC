"""
Post-processing for generated speech: hiss reduction, rumble removal, loudness leveling and
smooth joining of segments. Everything here works on float waveforms in [-1, 1] and only depends on
numpy and scipy.
"""
from typing import List

import numpy as np
from scipy import signal

from utils import logmmse


def _frame_rms(wav: np.ndarray, frame_len: int) -> np.ndarray:
    n_frames = len(wav) // frame_len
    frames = wav[:n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def remove_rumble(wav: np.ndarray, sample_rate: int, cutoff_hz: float = 70.0) -> np.ndarray:
    """High-pass filter that removes low frequency rumble and DC offset."""
    sos = signal.butter(4, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, wav)


def reduce_hiss(wav: np.ndarray, sample_rate: int, noise_fraction: float = 0.1,
                min_noise_seconds: float = 0.15) -> np.ndarray:
    """
    Suppresses the broadband hiss the vocoder leaves behind. The noise profile is estimated from
    the quietest frames of the waveform itself, so no separate noise sample is needed. Waveforms
    with too little quiet material to profile are returned untouched.
    """
    frame_len = int(0.02 * sample_rate)
    rms = _frame_rms(wav, frame_len)
    if len(rms) == 0:
        return wav

    n_noise_frames = max(1, int(len(rms) * noise_fraction))
    quietest = np.argsort(rms)[:n_noise_frames]
    if n_noise_frames * frame_len < min_noise_seconds * sample_rate:
        return wav

    noise = np.concatenate([wav[i * frame_len:(i + 1) * frame_len] for i in sorted(quietest)])
    profile = logmmse.profile_noise(noise, sample_rate)
    cleaned = logmmse.denoise(wav.astype(np.float64), profile, eta=0.15)
    return cleaned.astype(wav.dtype)


def level_loudness(wav: np.ndarray, target_rms_dbfs: float = -20.0, peak_ceiling: float = 0.97) -> np.ndarray:
    """
    Brings the speech to a consistent loudness, measured on the voiced frames only, then applies
    a soft limiter so peaks stay under the ceiling without hard clipping.
    """
    frame_len = 512
    rms = _frame_rms(wav, frame_len)
    if len(rms) == 0 or rms.max() == 0:
        return wav

    voiced = rms[rms > 0.1 * rms.max()]
    current_dbfs = 20 * np.log10(np.sqrt(np.mean(voiced ** 2)) + 1e-12)
    wav = wav * (10 ** ((target_rms_dbfs - current_dbfs) / 20))

    # Soft limiter: linear below the knee, smoothly saturating above it
    knee = 0.6 * peak_ceiling
    over = np.abs(wav) > knee
    excess = (np.abs(wav[over]) - knee) / (peak_ceiling - knee)
    wav[over] = np.sign(wav[over]) * (knee + (peak_ceiling - knee) * np.tanh(excess))
    return wav


def join_segments(segments: List[np.ndarray], sample_rate: int, gap_seconds: float = 0.18,
                  fade_seconds: float = 0.03) -> np.ndarray:
    """
    Concatenates segments with a short pause between them. Each segment is faded in and out so
    the joins never click, unlike inserting a hard cut to silence.
    """
    fade_len = int(fade_seconds * sample_rate)
    gap = np.zeros(int(gap_seconds * sample_rate), dtype=np.float64)

    pieces = []
    for i, seg in enumerate(segments):
        seg = np.array(seg, dtype=np.float64)
        n = min(fade_len, len(seg) // 2)
        if n > 0:
            ramp = np.linspace(0, 1, n)
            seg[:n] *= ramp
            seg[-n:] *= ramp[::-1]
        pieces.append(seg)
        if i < len(segments) - 1:
            pieces.append(gap)
    return np.concatenate(pieces) if pieces else np.zeros(0)


def enhance(wav: np.ndarray, sample_rate: int, denoise: bool = True, level: bool = True) -> np.ndarray:
    """Full cleanup chain applied to a generated waveform."""
    wav = np.asarray(wav, dtype=np.float64)
    if len(wav) == 0:
        return wav
    wav = remove_rumble(wav, sample_rate)
    if denoise:
        wav = reduce_hiss(wav, sample_rate)
    if level:
        wav = level_loudness(wav)
    return wav
