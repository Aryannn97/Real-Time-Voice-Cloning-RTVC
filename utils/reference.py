"""
Preparation and quality checks for the recording a voice is cloned from. A clean, steady
recording of at least ten seconds matters more for the result than any model setting.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import librosa
import numpy as np

from utils import postprocess

RECOMMENDED_SECONDS = 10.0
MIN_SPEECH_SECONDS = 6.0


@dataclass
class ReferenceReport:
    speech_seconds: float
    level_dbfs: float
    snr_db: float
    clipped_fraction: float
    issues: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def summary(self) -> str:
        if self.ok:
            return "Good recording (%.0f s of speech, %.0f dB signal-to-noise)" % (
                self.speech_seconds, self.snr_db)
        return " ".join(self.issues)


def _frame_rms(wav: np.ndarray, sample_rate: int, frame_seconds: float = 0.03) -> np.ndarray:
    frame_len = int(frame_seconds * sample_rate)
    n_frames = len(wav) // frame_len
    if n_frames == 0:
        return np.zeros(1)
    frames = wav[:n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def analyze_reference(wav: np.ndarray, sample_rate: int) -> ReferenceReport:
    """Measures how usable a recording is as a voice reference."""
    wav = np.asarray(wav, dtype=np.float64)
    if len(wav) == 0 or np.abs(wav).max() == 0:
        return ReferenceReport(0.0, -120.0, 0.0, 0.0, ["No sound was recorded. Check your microphone."])

    rms = _frame_rms(wav, sample_rate)
    peak_rms = np.percentile(rms, 90)
    speech_frames = rms > 0.25 * peak_rms
    speech_seconds = float(speech_frames.sum() * 0.03)
    level_dbfs = float(20 * np.log10(np.sqrt(np.mean(rms[speech_frames] ** 2)) + 1e-12))
    noise_rms = np.percentile(rms, 10) + 1e-9
    snr_db = float(20 * np.log10(peak_rms / noise_rms))
    clipped = float(np.mean(np.abs(wav) > 0.99))

    issues = []
    if speech_seconds < MIN_SPEECH_SECONDS:
        issues.append("Only %.0f s of speech was captured. Keep talking for the full %.0f seconds." %
                      (speech_seconds, RECOMMENDED_SECONDS))
    if level_dbfs < -38:
        issues.append("The recording is very quiet. Move closer to the microphone.")
    if clipped > 0.005:
        issues.append("The recording is distorted. Speak a little further from the microphone.")
    if snr_db < 15:
        issues.append("There is a lot of background noise. Try a quieter room.")
    return ReferenceReport(speech_seconds, level_dbfs, snr_db, clipped, issues)


def prepare_reference(wav: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int, ReferenceReport]:
    """
    Cleans a recording for cloning: mono, rumble and hiss removed, leading and trailing silence
    trimmed and the level evened out. Returns the cleaned audio, its sample rate and a quality
    report measured on the original recording.
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    report = analyze_reference(wav, sample_rate)
    if len(wav) == 0 or np.abs(wav).max() == 0:
        return wav, sample_rate, report

    cleaned = postprocess.remove_rumble(wav.astype(np.float64), sample_rate)
    cleaned = postprocess.reduce_hiss(cleaned, sample_rate)
    cleaned, _ = librosa.effects.trim(cleaned, top_db=30)
    cleaned = postprocess.level_loudness(cleaned, target_rms_dbfs=-20.0, peak_ceiling=0.95)
    return cleaned.astype(np.float32), sample_rate, report
