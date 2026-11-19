import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import postprocess

SR = 16000


def _tone_with_hiss(seconds=2.0, hiss=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    tone[: SR // 2] = 0.0  # a stretch of silence to profile the noise from
    return tone + hiss * rng.standard_normal(len(t))


def test_enhance_keeps_length_and_ceiling():
    out = postprocess.enhance(_tone_with_hiss(), SR)
    assert len(out) == int(2.0 * SR)
    assert np.abs(out).max() <= 0.97 + 1e-6


def test_reduce_hiss_lowers_noise_floor():
    wav = _tone_with_hiss()
    cleaned = postprocess.reduce_hiss(wav, SR)
    assert np.std(cleaned[: SR // 4]) < np.std(wav[: SR // 4])


def test_join_segments_has_gap_and_no_hard_edges():
    segs = [np.ones(SR), np.ones(SR)]
    out = postprocess.join_segments(segs, SR, gap_seconds=0.2, fade_seconds=0.05)
    assert len(out) == 2 * SR + int(0.2 * SR)
    assert out[0] == 0 and out[-1] == 0


def test_silence_does_not_crash():
    postprocess.enhance(np.zeros(SR), SR)
    postprocess.enhance(np.zeros(0), SR)
